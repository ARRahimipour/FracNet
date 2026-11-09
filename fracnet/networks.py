"""Three FracNet CNN backbones, attention, domain SSL and Grad-CAM."""
import math

import timm
import torch
from torch import nn
from torch.nn import functional as F


class SpatialAttention(nn.Module):
    """Nonlocal spatial attention over positions in the final CNN feature map."""
    def __init__(self, channels):
        super().__init__()
        reduced = max(8, channels // 8)
        self.q = nn.Conv2d(channels, reduced, 1)
        self.k = nn.Conv2d(channels, reduced, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(x).flatten(2)
        # The N-by-N matrix relates spatial positions at the CNN's final scale.
        weights = torch.softmax(q @ k / math.sqrt(k.shape[1]), dim=-1)
        v = self.v(x).flatten(2)
        y = (v @ weights.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.scale * y


class ChannelSE(nn.Module):
    """Squeeze-and-Excitation channel recalibration with a reduction bottleneck."""
    def __init__(self, channels):
        super().__init__()
        mid = max(8, channels // 16)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, mid, 1),
                                  nn.ReLU(), nn.Conv2d(mid, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class FracCNN(nn.Module):
    """CNN backbone + optional attention + pooled binary classification head."""
    def __init__(self, name, pretrained=False, attention=True):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
        channels = self.backbone.num_features
        self.attn = SpatialAttention(channels) if attention else nn.Identity()
        self.se = ChannelSE(channels) if attention else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, 1)

    def feature_map(self, x):
        map_ = self.backbone(x)
        if map_.ndim != 4:
            raise RuntimeError("Expected spatial BCHW feature map; check timm model name")
        return self.se(self.attn(map_))

    def features(self, x):
        return self.pool(self.feature_map(x)).flatten(1)

    def forward(self, x):
        return self.head(self.features(x)).flatten()


def freeze_fraction(model, fraction):
    leaves = [m for m in model.backbone.modules()
              if not list(m.children()) and any(True for _ in m.parameters(recurse=False))]
    n = int(len(leaves) * fraction)
    for layer in leaves[:n]:
        for parameter in layer.parameters(recurse=False):
            parameter.requires_grad_(False)
    return n, len(leaves)


class SSLModel(nn.Module):
    """Contrastive pretraining proxy with a projection head discarded later."""
    def __init__(self, model):
        super().__init__()
        self.encoder = model.backbone
        d = self.encoder.num_features
        self.project = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, x):
        fmap = self.encoder(x)
        return F.normalize(self.project(F.adaptive_avg_pool2d(fmap, 1).flatten(1)), dim=1)


def contrastive_loss(first, second, temperature):
    """NT-Xent loss with positive pairs from two augmented image views."""
    n = first.shape[0]
    if n < 2:
        raise ValueError("SSL needs two or more samples per batch")
    vectors = torch.cat((first, second))
    scores = vectors @ vectors.T / temperature
    scores.fill_diagonal_(-1e9)
    expected = (torch.arange(2 * n, device=vectors.device) + n) % (2 * n)
    return F.cross_entropy(scores, expected)


def grad_cam(model, x, positive=True):
    """Generate a model-sensitivity heatmap, not a validated fracture mask."""
    model.eval()
    model.zero_grad(set_to_none=True)
    maps = model.feature_map(x)
    maps.retain_grad()
    score = model.head(model.pool(maps).flatten(1)).flatten()[0]
    (score if positive else -score).backward()
    weight = maps.grad.mean((2, 3), keepdim=True)
    heat = F.relu((weight * maps).sum(1, keepdim=True))
    heat = F.interpolate(heat, x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    heat = heat.detach().cpu().float()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat.numpy(), float(torch.sigmoid(score).detach())

