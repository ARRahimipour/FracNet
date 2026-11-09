import unittest
import numpy as np

from fracnet.classical import FeatureProcessor, metrics


class TestClassical(unittest.TestCase):
    def test_metrics(self):
        result = metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["specificity"], .5)
        self.assertEqual(result["recall"], 1)

    def test_train_only_feature_selection(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(150, 20)).astype("float32")
        x[:, 1] = x[:, 0]
        y = (x[:, 0] + x[:, 2] > 0).astype(int)
        cfg = {"scenario": "full", "correlation_threshold": .95,
               "z_score_threshold": 3, "outlier_row_feature_fraction": .02,
               "rf_n_estimators": 100, "seed": 42, "pca_variance": .95}
        processor = FeatureProcessor(cfg)
        fitted, labels = processor.fit_transform(x, y)
        holdout = processor.transform(rng.normal(size=(10, 20)))
        self.assertEqual(fitted.shape[1], holdout.shape[1])
        self.assertEqual(len(fitted), len(labels))
        self.assertLess(len(processor.correlation_columns), 20)


if __name__ == "__main__":
    unittest.main()

