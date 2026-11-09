"""Tests of .env parsing, precedence, paths, types and validation."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fracnet.common import load_config


class TestEnvConfiguration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.file = Path(self.temp.name) / ".env"
        self.file.write_text("FRACNET_BATCH_SIZE=15\nFRACNET_OUTPUT_DIR=runs/dev\n", encoding="utf-8")

    def load(self):
        with patch.dict(os.environ, {}, clear=True):
            return load_config(self.file)

    def test_defaults_and_types(self):
        cfg = self.load()
        self.assertEqual(cfg["batch_size"], 15)
        self.assertIsInstance(cfg["batch_size"], int)
        self.assertEqual(cfg["dataset_names"], ["D1", "D2", "D3"])
        self.assertIsNone(cfg["ssl_max_images"])
        self.assertEqual(cfg["models"]["xception"], "legacy_xception")

    def test_relative_paths_resolve_from_env_file(self):
        cfg = self.load()
        # Windows temp directories may use 8.3 aliases (ALIREZ~1), whereas
        # Path.resolve() expands the same directory into its long name.
        self.assertEqual(
            Path(cfg["output_dir"]).resolve(),
            (Path(self.temp.name) / "runs" / "dev").resolve(),
        )

    def test_process_environment_overrides_file(self):
        with patch.dict(os.environ, {"FRACNET_BATCH_SIZE": "8", "FRACNET_EPOCHS": "7"}, clear=True):
            cfg = load_config(self.file)
        self.assertEqual(cfg["batch_size"], 8)
        self.assertEqual(cfg["epochs"], 7)

    def test_invalid_boolean_fails(self):
        self.file.write_text("FRACNET_PRETRAINED_IMAGENET=maybe\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be true or false"):
            self.load()

    def test_invalid_scenario_fails(self):
        self.file.write_text("FRACNET_SCENARIO=Unknown\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SCENARIO"):
            self.load()

    def test_unknown_variable_fails(self):
        self.file.write_text("FRACNET_SECRET_TOKEN=example\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unknown FracNet"):
            self.load()

    def test_invalid_voting_length_fails(self):
        self.file.write_text("FRACNET_VOTE_MODELS=xgboost,svm\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "odd number"):
            self.load()

    def test_missing_dotenv_fails_clearly(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(FileNotFoundError):
                load_config(Path(self.temp.name) / "missing.env")


if __name__ == "__main__":
    unittest.main()

