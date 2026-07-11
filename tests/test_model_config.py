"""Dependency-free tests for the H2ASeg model configuration contract."""

from __future__ import annotations

import json
import unittest

from utils.model_config import DEFAULT_MODEL_CONFIG, H2ASegModelConfig


class H2ASegModelConfigTests(unittest.TestCase):
    def test_miccai_2024_defaults_and_json_round_trip(self) -> None:
        config = H2ASegModelConfig()

        self.assertEqual(config.patch_size, (128, 128, 64))
        self.assertEqual(config.num_channels, (8, 16, 32, 64, 128))
        self.assertEqual(config.window_size, (8, 8, 4))
        self.assertEqual(config, DEFAULT_MODEL_CONFIG)
        payload = json.loads(json.dumps(config.to_dict()))
        self.assertEqual(H2ASegModelConfig.from_dict(payload), config)
        self.assertNotIn("downsample_levels", config.to_model_kwargs())

    def test_accepts_lists_from_cli_or_checkpoint_metadata(self) -> None:
        config = H2ASegModelConfig(
            patch_size=[128, 128, 64],
            num_channels=[8, 16, 32, 64, 128],
            window_size=[8, 8, 4],
        )

        self.assertIsInstance(config.patch_size, tuple)
        self.assertIsInstance(config.num_channels, tuple)
        self.assertIsInstance(config.window_size, tuple)

    def test_rejects_invalid_patch_size(self) -> None:
        invalid_values = ((128, 128), (128, 0, 64), (120, 128, 64))
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                H2ASegModelConfig(patch_size=value)

    def test_rejects_invalid_window_or_mcsa_divisibility(self) -> None:
        invalid_values = (
            (8, 8),
            (8, 0, 4),
            (10, 8, 4),  # stage-2 x axis is not divisible by its window
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                H2ASegModelConfig(window_size=value)

        # Axis-wise window partitioning works here, but the scalar convolution
        # pool/stride min(window_size)=6 cannot be inverted on every axis.
        with self.assertRaisesRegex(ValueError, "pool size"):
            H2ASegModelConfig(patch_size=(128, 96, 64), window_size=(8, 6, 16))

    def test_rejects_invalid_channels(self) -> None:
        invalid_values = (
            (8, 16, 32, 64),
            (8, 0, 32, 64, 128),
            (8, 3, 32, 64, 128),
            (8, 16, 3, 64, 128),
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                H2ASegModelConfig(num_channels=value)

    def test_rejects_architecture_changes_not_supported_by_current_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "in_channels must be 1"):
            H2ASegModelConfig(in_channels=2)
        with self.assertRaisesRegex(ValueError, "n_classes must be 2"):
            H2ASegModelConfig(n_classes=3)
        with self.assertRaisesRegex(ValueError, "downsample_levels must be 4"):
            H2ASegModelConfig(downsample_levels=3)


if __name__ == "__main__":
    unittest.main()
