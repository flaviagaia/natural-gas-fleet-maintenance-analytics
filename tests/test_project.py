from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.modeling import run_pipeline


class NaturalGasFleetMaintenanceAnalyticsTestCase(unittest.TestCase):
    def test_pipeline_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_pipeline(temp_dir)
            self.assertEqual(summary["dataset_source"], "natural_gas_fleet_sample_metropt3_style")
            self.assertEqual(summary["asset_count"], 8)
            self.assertGreaterEqual(summary["roc_auc"], 0.89)
            self.assertGreaterEqual(summary["average_precision"], 0.82)
            self.assertGreaterEqual(summary["f1"], 0.71)

            fleet_summary = pd.read_csv(Path(summary["fleet_summary_artifact"]))
            self.assertEqual(len(fleet_summary), 8)
            self.assertTrue(fleet_summary["health_score"].between(0, 100).all())


if __name__ == "__main__":
    unittest.main()
