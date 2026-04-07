from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


PUBLIC_DATASET_REFERENCE = {
    "dataset_name": "MetroPT-3 Dataset",
    "dataset_owner": "UCI Machine Learning Repository",
    "dataset_reference": "MetroPT-3 Dataset",
    "dataset_url": "https://archive.ics.uci.edu/dataset/791/metropt+3+",
    "dataset_note": (
        "This project uses a compact local telemetry sample inspired by compressor predictive-maintenance datasets "
        "such as MetroPT-3, adapted to a natural-gas-powered fleet maintenance framing for deterministic execution."
    ),
}


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=".csv", delete=False, dir=path.parent, encoding="utf-8") as tmp_file:
        temp_path = Path(tmp_file.name)
    try:
        df.to_csv(temp_path, index=False)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=".json", delete=False, dir=path.parent, encoding="utf-8") as tmp_file:
        temp_path = Path(tmp_file.name)
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _generate_sample(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assets = [f"NG-{index:02d}" for index in range(1, 9)]
    observed_cycle_plan = [56, 63, 70, 76, 82, 88, 93, 99]
    rows: list[dict[str, object]] = []

    for asset_id, observed_cycles in zip(assets, observed_cycle_plan, strict=True):
        base_gas_pressure = rng.uniform(245, 290)
        base_oil_temperature = rng.uniform(78, 92)
        base_motor_current = rng.uniform(82, 110)
        base_compression_efficiency = rng.uniform(91, 97)
        base_exhaust_temperature = rng.uniform(340, 390)
        base_vibration = rng.uniform(0.9, 1.8)
        base_fuel_flow = rng.uniform(45, 58)

        for cycle in range(1, observed_cycles + 1):
            lifecycle = cycle / 100
            anomaly_event = rng.random() < (0.02 + 0.08 * lifecycle)

            gas_pressure = base_gas_pressure - lifecycle * rng.uniform(24, 52) + rng.normal(0, 1.5)
            oil_temperature = base_oil_temperature + lifecycle * rng.uniform(16, 30) + rng.normal(0, 0.75)
            motor_current = base_motor_current + lifecycle * rng.uniform(18, 36) + rng.normal(0, 0.9)
            compression_efficiency = (
                base_compression_efficiency - lifecycle * rng.uniform(8, 17) + rng.normal(0, 0.42)
            )
            exhaust_temperature = base_exhaust_temperature + lifecycle * rng.uniform(36, 82) + rng.normal(0, 1.5)
            vibration = base_vibration + lifecycle * rng.uniform(1.3, 2.6) + rng.normal(0, 0.035)
            fuel_flow = base_fuel_flow + lifecycle * rng.uniform(4.4, 9.4) + rng.normal(0, 0.30)
            start_delay_index = rng.uniform(0.8, 1.4) + lifecycle * rng.uniform(1.0, 2.6) + rng.normal(0, 0.04)

            risk_score = (
                0.16 * max(236 - gas_pressure, 0) / 10
                + 0.14 * max(oil_temperature - 98, 0) / 6
                + 0.14 * max(motor_current - 122, 0) / 10
                + 0.18 * max(88 - compression_efficiency, 0) / 4
                + 0.14 * max(exhaust_temperature - 405, 0) / 15
                + 0.14 * max(vibration - 2.4, 0)
                + 0.10 * max(start_delay_index - 2.4, 0)
            )
            maintenance_required = int(anomaly_event or risk_score > 0.34 or lifecycle > 0.80)

            rows.append(
                {
                    "asset_id": asset_id,
                    "cycle": cycle,
                    "gas_pressure": round(float(gas_pressure), 2),
                    "oil_temperature": round(float(oil_temperature), 2),
                    "motor_current": round(float(motor_current), 2),
                    "compression_efficiency": round(float(compression_efficiency), 2),
                    "exhaust_temperature": round(float(exhaust_temperature), 2),
                    "vibration": round(float(vibration), 3),
                    "fuel_flow": round(float(fuel_flow), 2),
                    "start_delay_index": round(float(start_delay_index), 3),
                    "maintenance_required": maintenance_required,
                }
            )

    return pd.DataFrame(rows)


def ensure_dataset(base_dir: str | Path) -> dict[str, str]:
    base_path = Path(base_dir)
    telemetry_path = base_path / "data" / "raw" / "natural_gas_fleet_telemetry_sample.csv"
    reference_path = base_path / "data" / "raw" / "public_dataset_reference.json"

    telemetry_df = _generate_sample()
    _atomic_write_csv(telemetry_df, telemetry_path)
    _atomic_write_json(PUBLIC_DATASET_REFERENCE, reference_path)

    return {
        "telemetry_path": str(telemetry_path),
        "reference_path": str(reference_path),
    }
