"""Compare locally propagated metrics with externally generated full-device data."""

from __future__ import annotations

import csv
from pathlib import Path
import math


REQUIRED_COLUMNS = {
    "state",
    "wavelength_nm",
    "launch_layer",
    "il_db",
    "xt_db",
    "guided_output_power",
}


def _read(path: str | Path) -> dict[tuple, dict]:
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = {}
        for row in reader:
            key = (
                row["state"],
                round(float(row["wavelength_nm"]), 9),
                row["launch_layer"],
            )
            rows[key] = row
        return rows


def compare_full_device_csv(local_path: str | Path, reference_path: str | Path) -> dict:
    """Compare matching state/wavelength/input rows; full-device CSVs may come from EME/FDTD."""
    local = _read(local_path)
    reference = _read(reference_path)
    common = sorted(set(local).intersection(reference))
    if not common:
        raise ValueError("No matching state/wavelength/launch rows were found.")
    comparisons = []
    for key in common:
        a, b = local[key], reference[key]
        comparisons.append(
            {
                "state": key[0],
                "wavelength_nm": key[1],
                "launch_layer": key[2],
                "delta_il_db_local_minus_reference": float(a["il_db"]) - float(b["il_db"]),
                "delta_xt_db_local_minus_reference": float(a["xt_db"]) - float(b["xt_db"]),
                "delta_guided_power_local_minus_reference": (
                    float(a["guided_output_power"]) - float(b["guided_output_power"])
                ),
            }
        )
    return {
        "matched_rows": len(common),
        "local_approximation_file": str(local_path),
        "full_device_reference_file": str(reference_path),
        "comparisons": comparisons,
        "mean_abs_delta_il_db": sum(
            abs(r["delta_il_db_local_minus_reference"]) for r in comparisons
        ) / len(comparisons),
        "mean_abs_delta_xt_db": sum(
            abs(r["delta_xt_db_local_minus_reference"]) for r in comparisons
        ) / len(comparisons),
        "status": "comparison completed; inspect rows and solver/reference settings",
    }
