# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Apply the rust-vs-cpp benchmark tolerance gates from the AloneHaddock DoD.

Reads paired JSON outputs from `benchmarks/results/`:
  - kitti360_{cpp,rust}.json          full KITTI-360 run
  - place_recognition_{cpp,rust}.json Scan Context AP run
  - smoke_{cpp,rust}.json             smoke benchmark

For each pair, verifies the metrics meet the tolerance bands agreed with the
user (see DoD). Prints PASS/FAIL per gate and exits non-zero on any failure.

The harnesses that *produce* these JSONs are `benchmark_kitti360.py`,
`benchmark_place_recognition.py`, and `benchmark_kitti360_smoke.py` in this
module's parent directory. They must be run on a machine with the KITTI-360
dataset locally accessible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

# Per the DoD — locked tolerances:
PER_FRAME_LATENCY_GATE = "strict"  # rust median ≤ cpp median
END_TO_END_WALL_CLOCK_GATE = "strict"  # rust wall ≤ cpp wall
PEAK_RSS_RATIO_MAX = 1.10  # rust ≤ 1.10x cpp
ATE_RATIO_MAX = 1.05  # rust ATE ≤ 1.05x cpp ATE
LOOP_PRECISION_DELTA = 0.02  # rust ≥ cpp - 0.02 absolute
LOOP_RECALL_DELTA = 0.02
SCAN_CONTEXT_AP_DELTA = 0.02
SCAN_CONTEXT_AP_BAND = (0.65, 0.78)


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str

    def as_line(self) -> str:
        symbol = "PASS" if self.passed else "FAIL"
        return f"  [{symbol}] {self.name}: {self.detail}"


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _strict_le(name: str, rust: float, cpp: float, unit: str) -> GateResult:
    passed = rust <= cpp
    return GateResult(
        name=name,
        passed=passed,
        detail=f"rust={rust:.6g}{unit}, cpp={cpp:.6g}{unit} (rust ≤ cpp)",
    )


def _ratio_le(name: str, rust: float, cpp: float, max_ratio: float, unit: str) -> GateResult:
    if cpp == 0:
        return GateResult(name=name, passed=rust == 0, detail=f"both must be 0, got rust={rust}")
    ratio = rust / cpp
    passed = ratio <= max_ratio
    return GateResult(
        name=name,
        passed=passed,
        detail=f"rust={rust:.6g}{unit}, cpp={cpp:.6g}{unit}, ratio={ratio:.3f} (≤ {max_ratio})",
    )


def _delta_ge(name: str, rust: float, cpp: float, delta: float, unit: str) -> GateResult:
    threshold = cpp - delta
    passed = rust >= threshold
    return GateResult(
        name=name,
        passed=passed,
        detail=f"rust={rust:.6g}{unit}, cpp={cpp:.6g}{unit} (rust ≥ cpp - {delta} = {threshold:.6g})",
    )


def check_kitti360(cpp: dict[str, Any], rust: dict[str, Any]) -> list[GateResult]:
    return [
        _strict_le(
            "per-frame median latency",
            rust["per_frame_median_ms"],
            cpp["per_frame_median_ms"],
            "ms",
        ),
        _strict_le(
            "end-to-end wall clock", rust["wall_clock_seconds"], cpp["wall_clock_seconds"], "s"
        ),
        _ratio_le("peak RSS", rust["peak_rss_mb"], cpp["peak_rss_mb"], PEAK_RSS_RATIO_MAX, "MB"),
        _ratio_le("ATE", rust["ate_meters"], cpp["ate_meters"], ATE_RATIO_MAX, "m"),
        _delta_ge(
            "loop precision",
            rust["loop_precision"],
            cpp["loop_precision"],
            LOOP_PRECISION_DELTA,
            "",
        ),
        _delta_ge("loop recall", rust["loop_recall"], cpp["loop_recall"], LOOP_RECALL_DELTA, ""),
    ]


def check_place_recognition(cpp: dict[str, Any], rust: dict[str, Any]) -> list[GateResult]:
    rust_ap = rust["scan_context_ap"]
    cpp_ap = cpp["scan_context_ap"]
    low, high = SCAN_CONTEXT_AP_BAND
    in_band = low <= rust_ap <= high
    return [
        _delta_ge("scan context AP", rust_ap, cpp_ap, SCAN_CONTEXT_AP_DELTA, ""),
        GateResult(
            name="scan context AP in paper band",
            passed=in_band,
            detail=f"rust AP = {rust_ap:.4f} (band {low:.2f}-{high:.2f})",
        ),
    ]


def check_smoke(cpp: dict[str, Any], rust: dict[str, Any]) -> list[GateResult]:
    return [
        GateResult(
            name="smoke run completed",
            passed=bool(cpp.get("completed")) and bool(rust.get("completed")),
            detail=f"cpp.completed={cpp.get('completed')}, rust.completed={rust.get('completed')}",
        ),
        _strict_le(
            "smoke wall clock",
            rust["wall_clock_seconds"],
            cpp["wall_clock_seconds"],
            "s",
        ),
    ]


def run(results_dir: Path) -> int:
    pairs = [
        ("KITTI-360 full", "kitti360_cpp.json", "kitti360_rust.json", check_kitti360),
        (
            "Place recognition",
            "place_recognition_cpp.json",
            "place_recognition_rust.json",
            check_place_recognition,
        ),
        ("Smoke", "smoke_cpp.json", "smoke_rust.json", check_smoke),
    ]
    overall_pass = True
    for label, cpp_name, rust_name, checker in pairs:
        cpp_path = results_dir / cpp_name
        rust_path = results_dir / rust_name
        print(f"\n=== {label} ===")
        if not cpp_path.exists() or not rust_path.exists():
            missing = [str(path) for path in (cpp_path, rust_path) if not path.exists()]
            print(f"  [SKIP] missing result file(s): {', '.join(missing)}")
            overall_pass = False
            continue
        gates = checker(_load(cpp_path), _load(rust_path))
        for gate in gates:
            print(gate.as_line())
            if not gate.passed:
                overall_pass = False
    print(f"\n{'OVERALL: PASS' if overall_pass else 'OVERALL: FAIL'}")
    return 0 if overall_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare pgo_cpp vs pgo_rust benchmark results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing the JSON result files.",
    )
    args = parser.parse_args()
    return run(args.results_dir)


if __name__ == "__main__":
    sys.exit(main())
