# pgo_rust benchmarks

This directory holds the parity-gate harness for comparing `pgo_rust` against
`pgo_cpp`. The DoD-locked tolerances are encoded in `compare.py` (constants at
the top of the file) and applied to JSON result files in `results/`.

## Running the benchmarks

The actual benchmark scripts live one directory up (alongside `pgo_rust.py`):

- `benchmark_kitti360.py`           — full KITTI-360 evaluation, emits ATE / loop precision-recall / timing
- `benchmark_kitti360_smoke.py`     — quick liveness probe, no dataset required
- `benchmark_place_recognition.py`  — Scan Context AP only

Run each twice — once selecting `loop_closure='pgo_cpp'`, once `loop_closure='pgo_rust'` — and save the JSON output into `results/` with the names listed below.

## Expected JSON files

```
results/
├── kitti360_cpp.json
├── kitti360_rust.json
├── place_recognition_cpp.json
├── place_recognition_rust.json
├── smoke_cpp.json
└── smoke_rust.json
```

Each JSON should at minimum contain (other keys ignored):

```jsonc
// kitti360_*.json
{
  "git_sha": "26e6e2af6",
  "per_frame_median_ms": 8.3,
  "wall_clock_seconds": 412.0,
  "peak_rss_mb": 480.0,
  "ate_meters": 1.94,
  "loop_precision": 0.94,
  "loop_recall": 0.78
}
// place_recognition_*.json
{ "git_sha": "...", "scan_context_ap": 0.71 }
// smoke_*.json
{ "git_sha": "...", "completed": true, "wall_clock_seconds": 35.2 }
```

## Running the comparator

```bash
python compare.py
```

Exits 0 on PASS, 1 on FAIL. Each gate is logged with the source numbers so a
reviewer can audit which metric failed and by how much.

## Tolerance bands (from DoD, agreed with user)

| Metric                       | Gate                                  |
|------------------------------|---------------------------------------|
| Per-frame median latency     | rust ≤ cpp (strict)                   |
| End-to-end wall clock        | rust ≤ cpp (strict)                   |
| Peak RSS                     | rust ≤ 1.10 × cpp                     |
| ATE                          | rust ≤ 1.05 × cpp                     |
| Loop precision               | rust ≥ cpp − 0.02 (absolute)          |
| Loop recall                  | rust ≥ cpp − 0.02 (absolute)          |
| Scan Context AP              | rust ≥ cpp − 0.02 AND in [0.65, 0.78] |
