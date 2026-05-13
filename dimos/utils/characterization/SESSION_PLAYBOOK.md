# Go2 characterization + controller benchmark guide

A task-oriented walkthrough. Each section answers one "I want to ..."
question with one canonical command. Pick the section that matches your
goal and copy-paste.

> **TL;DR pipeline:** run a session → `process_session` → look at SVGs →
> (optional) run the controller benchmark.

---

## 0. One-time setup

```bash
# 1. Activate the venv at the repo root.
cd ~/Documents/repos/dimos && source .venv/bin/activate

# 2. Make recipes importable. Save the catalog at the bottom of this
#    file as ~/scripts/my_recipes.py, then:
export PYTHONPATH="$PYTHONPATH:$HOME/scripts"
# add that line to ~/.bashrc to persist.
```

Confirm: `python -c "import my_recipes; print(my_recipes.step_vx_1)"`
should print a `TestRecipe(...)` repr without raising.

**Surface tag** — pick one string and stick with it for the day. It's
free-form metadata used to bucket the data later.
```bash
SURFACE=carpet            # or vinyl, concrete, hardwood, ...
```

---

## 1. I want to do a 3-minute sanity check before a long session

One step, one repeat. Confirms odom flows, BMS captured, plot renders.

```bash
python -m dimos.utils.characterization.scripts.run_session \
    --recipes "my_recipes:step_vx_1:1" \
    --surface $SURFACE --notes "morning sanity" \
    --out-dir ~/char_runs
```

Then look at the result (see §3).

---

## 2. I want to run the full 7-experiment characterization (≈75 min)

Run these **in order**, one session each. Battery: start ≥90% SOC,
swap if the BMS readout drops below 50% (check after E2). Robot
expects WASD teleop on the pygame window between runs.

| # | Experiment | What it measures | Time | Runs |
|---|---|---|---|---|
| E1 | vx step matrix | linear-velocity gain at 4 amplitudes × 2 dir × 5 reps | 17 m | 40 |
| E3 | vx ramp 0→1.2 m/s | linear saturation point (5 m runway) | 5 m | 5 |
| E2 | wz step matrix | angular-velocity gain (in place) | 17 m | 40 |
| E4 | wz ramp 0→4 rad/s | yaw saturation (in place) | 5 m | 5 |
| E7a | pure-wz cross-coupling | unintended vx/vy leak when commanding wz | 8 m | 18 |
| E7b | pure-vx cross-coupling | unintended wz drift when commanding vx | 10 m | 18 |
| E8 | sharp 0.5 s vx step ×30 | deadtime distribution (most-actionable single number) | 10 m | 30 |

Each command pattern is identical:

```bash
python -m dimos.utils.characterization.scripts.run_session \
    --randomize --rng-seed <N> \
    --surface $SURFACE --notes "<experiment label>" \
    --out-dir ~/char_runs \
    --recipes "<csv list of my_recipes:<name>:<reps>>"
```

Just swap the recipe list. Use these:

- **E1** — `my_recipes:e1_vx_pos_0p3:5,my_recipes:e1_vx_neg_0p3:5,my_recipes:e1_vx_pos_0p6:5,my_recipes:e1_vx_neg_0p6:5,my_recipes:e1_vx_pos_1p0:5,my_recipes:e1_vx_neg_1p0:5,my_recipes:e1_vx_pos_1p5:5,my_recipes:e1_vx_neg_1p5:5`
- **E3** — `my_recipes:e3_vx_ramp_0_to_1p2_5m:5`
- **E2** — `my_recipes:e2_wz_pos_0p3:5,my_recipes:e2_wz_neg_0p3:5,my_recipes:e2_wz_pos_0p6:5,my_recipes:e2_wz_neg_0p6:5,my_recipes:e2_wz_pos_1p0:5,my_recipes:e2_wz_neg_1p0:5,my_recipes:e2_wz_pos_1p5:5,my_recipes:e2_wz_neg_1p5:5`
- **E4** — `my_recipes:e4_wz_ramp_0_to_4:5`
- **E7a** — `my_recipes:e7a_wz_pos_0p3:3,my_recipes:e7a_wz_neg_0p3:3,my_recipes:e7a_wz_pos_0p8:3,my_recipes:e7a_wz_neg_0p8:3,my_recipes:e7a_wz_pos_1p5:3,my_recipes:e7a_wz_neg_1p5:3`
- **E7b** — `my_recipes:e7b_vx_pos_0p3:3,my_recipes:e7b_vx_neg_0p3:3,my_recipes:e7b_vx_pos_0p6:3,my_recipes:e7b_vx_neg_0p6:3,my_recipes:e7b_vx_pos_1p0:3,my_recipes:e7b_vx_neg_1p0:3`
- **E8** — `my_recipes:e8_vx_short_step:15,my_recipes:e8_vx_short_step_neg:15`

**During each run**: drive robot to the start mark with WASD, **release
keys**, ENTER to fire. The robot runs the recipe and returns control.

After the day is done, back the data up:

```bash
TODAY=$(date +%Y-%m-%d)
mkdir -p ~/char_data/${TODAY}_${SURFACE}
cp -r ~/char_runs/session_* ~/char_data/${TODAY}_${SURFACE}/
```

---

## 3. I want to look at one session's plots

```bash
SESSION=$(ls -td ~/char_runs/session_*/ | head -1); SESSION=${SESSION%/}

# Per-run plots (writes plot.svg into each run dir)
for d in $SESSION/0*; do
    python -m dimos.utils.characterization.scripts.analyze run "$d"
done

# Multi-run overlay (one SVG with all repeats stacked)
python -m dimos.utils.characterization.scripts.analyze compare \
    $SESSION/0* --out $SESSION/compare_all.svg

xdg-open $SESSION/compare_all.svg
```

Want only the matching repeats? Glob smarter:
`$SESSION/*e1_vx_+1.0_r*` filters to one amplitude.

> Note: the `+` in recipe names becomes `_` in directory names. So
> `e1_vx_+1.0` on disk is `e1_vx__1.0` (double underscore).

---

## 4. I want to process a session into derived artifacts

Five subcommands; the order matters. Run on every session you collected.

```bash
SESSION=~/char_data/<date>_<surface>/session_<ts>

python -m dimos.utils.characterization.scripts.process_session validate   $SESSION
python -m dimos.utils.characterization.scripts.process_session aggregate  $SESSION
python -m dimos.utils.characterization.scripts.process_session deadtime   $SESSION   # E8 sessions only
python -m dimos.utils.characterization.scripts.process_session coupling   $SESSION   # E7 sessions only
python -m dimos.utils.characterization.scripts.process_session envelope   $SESSION1 $SESSION2 ... --out envelope.md
```

Each step writes to `$SESSION/processing/<step>.json|.md`. Nothing
destructive — re-run freely.

---

## 5. I want to run the same suite on rage mode

Rage mode = the Go2's high-performance gait FSM. The harness toggles it
by patching the `GO2Connection` atom right before `StandUp` →
`BalanceStand` → `enable_rage_mode`. **The robot will move noticeably
faster.** Make sure your runway is clear and you're ready on WASD.

Just add `--rage` to any §1, §2 command:

```bash
python -m dimos.utils.characterization.scripts.run_session \
    --rage \
    --recipes "my_recipes:e1_vx_pos_1p0:5" \
    --surface $SURFACE --notes "E1 vx +1.0 rage" \
    --out-dir ~/char_runs
```

The session banner prints `[RAGE]` and `session.json` records `"rage": true`.
Teleop speeds also bump (`linear=1.25`, `angular=1.2`) so manual repositioning
between runs isn't laggy compared to the recipe-driven motion.

---

## 6. I want to run the controller benchmark

Compares 5 path-following controllers over a path battery. Source of
truth: `dimos/utils/benchmarking/run_battery.py`.

The cohort matrix:

| cohort | role |
|---|---|
| `baseline_k0.5` | production anchor (LocalPlanner P-controller, k=0.5) |
| `pure_pursuit` | classic Pure Pursuit, no feedforward |
| `pure_pursuit_ff` | Pure Pursuit + plant-gain feedforward |
| `rpp_ff` | Regulated Pure Pursuit + FF (sim winner) |
| `lyapunov_ff` | Lyapunov reactive + FF (first-principles) |

### 6a. Sim sweep — full cohort × full path battery

```bash
python -m dimos.utils.benchmarking.run_battery --mode sim
```

Default output dir is `/tmp/benchmarking_baseline` (override with `--out`).
What you get:
- `sim_baseline_<sha>.json` — scores for every (cohort, path) pair.
- `<cohort>/<path>.svg` and `<cohort>/<path>_traj.json` per pair.
- `composite/<path>.svg` — all cohorts overlaid on one path.
- `index.html` — top-level browser view.

```bash
python3 -m http.server 8000 --directory /tmp/benchmarking_baseline
# open http://localhost:8000
```

### 6b. Re-render plots without re-simulating

If you tweak plot styling and don't want to wait for a fresh sim:

```bash
python -m dimos.utils.benchmarking.run_battery --mode replot
```

Reads the persisted `<cohort>/<path>_traj.json` files plus the
`sim_baseline_<sha>.json` from `--out` (default `/tmp/benchmarking_baseline`)
and rewrites the SVGs + index.html in seconds.

### 6c. Hardware run (one controller, one path)

```bash
python -m dimos.utils.benchmarking.run_battery \
    --mode hw \
    --controller rpp \
    --path straight_5m \
    --ff
```

- `--controller` ∈ {baseline, lyapunov, pure_pursuit, rpp, mpc}.
- `--path <name>` runs one path; omit for the full battery.
- `--ff` / `--pi` toggle feedforward and PI velocity-tracking (mutually exclusive).
- Default output dir: `/tmp/benchmarking_hw` (override with `--out`).

---

## 7. Where things live on disk

```
~/char_runs/                                # --out-dir from run_session
└── session_<YYYYMMDD-HHMMSS>/
    ├── session.json                        # plan + clock_anchor + rage flag + notes
    ├── recording.db                        # ONE sqlite store, sliced per run
    ├── 000_<recipe>_r1of5/
    │   ├── run.json                        # ts_window_wall, BMS, exit_reason
    │   ├── cmd_monotonic.jsonl             # commanded twist samples
    │   ├── plot.svg                        # written by `analyze run`
    │   └── metrics.json                    # written by `analyze run`
    ├── 001_<recipe>_r2of5/
    │   └── ...
    ├── compare_*.svg                       # written by `analyze compare`
    └── processing/                         # written by process_session
        ├── validation.json
        ├── session_summary.json
        ├── deadtime_stats.json
        ├── coupling_stats.json
        └── envelope.md

~/char_data/<date>_<surface>/               # your daily backup target
```

`run.json.clock_anchor` ties monotonic ↔ wall clocks. Don't compare
`tx_mono` to `obs.ts` directly — convert via the anchor first.

---

## 8. Spot-checking raw data

Plot looks weird, want to read actual sample values?

```bash
RUN=$(ls -d $SESSION/0* | head -1)
cat $RUN/run.json | head -40
head -3 $RUN/cmd_monotonic.jsonl

python -c "
from dimos.memory2.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
import json
meta = json.load(open('$RUN/run.json'))
win = meta['ts_window_wall']
s = SqliteStore(path='$SESSION/recording.db'); s.start()
try:
    obs = [o for o in s.stream('measured', PoseStamped).to_list()
           if win['start'] <= o.ts <= win['end']]
    print(f'{len(obs)} measured samples in run')
    for o in obs[:3]:
        print(f'  ts={o.ts:.3f} x={o.data.x:+.3f} y={o.data.y:+.3f} yaw={o.data.yaw:+.3f}')
finally:
    s.stop()
"
```

---

## 9. End-of-day checklist

- All 7 sessions present in `~/char_data/<date>_<surface>/`.
- Eyeballed at least one `compare_*.svg` per session — measured traces
  overlap each other (low noise) and aren't all flat (real signal).
- Logged anything anomalous in a sibling `notes.md` — surface
  inconsistencies, weird drifts, battery swap times.

---

## Recipe catalog — save as `~/scripts/my_recipes.py`

```python
# my_recipes.py
from dimos.utils.characterization.recipes import TestRecipe, constant, ramp, step

def _step(amp, channel, name):
    return TestRecipe(name=name, test_type="step", duration_s=3.0,
                      signal_fn=step(amplitude=amp, channel=channel))

def _const_hold(vx, vy, wz, name, duration=5.0):
    return TestRecipe(name=name, test_type="constant", duration_s=duration,
                      signal_fn=constant(vx=vx, vy=vy, wz=wz),
                      pre_roll_s=2.0, post_roll_s=2.0)

# E1 - vx step response
e1_vx_pos_0p3 = _step( 0.3, "vx", "e1_vx_+0.3");  e1_vx_neg_0p3 = _step(-0.3, "vx", "e1_vx_-0.3")
e1_vx_pos_0p6 = _step( 0.6, "vx", "e1_vx_+0.6");  e1_vx_neg_0p6 = _step(-0.6, "vx", "e1_vx_-0.6")
e1_vx_pos_1p0 = _step( 1.0, "vx", "e1_vx_+1.0");  e1_vx_neg_1p0 = _step(-1.0, "vx", "e1_vx_-1.0")
e1_vx_pos_1p5 = _step( 1.5, "vx", "e1_vx_+1.5");  e1_vx_neg_1p5 = _step(-1.5, "vx", "e1_vx_-1.5")

# E2 - wz step response
e2_wz_pos_0p3 = _step( 0.3, "wz", "e2_wz_+0.3");  e2_wz_neg_0p3 = _step(-0.3, "wz", "e2_wz_-0.3")
e2_wz_pos_0p6 = _step( 0.6, "wz", "e2_wz_+0.6");  e2_wz_neg_0p6 = _step(-0.6, "wz", "e2_wz_-0.6")
e2_wz_pos_1p0 = _step( 1.0, "wz", "e2_wz_+1.0");  e2_wz_neg_1p0 = _step(-1.0, "wz", "e2_wz_-1.0")
e2_wz_pos_1p5 = _step( 1.5, "wz", "e2_wz_+1.5");  e2_wz_neg_1p5 = _step(-1.5, "wz", "e2_wz_-1.5")

# E3 - vx saturation ramp (5 m runway)
e3_vx_ramp_0_to_1p2_5m = TestRecipe(
    name="e3_vx_ramp_0_to_1.2_5m", test_type="ramp", duration_s=7.0,
    signal_fn=ramp(start=0.0, end=1.2, duration=7.0, channel="vx"),
    pre_roll_s=1.5, post_roll_s=1.5,
)

# E4 - wz saturation ramp (in place)
e4_wz_ramp_0_to_4 = TestRecipe(
    name="e4_wz_ramp_0_to_4", test_type="ramp", duration_s=15.0,
    signal_fn=ramp(start=0.0, end=4.0, duration=15.0, channel="wz"),
    pre_roll_s=2.0, post_roll_s=2.0,
)

# E7a - pure wz, measure vx leak
e7a_wz_pos_0p3 = _const_hold(0, 0,  0.3, "e7a_wz_+0.3");  e7a_wz_neg_0p3 = _const_hold(0, 0, -0.3, "e7a_wz_-0.3")
e7a_wz_pos_0p8 = _const_hold(0, 0,  0.8, "e7a_wz_+0.8");  e7a_wz_neg_0p8 = _const_hold(0, 0, -0.8, "e7a_wz_-0.8")
e7a_wz_pos_1p5 = _const_hold(0, 0,  1.5, "e7a_wz_+1.5");  e7a_wz_neg_1p5 = _const_hold(0, 0, -1.5, "e7a_wz_-1.5")

# E7b - pure vx, measure wz leak
e7b_vx_pos_0p3 = _const_hold( 0.3, 0, 0, "e7b_vx_+0.3");  e7b_vx_neg_0p3 = _const_hold(-0.3, 0, 0, "e7b_vx_-0.3")
e7b_vx_pos_0p6 = _const_hold( 0.6, 0, 0, "e7b_vx_+0.6");  e7b_vx_neg_0p6 = _const_hold(-0.6, 0, 0, "e7b_vx_-0.6")
e7b_vx_pos_1p0 = _const_hold( 1.0, 0, 0, "e7b_vx_+1.0");  e7b_vx_neg_1p0 = _const_hold(-1.0, 0, 0, "e7b_vx_-1.0")

# E8 - deadtime precision (sharp 1.0 m/s vx step, 0.5 s hold)
e8_vx_short_step = TestRecipe(
    name="e8_vx_short_step", test_type="step", duration_s=0.5,
    signal_fn=step(amplitude=1.0, channel="vx"),
    pre_roll_s=1.0, post_roll_s=2.0,
)
e8_vx_short_step_neg = TestRecipe(
    name="e8_vx_short_step_neg", test_type="step", duration_s=0.5,
    signal_fn=step(amplitude=-1.0, channel="vx"),
    pre_roll_s=1.0, post_roll_s=2.0,
)

# Sanity step
step_vx_1 = TestRecipe(
    name="step_vx_1.0", test_type="step", duration_s=3.0,
    signal_fn=step(amplitude=1.0, channel="vx"),
)
```
