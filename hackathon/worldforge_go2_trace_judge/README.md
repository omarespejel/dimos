# WorldForge Go2 Trace Judge

Hackathon submission from Omar Espejel, Abdel, and Ciro.

## Summary

WorldForge Go2 Trace Judge is an inspectable decision layer for Unitree Go2
autonomy. It turns robot-view observations into candidate actions, scores those
candidates, selects one action, and writes the evidence trail as replayable JSON.

The core loop is:

```text
observation + goal + candidate action -> score -> selected action -> evidence
```

This submission keeps DimOS as the robot/runtime layer and uses WorldForge-style
traces as the scoring, comparison, dataset, and evidence layer around Go2
actions.

## Links

- Project repo: https://github.com/omarespejel/worldforge-go2-trace-judge
- WorldForge: https://github.com/AbdelStark/worldforge
- Hugging Face dataset: https://huggingface.co/datasets/espejelomar/worldforge-go2-dimos-replay-world-pairs
- Hugging Face model: https://huggingface.co/espejelomar/go2-dimos-replay-latent-dynamics
- Decision trace examples: https://github.com/omarespejel/worldforge-go2-trace-judge/tree/main/docs/decision_trace_examples
- Replay-MPC demo: https://github.com/omarespejel/worldforge-go2-trace-judge/tree/main/artifacts/replay_mpc_demo
- Final demo video: pending final voiceover upload

## What We Built

- Real Unitree Go2 venue captures and robot-view traces.
- Label-preserving counterfactual candidate scenes for cube/marker navigation.
- WorldForge-style trace artifacts:
  - `score_info.json`
  - `candidate_scores.json`
  - `selected_action.json`
  - `outcome_after_execution.json`
  - `run_manifest.json`
- A micro world scorer for action ranking from trace features.
- A DimOS replay-derived Hugging Face dataset of 2,557 action-conditioned Go2
  current/future frame pairs from six usable public replay DBs.
- A small frozen-DINOv2 residual latent dynamics head trained on those replay
  pairs.
- A no-robot replay-MPC demo: real DimOS replay frame, six candidate egomotion
  futures, selected action, and WorldForge-style trace JSON.

## Why It Fits DimOS

DimOS already exposes the robot runtime: streams, replay data, blueprints,
camera frames, odometry, and Go2 control skills. This project adds a thin
decision-evidence layer on top:

```text
DimOS observes and executes
WorldForge-style scorer compares candidate actions
trace files explain and replay the decision
```

That boundary lets DimOS remain the hardware/control system while making robot
decisions auditable and model-ready.

## Repro

Clone the project repo and run:

```bash
make check
make dimos-replay-stretch
make final-video
```

The DimOS replay target downloads public replay archives into an ignored local
cache, derives current/future Go2 frame pairs, trains a small latent dynamics
head, and exports Hugging Face-ready dataset/model folders.

Current replay-world-model outputs:

```text
pairs: 2557
validation lift vs no-motion baseline: +0.0507 cosine
test lift vs no-motion baseline: +0.0182 cosine
replay-MPC selected demo margin: +0.0170 over best counterfactual
```

If memory is tight, run the DimOS side with the venue-recommended memory limit,
for example:

```bash
dimos --memory-limit 1GB --replay run unitree-go2-agentic
```

## Submission Note

This PR intentionally does not vendor the whole external project into DimOS.
The full source, generated artifacts, Hugging Face links, and final video live in
the project repo. The demo video link will be added here and in the PR body as
soon as the final voiceover export is ready.
