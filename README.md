# dreamzero-yam

LoRA post-train of NVIDIA [DreamZero](https://github.com/dreamzero0/dreamzero) (14B world-action model, base `GEAR-Dreams/DreamZero-AgiBot`) on the 124-repo MolmoAct2 BimanualYAM split, trained on Modal — the first public DreamZero fine-tune on this dataset.

Target: `robocurve/dreamzero-yam-molmoact2` on HF, with held-out open-loop MSE on `allenai/19012026-block-13` compared against our [π0.5](https://huggingface.co/robocurve/pi05-yam-molmoact2) (0.00206) and [GR00T N1.7](https://huggingface.co/robocurve/gr00t-n1.7-yam-molmoact2) (0.00279) replications under a unified re-eval protocol.

**Plan:** [plans/0001-dreamzero-yam-finetune.md](plans/0001-dreamzero-yam-finetune.md) (6 critique rounds, approved 2026-07-31; budget cap $5k)

Status: Phases 0–2 and the pulled-forward Phase-3 benchmark/monitoring scaffold are implemented. No Modal function or GPU job has been run from this repository yet.

## Pipeline

```text
gr00t-yam-data:/gr00t_v2/allenai/*        existing v2 + H.264, never modified
                 │ hardlinks (symlinks if unsupported)
                 ▼
gr00t-yam-data:/dreamzero_gear/allenai/*  fresh DreamZero GEAR metadata
                 │ copy selected 124 repos to container NVMe
                 ▼
hardware benchmark ──► selected --gpu-spec
                              │
DreamZero LoRA train ──► dreamzero-yam-ckpt:/yam/<experiment>/checkpoint-*
       │                      │
       │                      └──► milestones ──► 1-GPU val monitor
       └──► WandB train/loss                         │
                                val/loss + 16/24-step val/open_loop_mse
```

`src/prepare_data.py` owns the CPU-only GEAR transform. `src/modal_app.py` owns checkpoint downloads, the 2×H100 smoke drill, hardware benchmark, tuning arms, and resumable train job. `src/val_monitor.py` watches milestones. `src/reconstruct.py` is the only supported LoRA reconstruction path. `src/eval_spec.py` defines the frozen cross-model sample and metric contract.
`src/eval_dreamzero.py` implements the DreamZero side of that contract as one
independent single-frame observation and one 24-action causal chunk per frozen
index; it never routes checkpoint loading through `GrootSimPolicy`.

## Run stages

Run these from the repository root. Data prep is intentionally a separate Modal app/image from training.

```bash
# CPU only: build and validate the sibling GEAR tree (124 train + held-out val)
modal run src/prepare_data.py::prepare_data

# CPU/network: populate the checkpoint volume with the three pinned dependencies
modal run src/modal_app.py::download_checkpoints

# 2x H100: 20-step kill + resume to step 40, with load/resume/parity assertions
modal run src/modal_app.py::smoke_test

# Phase-3 hardware matrix. Defaults cover H100:8/H100:4/B200:8/H200:8,
# dataloader workers 1/8, and container CPU counts 16/32. Each GPU/CPU
# container reuses one NVMe copy of the same deterministic 10-repo subset.
modal run src/modal_app.py::benchmark

# Equal-wall-clock arms on the benchmark winner (example winner shown).
modal run src/modal_app.py::tune --arm A --gpu-spec H100:4 --per-device-batch-size 8
modal run src/modal_app.py::tune --arm B --gpu-spec H100:4 --per-device-batch-size 8
modal run src/modal_app.py::tune --arm C --gpu-spec H100:4 --per-device-batch-size 8

# Main run consumes the selected GPU spec as a value, with no source edit.
# Keep --exp-name equal to the winning arm so it resumes that prefix, including
# optimizer/scheduler state; use tune-armB/C here if that arm won.
# save_steps defaults to 0 = auto: the Young-Daly optimum sqrt(2*write*MTBF),
# snapped to a divisor of the 2,500-step milestone interval.
modal run --detach src/modal_app.py::train --gpu-spec H100:4 \
  --per-device-batch-size 8 --exp-name tune-armA

# Restarts are config-portable: train/tune assert global batch 32 and compare
# the checkpoint's DeepSpeed shard count with the current world size. To
# restart on a different GPU count, pass --allow-fresh-optimizer: the old
# lineage is archived and the trained LoRA/head weights warm-start a fresh
# optimizer at step 0.
# Automatic retries and manual relaunches resume from the latest complete
# checkpoint; the main worker allows eight retries across Modal's 24 h limit.

# Run alongside training; it watches complete milestone copies (every saved
# tune checkpoint, then every 2,500 steps in main training) and mirrors each
# evaluated milestone to HF (robocurve/dreamzero-yam-molmoact2 under
# checkpoints/step-N/, via the hf-robocurve-write secret).
modal run --detach src/val_monitor.py::val_monitor --exp-name tune-armA
```

Arm A is r/α 4/4 at 1e-5, Arm B changes only LR to 5e-6, and Arm C changes only capacity to r/α 32/32. The checked-in `configs/arms/*.json` files fix warmup ratio at 0.01 for all arms; main-train defaults match it for warm-start compatibility. Arm runs stop after the same `--trial-minutes` wall-clock budget.

Generate the held-out indices on the evaluator's CPU-only Modal function after
the data exists, review its JSON stdout, and commit it as
`outputs/eval_indices.json` before evaluation:

```bash
modal run src/eval_dreamzero.py::generate_eval_indices

# One-checkpoint DreamZero evaluation; output defaults to
# /checkpoints/yam/eval/<experiment>-step-<N>-dreamzero.json.
modal run src/eval_dreamzero.py::eval_checkpoint \
  --checkpoint /checkpoints/yam/<experiment>/milestones/checkpoint-N

python -m src.eval_spec check-gt outputs/pi05.json outputs/gr00t.json outputs/dreamzero.json
python -m src.eval_spec metric outputs/dreamzero.json
```

Until `outputs/eval_indices.json` is generated and checked in, all eval consumers fail with an instruction explaining that prerequisite.

## Monitoring

Use `modal app logs dreamzero-yam` for container, preemption, NVMe-copy, resume, and checkpoint-commit logs, and `modal app logs dreamzero-yam-val-monitor` for the validation sidecar. Every run reports to the single `dreamzero-yam` WandB project with stage-bearing names (`smoke`, `tune-armA/B/C`, `train`, `benchmark-*`, and `val-monitor`). The metric contract is:

- `train/loss`: Hugging Face Trainer's flow-matching training objective.
- `val/loss`: the same objective on fixed held-out samples with fixed noise/timestep draws.
- `val/open_loop_mse`: teacher-forced raw-absolute 14-D MSE over predicted steps 0–15.
- `val/open_loop_mse_24`: the same teacher-forced raw-absolute MSE over all 24 predicted steps.
- `instability/*`: NaN flag, loss/rolling-median spike ratio, normalized loss trend over the trailing 20% arm window (`rising_tail`), and grad-norm/rolling-median ratio.
- `gpu/*`, `dcgm/source_tier`, and `perf/*`: per-GPU utilization, profiling tier, throughput, analytic model TFLOPs/s/GPU, MFU, cost per 1,000 samples, and dataloader wait (`perf/dataloader_wait_seconds_per_step`, `perf/dataloader_wait_fraction`).

The benchmark prints and logs a WandB table containing samples/s, dollars per 1,000 samples, TFLOPs/s/GPU, and MFU for every GPU/worker/CPU configuration. GPU telemetry tries DCGM profiling, then NVML GPM, then ordinary NVML utilization/VRAM at roughly 10-second cadence.

For live inspection, find the active container and open the image's `nvtop` TUI:

```bash
modal container list
modal container exec <container-id> -- nvtop
```

The main worker commits `dreamzero-yam-ckpt` about every 150 seconds, prunes incomplete checkpoints before resume, retains the five latest checkpoints, and preserves complete 2,500-step milestones. Tune stages preserve every complete checkpoint at their configured `save_steps` cadence so the val monitor can score short arms.

The data volume is treated as read-only during GPU jobs. Checkpoint `config.json` files intentionally contain the stable `/checkpoints/...` component paths used by both training and reconstruction.
