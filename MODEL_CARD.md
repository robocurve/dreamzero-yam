---
license: other
license_name: nvidia-open-model-license
base_model: GEAR-Dreams/DreamZero-AgiBot
tags:
  - robotics
  - world-action-model
  - dreamzero
  - yam
  - lerobot
---

# dreamzero-yam-molmoact2

The first public [DreamZero](https://github.com/dreamzero0/dreamzero) fine-tune
on the MolmoAct2 BimanualYAM dataset: a LoRA (r=32, α=32) post-train of
`GEAR-Dreams/DreamZero-AgiBot` (14B Wan2.1 video-diffusion world-action model)
on 124 LeRobot repositories (~5,145 episodes; block manipulation, box packing,
cable charging) collected on an I2RT bimanual YAM. Trained for 12,000 steps at
global batch 32 (~0.38M samples) on 8×B200 via Modal. The released checkpoint
is training step 10,000, selected on a held-out validation curve (see
Selection below).

## Headline result — three-model comparison under one frozen protocol

All three models score the **same 20 frozen (episode, frame) samples** of the
held-out repository `allenai/19012026-block-13`, predicting the next 16
actions from the observation at the anchor frame, scored as MSE against
ground-truth actions in **raw absolute 14-D joint space**. Cross-runner
bitwise ground-truth identity (and prompt identity) is machine-verified.

| Model | Recipe | Training samples | Mean MSE ↓ | Median | Per-sample std |
|---|---|---|---|---|---|
| [π0.5](https://huggingface.co/robocurve/pi05-yam-molmoact2) | full fine-tune | 10.2M | 0.002412 | 0.000569 | 0.00592 |
| [GR00T N1.7](https://huggingface.co/robocurve/gr00t-n1.7-yam-molmoact2) | action-head tune | ~5.1M | 0.002492 | 0.001432 | 0.00309 |
| **DreamZero (this model)** | LoRA r=32 | **0.38M** | 0.002524 | 0.001241 | 0.00347 |

**Interpretation (paired, per-sample):** the three models are statistically
indistinguishable on mean MSE at n=20 (paired mean differences ≈ 0.1σ).
Against GR00T, DreamZero splits the samples 10/20 with a paired mean
difference of +0.00003. Against π0.5, DreamZero wins 4/20 — π0.5 is better on
typical samples (lowest median) but carries the fattest outlier tail (largest
std); DreamZero is the more consistent model. The notable finding is
**sample efficiency**: DreamZero reaches parity having seen 13–27× fewer
training samples than the baselines, consistent with the world-action-model
transfer claims of the DreamZero paper.

Do not read significance into small mean gaps: samples are n=20, drawn with
replacement from 11 distinct episodes of a single held-out repository.
Per-sample MSEs for all three models ship in this repository's
`eval/` interchange JSONs for reanalysis.

## Protocol notes

- GT step 0 is the action at the anchor frame (current action + next 15).
- The primary metric is our frozen 16-step-chunk protocol — not upstream
  DreamZero's open-loop script, which scores only the first predicted step.
- DreamZero is evaluated in cache-reset mode (one independent 24-action chunk
  per sample; the auxiliary 24-step MSE is 0.00387). Deployment rollouts
  accrue KV-cache video context that this metric does not credit — a caveat
  inherited from upstream's own offline protocol.
- The previously published baseline numbers (π0.5 0.00206, GR00T 0.00279)
  came from two different protocols and are not comparable to this table;
  both checkpoints were re-scored under the frozen protocol here.

## Selection

Milestones every 500 steps were scored on a fixed held-out set:
deterministic flow-matching `val/loss` (seeded noise, step-comparable) plus
the frozen-protocol open-loop MSE. `val/loss` plateaued over steps
8,000–12,000 (range 0.01716–0.01720, adjacent-checkpoint noise ≈ 0.0001);
within that statistically-tied plateau, step 10,000 has the best open-loop
MSE (0.002524). We deliberately did not take the global MSE argmin over the
curve (an early checkpoint at step 2,000 scores 0.00232) because the
20-sample MSE noise band (±0.0007 between adjacent checkpoints) makes
argmin-over-24-points upward-biased toward luck; the full val curve is in
the WandB project.

## Training details

- Base: `GEAR-Dreams/DreamZero-AgiBot`, upstream commit `ab790c1`, with three
  build-time patches (in `patches/`): a `has_real_action` broadcasting fix and
  a collate-uniformity fix required for per-device batch > 1 (every prior
  public YAM run used batch 1), and a block-output unpacking fix for
  grad-disabled training-forward evaluation.
- LoRA r=32/α=32 on `q,k,v,o,ffn.0,ffn.2` + fully-trained action heads;
  lr 1e-5, warmup 1%, global batch 32 (per-device 4 × 8 GPUs), bf16,
  DeepSpeed ZeRO-2. Hyperparameters selected by a 3-arm warm-start trial
  (lr 1e-5 vs 5e-6 vs LoRA capacity r=4 vs r=32); the winning arm's
  checkpoint is a literal training prefix of this run.
- Data: LeRobot v3 → v2 conversion, H.264 re-encode, DreamZero GEAR
  metadata; relative 14-D joint actions (all dims), 33-frame / 3-camera /
  24-action chunks at 320×176.
- The `checkpoints/step-N/` folders are raw LoRA training milestones
  (research artifacts); the repository root is the merged, stock-loadable
  release.

## Loading

The release is merged (no PEFT required) and loadable through DreamZero's
standard `GrootSimPolicy` path. Note for anyone using the raw milestone
checkpoints instead: upstream's `load_lora` reconstructs the frozen DiT from
vanilla Wan2.1 rather than the AgiBot base these LoRAs were trained against —
use an explicit AgiBot-base reconstruction (see `src/reconstruct.py` in the
[training repo](https://github.com/robocurve/dreamzero-yam)).
