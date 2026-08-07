# Handoff: DreamZero × YAM (MolmoAct2) fine-tune

> Last updated: 2026-08-06
> Session focus: full project lifecycle — plan → pipeline → training → three-model evaluation → public release.

## Objective

Train and publish the first public DreamZero fine-tune on the MolmoAct2
BimanualYAM dataset, evaluated head-to-head against our π0.5 and GR00T N1.7
fine-tunes on the same data split under a single, machine-verified protocol.

## Current Status

**State**: RELEASED — project complete.
**Release**: https://huggingface.co/robocurve/dreamzero-yam-molmoact2 (public;
merged weights, model card with results, protocol, and compute provenance).
**Registry**: listed in [WorldPolicies](https://github.com/robocurve/worldpolicies).
**Result**: frozen-protocol open-loop MSE 0.002524 vs π0.5 0.002412 and GR00T
0.002492 — a statistical three-way tie achieved with 13–27× fewer training
samples. See `MODEL_CARD.md` for the full analysis and caveats.

## Technical Context

### Entry points
- `plans/0001-dreamzero-yam-finetune.md` — the reviewed plan plus run log
  (the run log at the bottom records every mid-run decision).
- `src/modal_app.py` — all training/benchmark/tune stages (Modal).
- `src/reconstruct.py` — **read this before touching checkpoints**: the only
  correct way to load `save_lora_only` checkpoints (upstream's `load_lora`
  silently rebuilds on the wrong base model).
- `src/eval_dreamzero.py` + `src/eval_spec.py` — the frozen 20-sample
  protocol, chunked causal evaluator, and cross-runner GT-identity harness.
  Sibling runners live in `pi05-yam-replication/eval_frozen.py` and
  `gr00t-n17-yam-replication/eval_frozen.py`.
- `src/val_monitor.py` — milestone harvesting + val/loss + MSE sidecar.
- `src/publish.py` — merge, stock-path publish gate, gated HF upload.
- `src/render_dreams.py` — imagined-vs-real video renderer.
- `patches/` — three upstream fixes applied at image build (see below).

### Key commands
```bash
modal run src/modal_app.py::smoke_test            # full integrity drill (~$15)
modal run src/modal_app.py::train --exp-name ...  # resumes from latest checkpoint
modal run src/val_monitor.py::val_monitor --exp-name ... --harvest-interval 500
modal run src/eval_dreamzero.py::eval_checkpoint --checkpoint <path>
modal run src/publish.py::publish                 # dry run + gate; --confirm uploads
```

### Environment notes
- Modal volumes (decommissioned 2026-08-06 after the HF archive was
  verified): `dreamzero-yam-ckpt` deleted; the `/dreamzero_gear` GEAR tree
  removed from the shared `gr00t-yam-data` volume (`/gr00t_v2` untouched —
  it belongs to the GR00T project). To retrain, re-run the download/convert
  stages in `src/modal_app.py`; to resume, pull
  `training-state/checkpoint-12000` from the archive repo.
- The training image pins upstream commit `ab790c1` and applies `patches/` at
  build; deps are a curated `--no-deps` set (see `modal_app.py` comments).
- H100-80GB cannot run this model at any working batch size; use H200/B200.

## Decision log (highlights)

| Decision | Rationale |
|---|---|
| B200:8, per-device batch 4 | Benchmark: ≈ H200 on $/sample, 37% faster wall-clock |
| LoRA r=32/α=32, lr 1e-5 | Won a 3-arm warm-start trial at every common milestone |
| 12,000 steps | Sized by val-curve need; halfway assessment confirmed continuation |
| Released step-10000 | Best MSE within the val/loss plateau (8k–12k); avoided the noisy MSE argmin |
| Frozen 20-index protocol + re-scored baselines | Original baseline numbers used two different protocols; the gap between them was ~4× exaggerated |
| Report a tie, lead with sample efficiency | Paired per-sample diffs ≈ 0.1σ; anything stronger would overclaim |

## What worked

- The smoke drill (kill/resume with bit-identity asserts, eval-reconstruction
  parity) caught every integration bug before the paid run; the main run then
  absorbed two real preemptions and two 24h-boundary restarts losing ≤56 steps.
- Monitor-side milestone harvesting densified the val curve to every 500 steps
  without touching the training container.
- The GT-identity + prompt-identity harness turned "same samples, same space"
  from an assertion into a machine-checked fact across three different stacks.
- CPU-first debugging: the dataloader "bottleneck" (30.7 s/step) was diagnosed
  for ~$5 on CPU containers as container-RAM starvation, not pipeline cost.

## What didn't work

> ⚠️ Do not retry without new information.

- **Upstream at per-device batch > 1** crashes twice over (broadcasting at
  wan_flow_matching_action_tf.py:795; collate on mixed chunk counts). Fixed in
  `patches/0001`, `0002`. Every prior public YAM run used batch 1.
- **Grad-disabled training forward** returns a raw tuple from the DiT block
  loop (`patches/0003`); this bit every eval-style use of `model(batch)`.
- **`load_lora` / `from_pretrained_for_tuning`** for these checkpoints: wrong
  base or silently dropped weights. Only `src/reconstruct.py` is correct.
- **torch DataLoader workers > 1** with stock 10k-step shards: ~41 GiB of
  decoded shard RAM per worker → container memory thrash (109–273 s/step).
  Small shards trip a `generate_shards` cutoff assert on this corpus. Use
  workers=1 with ≥512 GiB container memory.
- **Small-shard override (`num_steps_per_shard=2500`)**: fails on full corpus
  (episodes spanning >2 cutoffs).
- **HF cache on the checkpoint volume**: xet's open log file makes every
  `volume.commit()` fail; keep `HF_HOME` container-local in anything that
  commits volumes.
- **`--save-steps` on the train entrypoint** did not take effect (plumbing
  bug, never root-caused; run stayed at 100-step cadence, which turned out to
  reduce preemption losses).

## Open items (none blocking)

- [x] Upstream contributions filed (2026-08-06): PR
  [#86](https://github.com/dreamzero0/dreamzero/pull/86) (grad-disabled block
  unpacking, issue #85) and PR
  [#87](https://github.com/dreamzero0/dreamzero/pull/87) (collate uniformity,
  issue #80); the broadcast fix was already open as
  [#45](https://github.com/dreamzero0/dreamzero/pull/45) and is independently
  confirmed there. Watch for maintainer review feedback.
- [x] Training artifacts archived (2026-08-06) to
  [dreamzero-yam-molmoact2-checkpoints](https://huggingface.co/robocurve/dreamzero-yam-molmoact2-checkpoints)
  (26 weights-only milestones + resumable checkpoint-12000 + logs + renders);
  Modal volumes decommissioned after size-verified mirror. Intermediate
  DeepSpeed optimizer states (~2.4 TB) were deliberately discarded — resume
  from checkpoint-12000, or warm-start any milestone's weights via
  `--allow-fresh-optimizer`.
- [ ] NVML GPM profiling counters (tensor-active/DRAM-active) are unavailable
  inside Modal's gVisor; telemetry degrades to basic utilization.
- [ ] `--save-steps` plumbing bug above.
- [ ] Closed-loop deployment on a real YAM (TensorRT port required — the
  shipped TRT tooling is DROID-hardcoded) was explicitly scoped out; it would
  be a separate project.

## Recommended next steps (if this work continues)

1. **Closed-loop evaluation**: wire the released checkpoint into
   `inspect-robots-yam` via a `serve_dreamzero.py` adapter; the offline MSE
   parity makes a real-robot success-rate comparison the natural sequel.
2. **Longer training**: the val/loss curve had plateaued but not risen at
   12k steps; a continuation from `checkpoints/step-12000` is cheap to test.
3. **Upstream the patches** (see open items).

## Session notes

Development pattern that served this project: every substantive artifact
(plan, pipeline, eval stack) went through fresh-context adversarial review
rounds before money was spent on it; every paid stage had a cheaper gate in
front of it. The debug ladder in "What didn't work" is the distilled cost of
being first to run this model configuration — budget roughly 20 short GPU
iterations for first-contact with a research codebase of this size.
