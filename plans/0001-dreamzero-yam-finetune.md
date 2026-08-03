# Plan: DreamZero-YAM — first public DreamZero fine-tune on MolmoAct2-BimanualYAM

**Status:** draft for critique
**Date:** 2026-07-31
**Owner:** jeqcho / robocurve

## Goal

Post-train NVIDIA's DreamZero world-action model (base: `GEAR-Dreams/DreamZero-AgiBot`, 14B Wan2.1 DiT) on the **same 124-repo MolmoAct2 BimanualYAM split** used for our π0.5 and GR00T N1.7 replications, on Modal. Publish `robocurve/dreamzero-yam-molmoact2` to HF, add it to the WorldPolicies registry, and report held-out open-loop action MSE on `allenai/19012026-block-13` against our existing results:

| Model | Held-out open-loop MSE | Source |
|---|---|---|
| π0.5 (robocurve replication) | 0.00206 | `robocurve/pi05-yam-molmoact2` |
| GR00T N1.7 (robocurve) | 0.00279 | `robocurve/gr00t-n1.7-yam-molmoact2` |
| **DreamZero (this plan)** | ? | `robocurve/dreamzero-yam-molmoact2` |

**Why this exact scope:** (1) Nobody has published a DreamZero YAM fine-tune — the only artifact is an undocumented Ai2-internal single-task 2k-step LoRA dump (`swang23/dreamzero_yam_20260413_2000`). (2) Using the identical training split and held-out repo makes the number directly comparable to our two prior runs — a three-model-family comparison (VLA-JAX / VLA-PyTorch / world-action-model) on one benchmark, which is exactly the WorldPolicies pitch. (3) The official repo already ships a registered `yam` embodiment (state 14 / action 14 / 3 views, video concat order top,left,right) — the enum/config skeleton exists, though key names and action convention need adaptation (see Phase 0/1).

**Comparability requirement (load-bearing):** the two baseline numbers were NOT produced under one protocol — π0.5's 0.00206 is best-of-checkpoint-curve, n=20 random chunk samples, seed 0, raw absolute joint space (`pi05-yam-replication/eval_mse.py`); GR00T's 0.00279 is final-step-10,000 only, originally n=10 trajectories via Isaac-GR00T's evaluator. Comparing best-of-curve vs final-checkpoint is a selection-bias confound. Phase 5 therefore **re-evaluates all three published checkpoints under one script** (standardizing on the π0.5 protocol, the cheapest to port) and the model card reports per-row protocol.

## Non-goals (v1)

- Full 720 h / paper-scale full fine-tune (100k steps, batch 128): ≈$40k ±2× and needs Modal multi-node private beta or ~2 months single-node. Out of scope; revisit only if v1 results justify it.
- Closed-loop real-robot success rates as a *gating* deliverable. Stretch phase only; model card carries the same "open-loop MSE proxy" caveat as the prior two.
- Training from the DROID base or from scratch. Base is DreamZero-AgiBot (bimanual, the recipe NVIDIA used for their own YAM adaptation).

## Key facts the plan builds on (from research)

- **Recipe precedent:** community LoRA fine-tunes (Vizuara SO-101 2×H100/127 h; andreaskoepf DK-1 bimanual 4×H100/55 h; Ai2's own YAM baseline) all use PEFT LoRA (r=4 default, targets `q,k,v,o,ffn.0,ffn.2`) with action encoder/decoder heads fully trained. **The official `yam_training.sh` recipe: lr 1e-5, warmup 5%, per-device batch 4, `train_architecture=lora`, `save_lora_only=true`** (community runs from the AgiBot ckpt used lr down to 5e-6).
- **Action convention:** DreamZero's YAM pipeline is **relative** — `yam_relative.yaml` sets `relative_action: true` on all 14 dims and the converter emits `relative_stats_dreamzero.json` (action − reference state). Our dataset stores absolute joints; the loader derives relative on the fly (supported path), but **eval must un-normalize and add back the reference state** to score in raw absolute joint space, or the MSE is meaningless against the baselines.
- **Stack:** `torchrun --standalone` + DeepSpeed ZeRO-2, single node, no SLURM — fits a Modal 8-GPU container directly. flash-attn must be baked into the image. Torch 2.8.0 cu129 — fine on Modal H100/H200/B200 (avoid B300).
- **Data gap:** DreamZero's converter wants LeRobot **v2** (per-episode parquet+mp4); MolmoAct2 ships **v3** with AV1 video. Our `gr00t-n17-yam-replication` `prepare_data.py` already orchestrates this (Isaac-GR00T's upstream v3→v2 converter + a separate `libx264` re-encode pass, since torchcodec/decord can't do AV1) — its output may still sit in the Modal data volume.
- **Modal constraints:** 24 h function cap (solved by our existing retry+resume+volume-commit-thread pattern), FUSE volumes slow for random video reads (copy training set to container NVMe at startup, ~800 GB split → use per-repo subsetting or accept ~1 h copy).
- **Known model gotchas:** AR rollout MSE degrades ~6× vs teacher-forced (dk1: 0.050→0.317); imagined video drifts (arm 5–30% slow, post-task incoherence); language conditioning can collapse after ~2 chunks. Report both teacher-forced and AR-rollout MSE.

## Phases

Costs use Modal on-demand list: H100 $3.95/GPU·h, B200 $6.25/GPU·h. Throughput estimate ±2× until Phase 3 calibrates it. **Each phase ends with an abort/continue gate.**

### Phase 0 — Recon & budget gate (~$0, <1 day)
1. ✅ **Verified 2026-07-31:** volume `gr00t-yam-data` still holds the converted v2 tree (`/gr00t_v2/allenai/<repo>` incl. held-out `19012026-block-13`, plus `datasets.json`); sibling volumes `gr00t-yam-ckpt`, `pi05-yam-{data,ckpt}` also intact. Phase 1 is a light transform, not a re-run. Remaining recon: This decides whether Phase 1 is a re-run or a light transform. Also check the volume's **version** (Volumes v2 supports hardlinks; v1 does not — determines the Phase 1 tree-sharing approach).
2. Check Modal credit balance and set a workspace spend cap ≥ the Phase 4 budget.
3. **Known config mismatches to resolve (not "confirm match"):** the repo's `modality_config_yam` was built for Ai2's internal `YAM_play_data` naming — video keys `top_camera-images-rgb/left_camera-images-rgb/right_camera-images-rgb`, state/action sub-keys `left_joint_pos, left_gripper_pos, right_joint_pos, right_gripper_pos`. Our data has `observation.images.top/left/right`, and `convert_lerobot_to_gear.py` auto-strips prefixes to `top/left/right` with **no rename flag**. Decide the mapping approach here: pass `--state-keys/--action-keys` JSON with DreamZero's exact sub-key names, and post-edit each repo's `modality.json` video entries (or patch the base YAML) to the `*_camera-images-rgb` names. Also note the repo is internally inconsistent on the annotation key (`annotation.task` in config vs `annotation.human.task_description` in the script comment) — resolved empirically at smoke, not assumed. Regenerate q99 + relative stats for the 124-repo split regardless.
4. Pin the dreamzero commit; confirm `GEAR-Dreams/DreamZero-AgiBot` (~45–65 GB) + Wan2.1-I2V-14B-480P (~28 GB) + umt5-xxl download cleanly (check gating).

**Gate:** volumes + configs verified, credits confirmed.

### Phase 1 — Data prep (CPU-only Modal workers, ~$20–50)
1. Reuse/re-run the GR00T repo's `prepare_data.py` flow for the 124 repos (42 block + 42 box + 40 charging); exclude held-out `19012026-block-13` from training data, stage it separately for eval. Note: the actual v3→v2 converter it invokes is **Isaac-GR00T upstream** (`scripts/lerobot_conversion/convert_v3_to_v2.py`, needs `lerobot>=0.4.0`), and it stream-copies video (stays AV1) — the H.264 re-encode is a separate `libx264` pass. Keep this in a **separate CPU image** so the DreamZero training image isn't polluted with Isaac-GR00T deps.
2. Run `scripts/data/convert_lerobot_to_gear.py` with the key-mapping decided in Phase 0.3 → GEAR metadata (`modality.json`, `stats.json`, `relative_stats_dreamzero.json`, `embodiment.json`) per repo; post-edit video key names to `*_camera-images-rgb`; dataset YAML listing all 124. **Tree ownership:** the converter writes `meta/` in place (skip-if-exists without `--force`) — running it on the GR00T v2 tree either silently keeps GR00T's incompatible `modality.json` or, with `--force`, clobbers the metadata the Phase 5 GR00T baseline re-eval needs. So DreamZero gets its **own sibling prefix on the same volume** (hardlinks can't cross filesystems): hardlink `data/` and `videos/` from the GR00T tree, generate `meta/` fresh there, leave the GR00T tree untouched. Seeding the fresh `meta/`: the converter hard-exits without `info.json` and skips existing files — copy **only** `info.json`, `tasks.jsonl`, `episodes.jsonl` from the v2 tree (never `modality.json`/`stats.json`, or skip-if-exists silently keeps GR00T's incompatible schema) (avoid `--output-path`, which `copytree`s ~800 GB). **Caveat: hardlinks require Modal Volumes v2** — Phase 0 checks the existing volume's version; if it's v1, fall back to symlinks (supported) with a loader smoke-read through them, or assemble the tree in-container at NVMe-copy time (data gets copied to local NVMe for training anyway), or as a last resort a one-time copy into a new v2 volume with the storage cost added to the budget.
3. Validate: episode counts (~5,145 train), decodable H.264, action length = horizon × chunk size, relative stats sane (≈ zero-mean since actions ≈ states), and a loader dry-run that exercises the exact key names. **Plus a degenerate-range assert:** `q99 − q01 > ε` per dim on both the per-repo relative stats and (later, at Phase 2) the checkpoint's merged metadata — the normalizer masks degenerate dims on `apply` but NOT on `inverse`, so a collapsed dim (plausible for relative gripper dims, ≈0 most steps) would silently pin every decoded action to `q01` at eval. If any dim is truly degenerate, decide handling before training.

**Gate:** loader smoke-reads N random samples from every repo without error.

### Phase 2 — Image + smoke test (~$10–20)
1. Modal image: CUDA 12.9 devel base, torch 2.8.0 cu129, flash-attn compiled at image-build (`MAX_JOBS=8`, no-build-isolation), deepspeed, peft 0.5.0, transformers 4.51.3, decord2, dreamzero at pinned commit. HF + WandB secrets.
2. 20–50 step LoRA smoke on **2×H100**: proves data path, LoRA save (`save_lora_only`), checkpoint→resume, and our volume-commit background thread — the full preemption drill from `gr00t-n17-yam-replication/modal_app.py`.
3. Smoke also settles the repo's internal inconsistencies empirically: annotation key (`annotation.task` vs `annotation.human.task_description`) and `max_chunk_size` (yaml says 5, training script overrides to 4 — we follow the script). Note LeRobot v2 parquet carries numeric `task_index`, not an `annotation.*` column — the converter's auto-detect finds nothing, so the modality.json post-edit must map it explicitly (`annotation: {"task": {"original_key": "task_index"}}`); the loader resolves numeric index → `tasks.jsonl` and fails loudly on subkey mismatch, but the dry-run additionally asserts prompts are **non-empty strings** (an empty-language run would train silently).
4. **Pretrained-load integrity assert:** `create_model` loads the 45 GB AgiBot checkpoint with `strict=False` and discards the missing/unexpected-keys return. Smoke captures both lists at load and asserts empty (modulo the known-absent fresh-LoRA keys) — otherwise a commit bump or config drift silently trains a partially random-init 14B for the whole main run.
5. **Eval-reconstruction parity drill (BLOCKER-class landmine found in review):** with `save_lora_only=true`, the stock eval path (`GrootSimPolicy` → `load_lora`) rebuilds the frozen DiT from **vanilla `Wan2.1-I2V-14B-480P`**, NOT the AgiBot-post-trained base the LoRA was trained against — the two bases differ materially (verified ~100% mean relative diff on sampled DiT tensors). Every naive cross-check passes because the reference script uses the same wrong path. **No shipped one-call API does the right thing** (`from_pretrained` takes no LoRA path; `from_pretrained_for_tuning(lora_weights_path=...)` skips `inject_lora_after_loading`, so checkpoint keys carrying the PEFT `base_model.model.` prefix all land as *unexpected* keys and are silently dropped — you'd evaluate pure DreamZero-AgiBot and read the bad number as "LoRA underperforms"). Our eval/serving therefore implements the **training-identical sequence explicitly**: instantiate `VLA(config)` with `defer_lora_injection=true` + `skip_component_loading=true` → load the AgiBot shards → `action_head.inject_lora_after_loading()` → `load_lora_weight(<ckpt>)`.
   **Parity asserts at smoke (designed to fail in the failure modes, not around them):** (a) unexpected keys == ∅ AND the checkpoint's key set == the trained-parameter name set recorded at training time (NOT "missing keys empty" — `load_lora_weight` loads a LoRA-only dict into the full 14B with `strict=False`, so missing always contains every frozen base param; it also only *prints* the lists and returns `None`, so capture them via a `load_state_dict` wrapper); (b) bit-identity vs the in-training model of one LoRA-target `base_layer` tensor AND one `lora_A`/`lora_B` tensor AND one action-head tensor (a frozen-DiT-only probe passes even when the whole overlay is dropped); (c) bit-identical policy output on one fixed sample vs the in-training model. Also: checkpoint `config.json` bakes in training-container paths for dit/text-encoder/VAE — eval containers mount identically or override.
6. **Resume drill hardened for the `save_lora_only`+DeepSpeed path:** the overridden `save_model` writes only trainable params; on resume the base is repopulated via `create_model` with trainer/DeepSpeed state loaded on top — whether adapters/heads/optimizer state survive that boundary is exactly what's new vs our proven GR00T machinery. A resume that restores the step counter but re-initializes adapters would quietly corrupt the main run at the first 24 h boundary. Smoke therefore does a kill/resume cycle and asserts bit-identical LoRA + head tensors, an optimizer-state hash match, and loss continuity over ~20 post-resume steps.

**Gate:** loss decreases, kill/resume passes the hardened asserts, checkpoint files complete, no key-name errors, load-integrity and eval-reconstruction parity asserts pass.

### Phase 3 — Throughput calibration + warm-start tune (~$250–450)
1. 100-step benchmark **across a small GPU-config matrix, selected by cost-efficiency** (requirement from jeqcho 2026-07-31: pick the config that minimizes total $ for the same trained samples, not a default). Cost for a fixed sample count = $/GPU-hour ÷ samples/GPU-hour, so the decision metric is **$ per 1k samples**, subject to projected wall-clock ≤ ~1 week (resume machinery spans the 24 h cap either way). Matrix: 8×H100 ($31.6/h), 4×H100 ($15.8/h — tests whether ZeRO-2 comm overhead makes fewer GPUs *more* efficient per GPU), 8×B200 ($50/h — 192 GB VRAM may allow larger per-device batch; wins if throughput/$ beats H100), 8×H200 ($36.3/h — more VRAM headroom at +15%). ~$30–60 of benchmark spend to potentially save thousands. All runs **read from container-local NVMe, not the FUSE volume** (33 frames × 3 views per sample makes throughput I/O-sensitive; benchmarking off the volume would mis-calibrate Phase 4) and **sweep `dataloader_num_workers` and container CPU count** — the official recipe pins `num_workers=1` while each sample decodes ~25–33 frames × 3 views; if throughput comes in slow, the cheap fix is workers/CPUs, not accepting the 2× GPU-cost case. Winner's real samples/s → recompute Phase 4 cost (research estimate: ~0.3 samples/s/GPU on H100; ±2×). Tune arms and the main run then use the winning config.
2. **GPU utilization telemetry in every GPU stage** (benchmark, arms, main run): logged to WandB alongside loss, reusing the SO-101 repo's `gpu_metrics.py` tiered approach (DCGM first, tagged fallback — the DCGM hostengine is blocked under Modal's gVisor) and FLOPs/MFU accounting. **Required metrics (jeqcho 2026-07-31): profiling-grade counters, not just kernel-residency "utilization" —** `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` and `DCGM_FI_PROF_DRAM_ACTIVE` (plus SM_ACTIVE and PCIe/NVLink traffic where available). Under gVisor, obtain the equivalents via **NVML GPM** (Hopper+/Blackwell, i.e. every config in our matrix): `NVML_GPM_METRIC_ANY_TENSOR_UTIL` ≈ PIPE_TENSOR_ACTIVE, `NVML_GPM_METRIC_DRAM_BW_UTIL` ≈ DRAM_ACTIVE, sampled ~10 s cadence, logged per GPU with the `source_tier` tag (dcgm / nvml-gpm / nvml-basic) so dashboards state what they're measuring. **FLOPs:** log both achieved model TFLOPs/s per GPU (analytic FLOPs-per-sample × throughput, the SO-101 accounting) and **MFU** against a per-architecture dense-BF16 peak table (H100/H200/B200 differ — MFU is the cross-config comparable number and feeds the $/1k-samples decision). **Live view ("nvtop"):** bake `nvtop` into the training image; while any stage is running, `modal container list` + `modal container exec <id> -- nvtop` gives the live per-GPU TUI on demand — document the exact two commands in the README. WandB system panels carry the historical curves. This is the diagnostic that separates "model is this slow" from "we're paying for idle GPUs": sustained util below ~85% or low MFU with high dataloader wait ⇒ fix workers/CPUs/prefetch before buying more GPU-hours, and the Phase 4 gate reviews the util traces before approving the spend.
3. Warm-start tuning per our `TUNING.md` method — full-schedule single-change arms, equal wall-clock, instability screen (NaN, spike ≥1.3×, rising tail, grad-norm >3× median), winner's checkpoint is a prefix of the main run. **Warmup: ratio ≈0.01 (250–300 steps) for the arms AND the main run** — they must match for the warm-start-prefix property to hold, and TUNING.md's own rule is that trials must get *past* warmup within their budget (the official 5% of a 25–30k schedule would leave every arm on the lr ramp for its whole trial). Arm trial length sized so each arm gets meaningful post-warmup observation (~2× warmup steps minimum past the ramp).
   - **Arm A (control = official recipe apart from warmup):** LoRA r=4/α=4, lr 1e-5, per-device batch 4 (global 32)
   - **Arm B:** lr 5e-6 (community value for AgiBot-ckpt starts)
   - **Arm C:** LoRA **r=32/α=32** — α scales with r so PEFT scaling (α/r) stays 1.0 and this remains a single-variable capacity check (r=32/α=8 would silently down-scale the adapter update 4×). Rationale: 5k episodes is far beyond the 55-trajectory few-shot regime the r=4 default was tuned for; the paper itself flags LoRA as suboptimal at scale.
4. Val-monitor sidecar (1×H100): logs to the **same WandB project** as the trainer so train and held-out curves sit side by side (requirement from jeqcho 2026-07-31). Metrics: `train/loss` (from the trainer itself, `report_to=wandb` + WANDB secret); from the sidecar per milestone checkpoint: `val/loss` — the *same flow-matching objective as training* computed on frozen held-out samples with a seeded `fork_rng` (SO-101 pattern) so it's step-comparable to `train/loss` and reads directly as train-vs-test loss — and `val/open_loop_mse` (teacher-forced, the checkpoint-selection metric). Built by **adapting the repo's own `scripts/open_loop_yam.py`** (single-GPU, absolute-action output) — not a from-scratch port. Known traps to fix in the adaptation: **its checkpoint loading uses the defective stock `load_lora` path (wrong base — see Phase 2.5); replace with the explicit training-identical reconstruction sequence from Phase 2.5**; it scores only the first predicted timestep (`pred_val[0]`), not 16-step chunks; its hardcoded `STATE_SLICES`/`ACTION_SLICES` assume Ai2's 46-dim internal state layout (ours is 14-dim — copying verbatim silently mis-maps joints); verify per-sample statelessness (fresh `latent_video=None` per index). Run on milestone cadence rather than continuously to bound cost.

**Gate:** winning arm chosen; projected Phase 4 cost within budget. If all arms plateau with val MSE >10× π0.5's 0.00206, stop and reassess (bigger rank / more steps / full-FT decision goes back to jeqcho).

### Phase 4 — Main run (central estimate ~$3–3.5k; range $1.5–7k until Phase 3 calibrates)
- Winning config, **8×H100** (fallback H200 +15% if H100 capacity is tight), global batch 32, **~25–30k steps** = 0.8–0.96M samples ≈ **7–8.5% of the 11.35M chunk positions**. At the estimated 2.4 samples/s aggregate that's ~93–111 h wall (spanning the 24 h cap via resume), ≈ $2.9–3.5k central; the −2× throughput case is ~$7k, which is why the Phase 3 benchmark is a hard gate.
- **Known confound, stated up front:** π0.5 saw ~10.2M samples (~90% of an epoch) and the GR00T reference ~5.1M — DreamZero will see **5–10× less data** than the baselines. Matched coverage ≈ the $40k non-goal, so v1 accepts the confound, documents it in the model card, and the Phase 3/4 gate logic treats an early val plateau as possibly data-starvation rather than pure capacity shortfall.
- **Sampler knob to know before extending steps:** `yam_relative.yaml` sets `dataset_shard_sampling_rate: 0.1` — each shard visit yields only 10% of its steps (sampler epoch ≈ 1.135M samples). Our 0.8–0.96M samples stay inside one sampler epoch, so the coverage math holds; but pushing past ~35k steps wraps onto a *fresh random 10%* rather than linearly increasing coverage — any "train longer" decision at the gates must account for this (or raise the rate).
- `--detach`; retries=3 with auto-resume from latest complete checkpoint; volume commit every ~150 s; milestone copies every 2,500 steps (Young–Daly cadence re-derived for the ~2 GB LoRA+heads checkpoint size — small, so cadence can be tight); sliding window save_total_limit=5.
- Sidecar val monitor keeps running; best-checkpoint selection by held-out teacher-forced MSE, not train loss (π0.5 lesson: best was step 10k of 20k).
- **Milestone checkpoints upload to HF as they land** (jeqcho 2026-07-31): each rotation-proof milestone is pushed to `robocurve/dreamzero-yam-molmoact2` under `checkpoints/step-<N>/` using the **`hf-robocurve-write`** secret (verified: user jeqcho, robocurve org) — off-Modal durability plus the ability to eval any milestone from anywhere; final merged publish (Phase 6) overwrites the repo root.

**Monitoring:** `modal app logs` + WandB link in README; no tmux needed — Modal `--detach` fills that role.

**Mid-run efficiency intervention policy (jeqcho, 2026-07-31):** the telemetry stack runs through the scans *and* the main run, and there is standing authority to **kill and restart the main run with a fixed configuration whenever the projected net saving exceeds $500** (projected = remaining-run cost at current efficiency − remaining-run cost at fixed efficiency − restart overhead; overhead is small since resume picks up from the last ~150 s volume commit). Interventions in order of preference:
1. **Data-latency fixes (no world-size change):** more dataloader workers / container CPUs / prefetch depth, decode parallelization — resume is seamless, do these aggressively at the first sign of input starvation (low tensor-active + low DRAM-active).
2. **GPU type swap at same count** (e.g. 8×H100 → 8×H200/B200): world size unchanged, optimizer state resumes cleanly; per-device batch may be raised only via grad-accum-neutral configs that keep **global batch fixed at 32** — mid-run global-batch changes would confound the training dynamics and are not allowed.
3. **GPU count change** (last resort): DeepSpeed ZeRO-2 shards optimizer state by world size, so a naive resume on a different count breaks — requires DeepSpeed universal-checkpoint conversion, or accepting a restart from the latest milestone with fresh optimizer state (documented in the run log if taken; only when savings clearly dominate the warmup-again cost).

**Gate:** val MSE curve plateaued or budget consumed.

### Phase 5 — Eval (~$70–150, includes baseline re-evals)
1. **Unified protocol (implementation requirement, not aspiration):** a small shared **spec**, not literally one script — the three stacks (openpi/JAX, Isaac-GR00T, dreamzero) can't share an image. The spec is a checked-in frozen list of 20 (episode, frame) sample indices on `19012026-block-13` plus one metric function: MSE in **raw absolute joint space**, first 16 steps of the predicted chunk. Indices are frozen once and hardcoded (not derived from `rng.randint(0, len(ds))` at runtime, which silently shifts with lerobot-version-dependent `len(ds)`), and constrained to `frame ≤ ep_len − 24` (the max horizon of the three models) — the v3 API pads chunks past episode end by repeating the last frame while v2 parquet readers don't, so near-end indices would fail the GT-equality assert by construction. **Sample identity is proven, not assumed:** the π0.5 runner reads the v3 HF repo while GR00T/DreamZero read the v2 + H.264 conversion, whose documented frame jitter/deficits could shift frames — so each runner must emit the raw GT action array at every frozen index, and the harness asserts equality across all three **after casting to a canonical dtype (float64)** — the v3 API may hand back float32-cast torch tensors while v2 parquet reads differently, and a spurious dtype mismatch must not masquerade as the frame-shift failure mode — plus one visual frame spot-check, before any MSE enters the comparison table.
2. **DreamZero decoding path — the silent-failure hotspot:** training merges all 124 repos' stats into one per-embodiment `merged_metadata` and dumps it to the checkpoint's `experiment_cfg/metadata.json`; the policy reloads that and returns **already-absolute** actions. Eval therefore goes through the policy's own unapply path with **two hard requirements**: (a) stats = the checkpoint's merged `metadata.json`, never any single repo's `relative_stats_dreamzero.json` (a GT round-trip test passes for *any* stats file, so it can't catch this); (b) model reconstruction = **AgiBot base + LoRA overlay** via the explicit training-identical sequence in Phase 2.5, never stock `load_lora` (which rebuilds on vanilla Wan2.1; the unmodified `open_loop_yam.py` has this same defect, so it is a reference for the obs/decoding plumbing, NOT for checkpoint loading). Validity checks: stats-identity assert + the Phase 2.5 parity asserts (keys-empty, base_layer/lora/head tensor bit-identity, fixed-sample output identity).
3. **Re-evaluate the published baselines under the same spec:** `robocurve/pi05-yam-molmoact2` and `robocurve/gr00t-n1.7-yam-molmoact2` (~$20–50 of GPU time). The existing 0.00206 / 0.00279 came from different protocols (best-of-curve n=20 chunk samples vs final-step n=10 trajectories) — the model card table uses the re-evaluated numbers as the primary comparison and footnotes the originals. Footnotes describe the original protocols **from the code, not the old card text** (the π0.5 card's "n=20 trajectories, quantile-normalized" mischaracterizes what `eval_mse.py` actually does: n=20 random samples, raw absolute space).
4. **Checkpoint selection parity:** DreamZero's reported number is best-of-val-curve (like π0.5); the table's protocol column says so per row. Also report full-24-step MSE as a secondary number.
5. **Secondary (honest):** autoregressive rollout MSE over ≥12 chunks (the community drift metric) — expect ~6× worse; report it anyway.
6. **Qualitative:** dump imagined-video rollouts for 5–10 held-out episodes side-by-side with real footage. (These are also the marketing asset — the model *dreams* the YAM doing the task.)

### Phase 6 — Publish

**Model-card statistics wording (from the eval-stack review, 2026-08-02):**
publish per-sample MSEs (or at least the std) from the interchange JSONs;
frame the comparison as *paired* per-sample differences (all models score
identical samples — the protocol's strength); claim no significance for
small mean gaps at n=20 clustered samples (11 distinct episodes, one repo);
say "GT step 0 is the action at the anchor frame"; the primary metric is
the 16-step-chunk MSE under OUR frozen protocol — do not describe it as
upstream's metric (upstream's open-loop script scores only the first
predicted step). Note DreamZero is scored in cache-reset mode; deployment
rollouts accrue KV-cache video context this metric doesn't credit (caveat
inherited from upstream's own protocol).
 (~$0)
- HF `robocurve/dreamzero-yam-molmoact2`: **merged weights produced from the correct explicit reconstruction** (`merge_and_unload` on that model — publishing LoRA-only adapters alone would hand every downstream `GrootSimPolicy` user the wrong-base landmine) plus the raw LoRA adapters for provenance, `experiment_cfg` incl. merged `metadata.json`, model card with the three-model comparison table, training curves, AR-drift caveat, the wrong-base warning for anyone using stock `load_lora`, and imagined-video GIFs.
- **The published artifact must be loadable by the STOCK path, which requires config rewrites** — merged weights alone aren't enough: ship `config.json` with `train_architecture: "full"` + `defer_lora_injection: false` and `experiment_cfg` with `save_lora_only: false` (otherwise stock `GrootSimPolicy` either routes to `load_lora` → vanilla-Wan base, or injects LoRA at init so plain merged keys mismatch `base_layer.*` model keys and `strict=False` swallows a near-random model). Include the full state dict (DiT+VAE+T5+image-encoder, mirroring the AgiBot ckpt layout); note `from_pretrained` ignores its `config` argument — only the shipped `config.json` matters.
- **Publish gate:** load the published artifact through *stock* `GrootSimPolicy` and assert fixed-sample output identity against the explicit reconstruction before announcing.
- Add to WorldPolicies collection + registry with eval numbers; CI validation.
- README updates in the new repo per our conventions.

### ~~Phase 7~~ — OUT OF SCOPE (reference only): closed-loop on the real YAM
- Inference server needs min 2 GPUs; ~3 s/chunk on 2×H100 ($7.9/h) — too slow for smooth control. The 7 Hz paper number is DreamZero-Flash + TensorRT NVFP4 on Blackwell; Modal **B200** is Blackwell, but this is a **porting task, not a script run**: `build_trt_engine.sh` calls a `build_trt_engine_droid.py` that doesn't exist, and the shipped `build_trt_engine.py` is hardcoded to the DROID model type, embodiment tag, and calibration data. Budget porting effort into the go/no-go.
- Wire into `inspect-robots-yam` (new `serve_dreamzero.py` following `serve_gr00t_act.py` — dreamzero's own WebSocket protocol won't match ours either).
- Latency over the internet Modal↔robot is the open risk; measure before promising anything.

## New repo layout

```
robocurve/dreamzero-yam/
├── README.md               # goal, results table, monitoring instructions
├── CLAUDE.md               # agent context: pipeline map, gotchas, volume names
├── plans/
│   └── 0001-dreamzero-yam-finetune.md   # this plan, finalized
├── src/
│   ├── modal_app.py        # staged: prepare_data → smoke → tune → train → eval
│   ├── prepare_data.py     # adapted from gr00t-n17-yam-replication; invokes Isaac-GR00T's
│   │                       #   upstream convert_v3_to_v2.py in a separate CPU image
│   ├── eval_spec.py        # frozen 20 (episode,frame) indices + shared metric fn
│   ├── eval_dreamzero.py   # runner via GrootSimPolicy (adapted open_loop_yam.py)
│   ├── eval_baselines.md   # how the pi05/gr00t runners consume eval_spec (live in
│   │                       #   their own repos/stacks; results checked in here)
│   ├── val_monitor.py      # sidecar, adapted
│   └── publish.py          # HF upload
├── configs/
│   ├── yam_dataset.yaml    # 124-repo list (reuse repos.json)
│   └── arms/               # tuning arm configs
└── outputs/                # eval results, curves (small files only)
```

## Budget summary

| Item | Est. cost |
|---|---|
| Phases 0–2 (recon, data, smoke) | $30–70 |
| Phase 3 (calibrate + 3-arm tune, arms sized to clear the 0.01 warmup) | $250–450 |
| Phase 4 (main run) | central $2.9–3.5k; range $1.5k (fast) – $7k (−2× throughput); Phase 3 gate decides before spend |
| Val sidecar (1×H100, milestone cadence through Phases 3–4) | $100–250 |
| Phase 5 (eval incl. baseline re-evals) | $70–150 |
| Storage (if data volume must be recreated) | ~$70–105/mo, prorated |
| **Total core** | **central ~$3.4–4.5k; worst-case ~$8k, gated** (trim steps at Phase 3 if calibration comes in slow) |
| Phase 7 stretch (serving) | $12.5/h while live + TRT build ~$50 |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Wrong-base or dropped-overlay eval reconstruction** (`load_lora` rebuilds on vanilla Wan2.1; `from_pretrained_for_tuning(lora_weights_path=...)` silently drops LoRA+heads as unexpected keys; reference script shares the defect so naive cross-checks pass) | Explicit training-identical sequence (init deferred → AgiBot shards → `inject_lora_after_loading` → `load_lora_weight`); Phase 2.5 parity asserts (unexpected==∅ + ckpt-keys==trained-set, tensor bit-identity, fixed-sample output identity); publish merged weights + rewritten configs, gated on stock-path load parity |
| Resume across the 24 h boundary silently re-initializes adapters/optimizer (new `save_lora_only`+DeepSpeed path, unproven vs our GR00T machinery) | Phase 2.6 kill/resume drill: bit-identical LoRA+head tensors, optimizer-state hash, 20-step loss continuity |
| `meta/` collision between GR00T and DreamZero stacks on the shared v2 tree | Separate DreamZero volume prefix with hardlinked data/videos and fresh `meta/`; GR00T tree untouched for Phase 5 |
| Relative→absolute decoding botched at eval (silently wrong headline number) | Decode via policy unapply path with the checkpoint's merged `experiment_cfg/metadata.json` (never per-repo stats); stats-identity assert (a GT round-trip test alone is blind to wrong-stats errors) |
| Silent partial load of the 45 GB base (`strict=False`, return discarded) | Smoke captures missing/unexpected keys and asserts empty modulo fresh-LoRA keys |
| v2 conversion frame-shift makes cross-stack samples non-identical | Harness asserts bitwise-equal GT actions at all frozen indices across the three runners before accepting any MSE |
| `open_loop_yam.py` adaptation traps (first-timestep-only scoring; 46-dim `STATE_SLICES` vs our 14-dim) | Both called out in Phase 3.3/5; slices rewritten for 14-dim layout; GT-identity harness catches mis-mapping |
| Modality key mismatch (KeyError or, worse, silently mis-mapped cameras) | Phase 0.3 explicit mapping + Phase 1 loader dry-run + smoke gate on exact key names; camera order verified visually on one decoded sample |
| LoRA under-capacity at 5k-episode scale (paper: "suboptimal" beyond few-shot) | r=32 arm in Phase 3; escalate to jeqcho with data before any full-FT decision (full-FT 8×H100 + CPU-offload is possible but ~3–4× cost) |
| Baselines saw 5–10× more data (coverage confound) | Stated in model card; plateau interpretation in gate logic; matched-coverage run explicitly deferred |
| MSE not apples-to-apples (24- vs 16-step horizon, protocols differ across baselines) | One unified eval script re-scores all three published checkpoints; first-16-steps primary; per-row protocol column |
| v3→v2 conversion subtly wrong (timestamp jitter, frame deficits documented in prior runs) | Reuse the already-debugged GR00T conversion incl. its 5 ms decode tolerance lessons; per-repo validation pass |
| Modal preemption / 24 h cap mid-run | Proven retry+resume+commit-thread+milestone machinery, reused verbatim |
| Throughput estimate off ±2× | Phase 3 benchmark before committing Phase 4 spend; hard gate |
| lr instability (GR00T lessons: 4e-4 stuck, 6e-4 NaN) | Conservative lr arms (5e-6/1e-5), instability screen, full-schedule warm-start method |
| HF gating on base ckpts / dataset | Phase 0 verifies all downloads before any GPU spend |
| AR rollout drift makes demo videos embarrassing | Teacher-forced videos for the model card; AR drift reported as a known limitation with the dk1 precedent cited |

## Decisions (jeqcho, 2026-07-31)

1. **Budget: $5k soft target, $8k hard cap** (jeqcho, 2026-07-31 22:54: "ideally this should be lower tho, but just in case things happen overnight, i don't want u to be blocked"). Interpretation: plan and optimize as if the budget were $5k — trim/choose configs to land near the $3.4–4.5k central estimate; the $5k→$8k band exists solely so an overnight surprise (slow calibration, a preemption-heavy night, a justified >$500-savings restart) doesn't stall the pipeline waiting for approval. Crossing $5k projected spend requires a logged justification in the run log. Update 2026-08-02: additional emergency headroom exists beyond the cap (jeqcho), but it is not a target — size the main run by val-curve need (start ~12-15k steps, extend per milestone val MSE), never by available budget.
2. **Repo name: `robocurve/dreamzero-yam`.**
3. **Phase 7 (closed-loop): out of scope** (jeqcho, 2026-07-31). The project ends at Phase 6 (publish). Phase 7's section is retained below as reference only, in case results ever motivate reviving it as a separate project.
