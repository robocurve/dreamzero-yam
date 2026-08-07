# DreamZero-YAM agent context

Implement against `plans/0001-dreamzero-yam-finetune.md`; the plan wins over upstream reference code. Never run Modal, download checkpoints, or start a GPU job unless the user explicitly asks for execution.

## Durable storage and tree ownership

**PROJECT RELEASED AND DECOMMISSIONED (2026-08-06).** The `dreamzero-yam-ckpt` volume was deleted and `/data/dreamzero_gear` removed from `gr00t-yam-data` after a size-verified HF archive. Artifacts now live at HF `robocurve/dreamzero-yam-molmoact2` (merged release) and `robocurve/dreamzero-yam-molmoact2-checkpoints` (weights-only milestones, resumable checkpoint-12000, logs, renders). To work on this project again, re-run the download/convert stages first; the layout below describes what they rebuild.

- Existing data volume: `gr00t-yam-data`, mounted at `/data`.
- New checkpoint volume: `dreamzero-yam-ckpt`, mounted at `/checkpoints`.
- GR00T-owned input: `/data/gr00t_v2/allenai/<repo>`. Never edit its metadata or payload.
- DreamZero-owned sibling: `/data/dreamzero_gear/allenai/<repo>`.
- The DreamZero tree owns fresh `meta/`. It shares `data/` and `videos/` with the GR00T tree through hardlinks on Volumes v2, falling back to directory symlinks after a smoke-read.
- GPU jobs copy the 124 training repos to `/local/dreamzero_gear/allenai/<repo>` before `torchrun`; training must not random-read video from the FUSE volume.

## Model paths and wrong-base landmine

The stable paths baked into checkpoint `config.json` are:

- `/checkpoints/DreamZero-AgiBot`
- `/checkpoints/Wan2.1-I2V-14B-480P`
- `/checkpoints/umt5-xxl`

**Never use stock `VLA.load_lora`, `GrootSimPolicy`'s `save_lora_only` branch, or `from_pretrained_for_tuning(lora_weights_path=...)` for evaluation.** Stock `load_lora` reconstructs the frozen DiT from vanilla Wan2.1 instead of DreamZero-AgiBot; the tuning helper injects at the wrong time and silently drops overlay keys. Always use `src/reconstruct.py`, whose sequence is VLA with deferred injection and skipped components → strict AgiBot shard load → `action_head.inject_lora_after_loading()` → captured `load_lora_weight()`.

## Phase-3 launch and observability contract

- Do not bake a production GPU allocation into a decorator. `benchmark`, `tune`, and `train` pass a validated `H100:<n>`, `H200:<n>`, or `B200:<n>` spec to the dynamic GPU worker. The selected benchmark winner is a CLI/config value.
- The benchmark defaults to H100:8, H100:4, B200:8, and H200:8, and crosses those with dataloader-worker and container-CPU sweeps. It must read a fixed 10-repo train subset from container NVMe, reuse that copy for the worker sweep, and keep global batch 32.
- Arm files live in `configs/arms/*.json`: A is lr 1e-5, r/α 4/4; B is lr 5e-6, r/α 4/4; C is lr 1e-5, r/α 32/32. Every arm and main train default uses warmup ratio 0.01. Arms get equal wall-clock limits.
- All runs use the single WandB project `dreamzero-yam`. Required names carry `smoke`, `tune-armA/B/C`, `train`, `benchmark`, or `val-monitor`. Trainer loss must appear as `train/loss`.
- Tune screening metrics are `instability/nan_flag`, `instability/loss_spike_ratio`, `instability/rising_tail`, and `instability/grad_norm_ratio`.
- `src/val_monitor.py` is a 1-GPU heartbeat sidecar over complete milestone copies. It logs deterministic `val/loss` plus teacher-forced raw-absolute `val/open_loop_mse` (16 steps) and `val/open_loop_mse_24` at the checkpoint global step. Its open-loop path is `src/eval_dreamzero.py`: one raw single-frame observation and one independent 24-action causal call per frozen index, with runner JSON under `/checkpoints/yam/eval/`. Both standalone and monitored evaluation must reconstruct only through `src/reconstruct.py` and invert normalization with the checkpoint's merged metadata.
- `src/gpu_metrics.py` samples every GPU stage at about 10 seconds. Keep its DCGM → NVML GPM → NVML-util fallback and `dcgm/source_tier` tag. Benchmark tables include throughput, cost per 1,000 samples, analytic model TFLOPs/s/GPU, and MFU.
- `nvtop` is installed in the training image. For inspection: `modal container list`, then `modal container exec <container-id> -- nvtop`.
- `save_steps=0` (the train default) derives the cadence from the Young-Daly optimum `sqrt(2 * checkpoint-write-seconds * MTBF)`, snapped to a divisor of the 2,500-step main-run milestone interval so milestones always land on a saved checkpoint. Tune stages use `save_steps` itself as their milestone interval.
- Restarts are config-portable: train/tune assert `per_device_batch_size * num_gpus == 32`, and the launcher compares the latest checkpoint's DeepSpeed shard count against the current world size. A mismatch is a hard error unless `--allow-fresh-optimizer` is passed, which archives the old lineage, warm-starts the trained overlay via `DREAMZERO_WARM_START_CHECKPOINT`, and resets the optimizer and step counter.
- The val monitor mirrors every evaluated milestone to HF using the `hf-robocurve-write` secret's `HF_WRITE_TOKEN` (the read secret owns `HF_TOKEN`). Historical note: mirrors originally went to the release repo under `checkpoints/step-N/`; they now live weights-only in `robocurve/dreamzero-yam-molmoact2-checkpoints` (see `src/archive_upload.py`), and any future monitor run should target that archive repo.
- The training process reports dataloader wait per step and as a wall-clock fraction (`perf/dataloader_wait_*`); read these before buying more GPU throughput.

## Data and checkpoint gotchas

- Train split is exactly 124 repos from `configs/repos.json`; `allenai/19012026-block-13` is held out.
- Source vectors are 14-D: left joints `[0:6]`, left gripper `[6:7]`, right joints `[7:13]`, right gripper `[13:14]`.
- DreamZero modality subkeys are `left_joint_pos`, `left_gripper_pos`, `right_joint_pos`, `right_gripper_pos`.
- Video subkeys must be `top_camera-images-rgb`, `left_camera-images-rgb`, `right_camera-images-rgb`, in that order.
- Language is `annotation.task` mapped to numeric parquet `task_index`; the loader resolves it through seeded `tasks.jsonl`. Every dry run must assert non-empty strings.
- Seed only `info.json`, `tasks.jsonl`, and `episodes.jsonl`. Copying GR00T `modality.json` or `stats.json` makes the converter silently preserve incompatible metadata.
- Do not pass converter `--output-path` (it copies roughly 800 GB) or `--force` after task seeding (it replaces the seeded language metadata with empty defaults).
- Relative action applies to all 14 dims. Use `max_chunk_size=4` even though upstream YAML says 5.
- Degenerate `q99-q01` is fatal because inverse normalization does not mask it.
- A checkpoint is resumable only when trainer/config/metadata/model files and every DeepSpeed optimizer/model shard are complete. Prune partial `checkpoint-*` directories before upstream latest-checkpoint discovery.
- `save_lora_only=true` still requires DeepSpeed optimizer state for resume. The smoke callback proves bit-identical LoRA, head, and optimizer hashes at the boundary.
- Cross-stack evaluation is raw absolute 14-D joint space over the first 16 predicted steps. Relative DreamZero outputs must be decoded with the checkpoint's merged `experiment_cfg/metadata.json`, never a per-repo stats file.
- Generate frozen indices with `modal run src/eval_dreamzero.py::generate_eval_indices`, review and commit the printed payload as `outputs/eval_indices.json`, then run a checkpoint with `modal run src/eval_dreamzero.py::eval_checkpoint --checkpoint /checkpoints/yam/<experiment>/milestones/checkpoint-N`. Evaluation never auto-generates indices.
