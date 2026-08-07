"""Archive the training run to an accompanying HF repo, then verify.

Uploads the 26 milestone checkpoints WEIGHTS-ONLY (DeepSpeed optimizer
shards excluded — each milestone dir is ~95 GB of which ~94 GB is resume
state nobody will ever resume from; the launcher can warm-start any
milestone's weights with --allow-fresh-optimizer), the final full training
state (checkpoint-12000 including shards, resumable), the run logs/configs,
and the imagined-vs-real renders to ARCHIVE_REPO. The merged release itself
lives in RELEASE_REPO and is not duplicated here.

Run `survey` first to size the tree, then `upload_archive --confirm`.
The upload verifies every local file exists remotely with matching size, so
volume cleanup can be gated on a verified mirror. `finalize_release` then
re-syncs MODEL_CARD.md to the release repo and deletes its stale 1.9 TB
`checkpoints/` mirror folder (weights preserved in ARCHIVE_REPO).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from src.modal_app import checkpoint_volume, secrets, training_image

app = modal.App("dreamzero-yam-archive")
ARCHIVE_REPO = "robocurve/dreamzero-yam-molmoact2-checkpoints"
RELEASE_REPO = "robocurve/dreamzero-yam-molmoact2"
RUN_ROOT = Path("/checkpoints/yam/tune-armC")
RENDERS = Path("/checkpoints/yam/renders")
RUN_TOP_FILES = [
    "trainer_state.json",
    "config.json",
    "runtime_metrics.jsonl",
    "loss_log.jsonl",
    "wandb_config.json",
]

ARCHIVE_README = """---
license: other
license_name: nvidia-open-model-license
base_model: GEAR-Dreams/DreamZero-AgiBot
tags:
  - robotics
  - world-action-model
  - dreamzero
  - yam
---

# dreamzero-yam-molmoact2-checkpoints

Training-run archive for
[robocurve/dreamzero-yam-molmoact2](https://huggingface.co/robocurve/dreamzero-yam-molmoact2)
(the merged release; start there). Training code, eval protocol, and model
card: [robocurve/dreamzero-yam](https://github.com/robocurve/dreamzero-yam).

Contents:

- `checkpoints/step-N/` — all 26 raw LoRA training milestones, weights only
  (steps 100, 200, 300, then every 500 from 1,000 to 12,000; DeepSpeed
  optimizer shards are omitted). **Do not load these with stock
  `load_lora`** — it rebuilds the frozen DiT on vanilla Wan2.1 instead of
  the DreamZero-AgiBot base they were trained against. Use the explicit
  reconstruction in
  [`src/reconstruct.py`](https://github.com/robocurve/dreamzero-yam/blob/main/src/reconstruct.py).
- `training-state/checkpoint-12000/` — the final checkpoint with full
  DeepSpeed optimizer state, resumable for continued training (the val/loss
  curve had plateaued but not risen at 12k steps). Earlier milestones can be
  warm-started weights-only via the training repo's
  `--allow-fresh-optimizer` path.
- `training-state/` top level — trainer state, run config, per-step runtime
  metrics, and loss logs of the main run.
- `renders/` — imagined-vs-real rollout videos from the released checkpoint
  (model's 2x2 camera-grid dream next to ground-truth frames).
"""


def _dir_stats(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


@app.function(
    image=training_image,
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[*secrets, modal.Secret.from_name("hf-robocurve-write")],
    timeout=60 * 60,
    cpu=8,
    memory=32_768,
)
def survey() -> None:
    for root in [RUN_ROOT, RENDERS, Path("/checkpoints/yam")]:
        print(f"== {root}")
        for child in sorted(root.iterdir()):
            if child.is_dir():
                count, size = _dir_stats(child)
                print(f"  {child.name:32s} {count:5d} files {size / 1e9:9.2f} GB")
            else:
                print(f"  {child.name:32s}       file {child.stat().st_size / 1e9:9.2f} GB")


@app.function(
    image=training_image,
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[*secrets, modal.Secret.from_name("hf-robocurve-write")],
    timeout=12 * 60 * 60,
    cpu=8,
    memory=32_768,
)
def upload_archive(confirm: bool = False) -> None:
    os.environ["HF_HOME"] = "/tmp/hf-cache"
    from huggingface_hub import HfApi

    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        raise RuntimeError("HF_WRITE_TOKEN missing")
    api = HfApi(token=token)

    # (dest, src, weights_only) — weights_only skips DeepSpeed shard dirs.
    sections: list[tuple[str, Path, bool]] = []
    for milestone in sorted(
        (RUN_ROOT / "milestones").iterdir(), key=lambda p: int(p.name.split("-")[1])
    ):
        step = int(milestone.name.split("-")[1])
        sections.append((f"checkpoints/step-{step}", milestone, True))
    sections.append(
        ("training-state/checkpoint-12000", RUN_ROOT / "checkpoint-12000", False)
    )
    sections.append(
        ("training-state/experiment_cfg", RUN_ROOT / "experiment_cfg", False)
    )
    sections.append(("renders", RENDERS, False))

    def _section_files(src: Path, weights_only: bool) -> list[Path]:
        return [
            p
            for p in src.rglob("*")
            if p.is_file()
            and not (
                weights_only
                and any(part.startswith("global_step") for part in p.parts)
            )
        ]

    manifest = {}
    for dest, src, weights_only in sections:
        files = _section_files(src, weights_only)
        manifest[dest] = {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}
    for name in RUN_TOP_FILES:
        path = RUN_ROOT / name
        manifest[f"training-state/{name}"] = {"files": 1, "bytes": path.stat().st_size}
    total = sum(entry["bytes"] for entry in manifest.values())
    print(json.dumps(manifest, indent=2))
    print(f"total {total / 1e9:.2f} GB across {len(manifest)} sections", flush=True)
    if not confirm:
        print("dry run complete; re-run with --confirm to upload")
        return

    api.create_repo(ARCHIVE_REPO, repo_type="model", private=False, exist_ok=True)
    for dest, src, weights_only in sections:
        api.upload_folder(
            repo_id=ARCHIVE_REPO,
            repo_type="model",
            folder_path=str(src),
            path_in_repo=dest,
            ignore_patterns=["global_step*/**"] if weights_only else None,
            commit_message=f"Archive {dest}",
        )
        print(f"uploaded {dest}", flush=True)
    for name in RUN_TOP_FILES:
        api.upload_file(
            repo_id=ARCHIVE_REPO,
            repo_type="model",
            path_or_fileobj=str(RUN_ROOT / name),
            path_in_repo=f"training-state/{name}",
            commit_message=f"Archive training-state/{name}",
        )
    api.upload_file(
        repo_id=ARCHIVE_REPO,
        repo_type="model",
        path_or_fileobj=ARCHIVE_README.encode(),
        path_in_repo="README.md",
        commit_message="Archive README",
    )

    # Verify: every local file must exist remotely with matching size.
    remote = {
        entry.path: entry.size
        for entry in api.list_repo_tree(ARCHIVE_REPO, recursive=True)
        if hasattr(entry, "size") and entry.size is not None
    }
    missing, mismatched = [], []
    for dest, src, weights_only in sections:
        for path in _section_files(src, weights_only):
            key = f"{dest}/{path.relative_to(src)}"
            if key not in remote:
                missing.append(key)
            elif remote[key] != path.stat().st_size:
                mismatched.append(key)
    for name in RUN_TOP_FILES:
        key = f"training-state/{name}"
        if key not in remote:
            missing.append(key)
        elif remote[key] != (RUN_ROOT / name).stat().st_size:
            mismatched.append(key)
    if missing or mismatched:
        raise AssertionError(
            f"verification FAILED: {len(missing)} missing, "
            f"{len(mismatched)} size-mismatched; first: {(missing + mismatched)[:5]}"
        )
    print(
        f"verification PASS: {len(remote)} remote files cover all local sections",
        flush=True,
    )


@app.function(
    image=training_image,
    volumes={
        "/checkpoints": checkpoint_volume,
        "/pi05": modal.Volume.from_name("pi05-yam-ckpt"),
        "/gr00t": modal.Volume.from_name("gr00t-yam-ckpt"),
    },
    secrets=[*secrets, modal.Secret.from_name("hf-robocurve-write")],
    timeout=60 * 60,
    cpu=4,
    memory=16_384,
)
def upload_eval_jsons() -> None:
    """Ship the frozen-protocol interchange JSONs the model card promises.

    eval/ in the release repo gets all 24 DreamZero milestone runner files
    plus the pi0.5 and GR00T baseline re-scores (20 samples each, same
    frozen indices, machine-verified GT identity).
    """
    os.environ["HF_HOME"] = "/tmp/hf-cache"
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
    uploads = [
        (path, f"eval/{path.name}")
        for path in sorted(Path("/checkpoints/yam/eval").glob("*.json"))
    ]
    uploads.append((Path("/pi05/eval_frozen/pi05.json"), "eval/pi05.json"))
    uploads.append((Path("/gr00t/eval_frozen/gr00t.json"), "eval/gr00t.json"))
    ops = []
    from huggingface_hub import CommitOperationAdd

    for src, dest in uploads:
        ops.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src)))
    api.create_commit(
        repo_id=RELEASE_REPO,
        repo_type="model",
        operations=ops,
        commit_message="Ship frozen-protocol per-sample interchange JSONs (eval/)",
    )
    print(f"uploaded {len(ops)} eval JSONs to {RELEASE_REPO}/eval", flush=True)


@app.function(
    image=training_image,
    secrets=[*secrets, modal.Secret.from_name("hf-robocurve-write")],
    timeout=60 * 60,
    cpu=4,
    memory=16_384,
)
def finalize_release(confirm: bool = False) -> None:
    """Re-sync the model card and drop the stale 1.9 TB mirror folder.

    Run only after upload_archive prints verification PASS: the release
    repo's checkpoints/ folder holds partial full-state milestone mirrors
    whose weight files are superseded by ARCHIVE_REPO.
    """
    os.environ["HF_HOME"] = "/tmp/hf-cache"
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
    stale = [
        entry.path
        for entry in api.list_repo_tree(RELEASE_REPO, recursive=False)
        if entry.path == "checkpoints"
    ]
    print(f"release repo stale mirror folder present: {bool(stale)}")
    if not confirm:
        print("dry run complete; re-run with --confirm to finalize")
        return
    card = Path("/root/dreamzero_yam/MODEL_CARD.md")
    api.upload_file(
        repo_id=RELEASE_REPO,
        repo_type="model",
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        commit_message="Point milestone checkpoints at the archive repo",
    )
    if stale:
        api.delete_folder(
            repo_id=RELEASE_REPO,
            repo_type="model",
            path_in_repo="checkpoints",
            commit_message="Move milestone mirrors to dreamzero-yam-molmoact2-checkpoints (weights-only)",
        )
        print("deleted release repo checkpoints/ folder", flush=True)
