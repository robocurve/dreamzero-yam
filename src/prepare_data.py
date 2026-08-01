"""Build DreamZero GEAR metadata beside the existing GR00T LeRobot-v2 tree.

This is a CPU-only Modal app. It never writes below ``/data/gr00t_v2`` and it
never copies video payloads: ``data/`` and ``videos/`` are hardlinked when the
volume supports it, with directory symlinks as the runtime fallback.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import modal

DREAMZERO_REF = "ab790c198fbce33503358efbbd4187ce9a89adf3"
APP_ROOT = Path("/root/dreamzero_yam")
LOCAL_ROOT = Path(__file__).resolve().parents[1]
DREAMZERO_ROOT = Path("/root/dreamzero")
REPOS_PATH = APP_ROOT / "configs/repos.json"
SOURCE_ROOT = Path("/data/gr00t_v2")
OUTPUT_ROOT = Path("/data/dreamzero_gear")
CONVERTER = DREAMZERO_ROOT / "scripts/data/convert_lerobot_to_gear.py"
READY_MARKER = ".dreamzero_gear_ready"

ACTION_HORIZON = 24
MAX_CHUNK_SIZE = 4
EXPECTED_TRAIN_EPISODES = 5_145
RANGE_EPS = 1e-8

KEY_RANGES = {
    "left_joint_pos": [0, 6],
    "left_gripper_pos": [6, 7],
    "right_joint_pos": [7, 13],
    "right_gripper_pos": [13, 14],
}
VIDEO_RENAMES = {
    "top": "top_camera-images-rgb",
    "left": "left_camera-images-rgb",
    "right": "right_camera-images-rgb",
}
REQUIRED_SEED_METADATA = ("info.json", "tasks.jsonl", "episodes.jsonl")


app = modal.App("dreamzero-yam-prepare")

prepare_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "git-lfs", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential"
    )
    .run_commands(
        f"git clone https://github.com/dreamzero0/dreamzero.git {DREAMZERO_ROOT}",
        f"cd {DREAMZERO_ROOT} && git checkout {DREAMZERO_REF}",
        "python -m pip install --index-url https://download.pytorch.org/whl/cpu "
        "torch==2.8.0 torchvision==0.23.0",
        "python -m pip install numpy==1.26.4 pandas pyarrow pydantic PyYAML tqdm "
        "decord2 av==15.0.0 hydra-core dm_tree scipy albumentations==1.4.18 "
        "einops transformers==4.51.3 peft==0.5.0 datasets==3.6.0 "
        "opencv-python-headless==4.9.0.80 h5py imageio matplotlib typing_extensions",
        f"python -m pip install --no-deps -e {DREAMZERO_ROOT}",
    )
    .add_local_dir(str(LOCAL_ROOT), str(APP_ROOT), copy=True)
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
)

data_volume = modal.Volume.from_name("gr00t-yam-data", create_if_missing=False)


def _load_repos(path: Path = REPOS_PATH) -> dict:
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "configs/repos.json"
    repos = json.loads(path.read_text())
    train = repos.get("train", [])
    assert len(train) == 124, f"expected 124 train repos, found {len(train)}"
    assert len(set(train)) == 124, "training repo list contains duplicates"
    assert repos["val"] not in train, "held-out repo leaked into the training split"
    return repos


def _repo_path(root: Path, repo_id: str) -> Path:
    owner, name = repo_id.split("/", 1)
    return root / owner / name


def _first_file(root: Path) -> Path:
    try:
        return next(path for path in root.rglob("*") if path.is_file())
    except StopIteration as exc:
        raise FileNotFoundError(f"no files found below {root}") from exc


def _hardlink_tree(source: Path, target: Path) -> None:
    for current, dirs, files in os.walk(source):
        current_path = Path(current)
        relative = current_path.relative_to(source)
        target_dir = target / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for directory in dirs:
            (target_dir / directory).mkdir(exist_ok=True)
        for filename in files:
            os.link(current_path / filename, target_dir / filename)


def _share_payload(source_repo: Path, output_repo: Path) -> str:
    """Share data/videos, falling back atomically to directory symlinks."""
    probe_source = _first_file(source_repo / "data")
    probe_target = output_repo / ".hardlink_probe"
    try:
        os.link(probe_source, probe_target)
        probe_target.unlink()
        _hardlink_tree(source_repo / "data", output_repo / "data")
        _hardlink_tree(source_repo / "videos", output_repo / "videos")
        mode = "hardlink"
    except OSError as exc:
        probe_target.unlink(missing_ok=True)
        for name in ("data", "videos"):
            target = output_repo / name
            if target.is_symlink():
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)
            target.symlink_to(source_repo / name, target_is_directory=True)
        mode = f"symlink ({errno.errorcode.get(exc.errno, type(exc).__name__)})"

    # A payload-sharing mode is not accepted until a real parquet and MP4 can
    # be read through the resulting DreamZero-side paths.
    with _first_file(output_repo / "data").open("rb") as parquet_file:
        parquet_file.read(64)
    with _first_file(output_repo / "videos").open("rb") as video_file:
        video_file.read(64)
    return mode


def _seed_metadata(source_repo: Path, output_repo: Path) -> None:
    meta = output_repo / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_SEED_METADATA:
        source = source_repo / "meta" / filename
        if not source.is_file():
            raise FileNotFoundError(f"required source metadata is absent: {source}")
        shutil.copy2(source, meta / filename)

    forbidden = ("modality.json", "stats.json", "relative_stats_dreamzero.json")
    present = [name for name in forbidden if (meta / name).exists()]
    assert not present, f"fresh metadata was contaminated by: {present}"


def _run_converter(output_repo: Path) -> None:
    mapping = json.dumps(KEY_RANGES, separators=(",", ":"))
    command = [
        sys.executable,
        str(CONVERTER),
        "--dataset-path",
        str(output_repo),
        "--embodiment-tag",
        "yam",
        "--state-keys",
        mapping,
        "--action-keys",
        mapping,
        "--relative-action-keys",
        *KEY_RANGES,
        "--action-horizon",
        str(ACTION_HORIZON),
    ]
    # No --output-path (it copytrees the payload), no --force (it would replace
    # the seeded tasks/episodes with empty language because task_index is not an
    # annotation-prefixed column).
    subprocess.run(command, cwd=DREAMZERO_ROOT, check=True)


def _post_edit_modality(output_repo: Path) -> None:
    path = output_repo / "meta/modality.json"
    modality = json.loads(path.read_text())
    source_video = modality.get("video", {})
    if set(source_video) != set(VIDEO_RENAMES):
        raise AssertionError(
            f"converter emitted unexpected video keys for {output_repo}: {sorted(source_video)}"
        )
    modality["video"] = {
        renamed: source_video[short] for short, renamed in VIDEO_RENAMES.items()
    }
    modality["annotation"] = {"task": {"original_key": "task_index"}}
    path.write_text(json.dumps(modality, indent=2) + "\n")


def _assert_relative_stats(output_repo: Path, eps: float = RANGE_EPS) -> None:
    path = output_repo / "meta/relative_stats_dreamzero.json"
    stats = json.loads(path.read_text())
    offenders: list[str] = []
    noncentral: list[str] = []
    for key in KEY_RANGES:
        if key not in stats:
            offenders.append(f"{key}:missing")
            continue
        q01 = stats[key]["q01"]
        q99 = stats[key]["q99"]
        mean = stats[key]["mean"]
        for dim, (lo, hi, center) in enumerate(zip(q01, q99, mean, strict=True)):
            width = float(hi) - float(lo)
            if not width > eps:
                offenders.append(f"{key}[{dim}] q01={lo} q99={hi}")
            scale = max(abs(float(lo)), abs(float(hi)), eps)
            if abs(float(center)) / scale > 0.5:
                noncentral.append(f"{key}[{dim}] mean={center} q01={lo} q99={hi}")
    if offenders:
        raise AssertionError(
            f"degenerate relative ranges in {output_repo}: " + "; ".join(offenders)
        )
    if noncentral:
        raise AssertionError(
            f"relative actions are not approximately zero-centered in {output_repo}: "
            + "; ".join(noncentral)
        )


def _assert_h264_decode(output_repo: Path) -> None:
    mp4 = next((output_repo / "videos").rglob("*.mp4"), None)
    if mp4 is None:
        raise FileNotFoundError(f"no MP4 found under {output_repo / 'videos'}")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nk=1:nw=1",
            str(mp4),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "h264", f"video is not H.264: {mp4}"
    from decord import VideoReader, cpu

    reader = VideoReader(str(mp4), ctx=cpu(0))
    assert len(reader) > 0, f"empty video: {mp4}"
    frame = reader[0].asnumpy()
    assert frame.ndim == 3 and frame.shape[-1] == 3, (
        f"bad decoded frame {frame.shape}: {mp4}"
    )


def _loader_dry_run(output_repo: Path, samples: int, seed: int) -> None:
    """Exercise DreamZero's real sharded loader and its exact YAM key names."""
    import numpy as np
    from groot.vla.data.dataset import ModalityConfig
    from groot.vla.data.dataset.lerobot_sharded import (
        ShardedLeRobotSubLangSingleActionChunkDatasetDROID,
    )

    video_keys = [f"video.{name}" for name in VIDEO_RENAMES.values()]
    state_keys = [f"state.{name}" for name in KEY_RANGES]
    action_keys = [f"action.{name}" for name in KEY_RANGES]
    configs = {
        "video": ModalityConfig(
            delta_indices=list(range(25)), modality_keys=video_keys
        ),
        "state": ModalityConfig(delta_indices=[0], modality_keys=state_keys),
        "action": ModalityConfig(
            delta_indices=list(range(ACTION_HORIZON)), modality_keys=action_keys
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.task"]
        ),
    }
    dataset = ShardedLeRobotSubLangSingleActionChunkDatasetDROID(
        dataset_path=output_repo,
        embodiment_tag="yam",
        modality_configs=configs,
        use_global_metadata=False,
        video_backend="decord",
        max_chunk_size=MAX_CHUNK_SIZE,
        relative_action=True,
        relative_action_keys=list(KEY_RANGES),
    )

    # The sharded dataset only serves trajectories from a cached shard —
    # get_step_data without start/finish_cache_shard hits the "Cached
    # dataframe is None" assert. Walk shards until enough long-enough
    # trajectories are available.
    length_by_id = {
        int(tid): int(length)
        for tid, length in zip(
            dataset.trajectory_ids, dataset.trajectory_lengths, strict=True
        )
    }
    minimum_length = ACTION_HORIZON * MAX_CHUNK_SIZE + 2
    candidates: list[tuple[int, int]] = []
    for shard_index in range(dataset.num_shards):
        dataset.start_cache_shard(shard_index)
        dataset.finish_cache_shard()
        candidates = [
            (tid, length_by_id[tid] // 2)
            for tid in dataset.get_trajectories_in_shard()
            if length_by_id[tid] >= minimum_length
        ]
        if candidates:
            break
    if not candidates:
        raise AssertionError(
            f"no episode long enough for a {MAX_CHUNK_SIZE}-chunk sample: {output_repo}"
        )

    rng = random.Random(seed)
    for trajectory_id, base_index in rng.sample(
        candidates, k=min(samples, len(candidates))
    ):
        indices = {
            key: np.asarray(delta, dtype=np.int64) + base_index
            for key, delta in dataset.delta_indices.items()
        }
        sample = dataset.get_step_data(trajectory_id, indices)
        expected = set(video_keys + state_keys + action_keys + ["annotation.task"])
        assert set(sample) == expected, (
            f"loader key mismatch for {output_repo}: missing={expected - set(sample)} "
            f"extra={set(sample) - expected}"
        )
        action_lengths = {
            key: int(np.asarray(sample[key]).shape[0]) for key in action_keys
        }
        expected_length = ACTION_HORIZON * MAX_CHUNK_SIZE
        assert set(action_lengths.values()) == {expected_length}, (
            f"action length != horizon x chunk size ({expected_length}) in {output_repo}: "
            f"{action_lengths}"
        )
        prompts = sample["annotation.task"]
        assert prompts and all(isinstance(p, str) and p.strip() for p in prompts), (
            f"empty/non-string annotation.task in {output_repo}: {prompts!r}"
        )


def validate_repo(output_repo: Path, samples: int = 2, seed: int = 0) -> int:
    for filename in (
        "info.json",
        "tasks.jsonl",
        "episodes.jsonl",
        "modality.json",
        "stats.json",
        "relative_stats_dreamzero.json",
        "embodiment.json",
    ):
        path = output_repo / "meta" / filename
        if not path.is_file():
            raise FileNotFoundError(f"incomplete GEAR metadata: {path}")
    info = json.loads((output_repo / "meta/info.json").read_text())
    episodes = int(info["total_episodes"])
    assert episodes > 0, f"zero episodes in {output_repo}"
    _assert_relative_stats(output_repo)
    _assert_h264_decode(output_repo)
    _loader_dry_run(output_repo, samples=samples, seed=seed)
    return episodes


def prepare_all(samples_per_repo: int = 2) -> None:
    repos = _load_repos()
    all_repos = [*repos["train"], repos["val"]]
    train_episodes = 0
    sharing_modes: dict[str, str] = {}

    for index, repo_id in enumerate(all_repos, 1):
        source_repo = _repo_path(SOURCE_ROOT, repo_id)
        output_repo = _repo_path(OUTPUT_ROOT, repo_id)
        if not source_repo.is_dir():
            raise FileNotFoundError(f"source v2 repo is absent: {source_repo}")
        print(f"[{index}/{len(all_repos)}] {repo_id}", flush=True)

        if not (output_repo / READY_MARKER).is_file():
            if output_repo.exists() or output_repo.is_symlink():
                if output_repo.is_symlink():
                    output_repo.unlink()
                else:
                    shutil.rmtree(output_repo)
            output_repo.mkdir(parents=True)
            sharing_modes[repo_id] = _share_payload(source_repo, output_repo)
            _seed_metadata(source_repo, output_repo)
            _run_converter(output_repo)
            _post_edit_modality(output_repo)
        else:
            sharing_modes[repo_id] = "existing"

        episodes = validate_repo(output_repo, samples=samples_per_repo, seed=index)
        if repo_id in repos["train"]:
            train_episodes += episodes
        (output_repo / READY_MARKER).write_text("validated\n")

    assert train_episodes == EXPECTED_TRAIN_EPISODES, (
        f"training episode count mismatch: expected {EXPECTED_TRAIN_EPISODES}, got {train_episodes}"
    )
    manifest = {
        "train": [str(_repo_path(OUTPUT_ROOT, repo)) for repo in repos["train"]],
        "val": str(_repo_path(OUTPUT_ROOT, repos["val"])),
        "train_episodes": train_episodes,
        "sharing_modes": sharing_modes,
    }
    (OUTPUT_ROOT / "datasets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"validated 124 train repos / {train_episodes} episodes plus held-out val",
        flush=True,
    )


@app.function(
    image=prepare_image,
    volumes={"/data": data_volume},
    timeout=24 * 60 * 60,
    cpu=16,
    memory=131_072,
)
def prepare_data(samples_per_repo: int = 2) -> None:
    prepare_all(samples_per_repo=samples_per_repo)
    data_volume.commit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-repo", type=int, default=2)
    args = parser.parse_args()
    prepare_all(samples_per_repo=args.samples_per_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
