"""Faithful one-frame, one-action-chunk DreamZero-YAM evaluation."""

from __future__ import annotations

import copy
import glob
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import modal

from src.modal_app import checkpoint_volume, data_volume, secrets, training_image

app = modal.App("dreamzero-yam-open-loop-eval")
LOCAL_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = Path("/root/dreamzero_yam")
DATA_VAL = Path("/data/dreamzero_gear/allenai/19012026-block-13")
LOCAL_VAL = Path("/local/dreamzero_gear/allenai/19012026-block-13")
CHECKPOINT_ROOT = Path("/checkpoints/yam")
DEFAULT_INDICES_PATH = PIPELINE_ROOT / "outputs/eval_indices.json"
CHUNK_HORIZON = 24

VIDEO_CAMERAS = (
    ("video.top_camera-images-rgb", "observation.images.top"),
    ("video.left_camera-images-rgb", "observation.images.left"),
    ("video.right_camera-images-rgb", "observation.images.right"),
)
STATE_SLICES = (
    ("state.left_joint_pos", 0, 6),
    ("state.left_gripper_pos", 6, 7),
    ("state.right_joint_pos", 7, 13),
    ("state.right_gripper_pos", 13, 14),
)
ACTION_KEYS = (
    "left_joint_pos",
    "left_gripper_pos",
    "right_joint_pos",
    "right_gripper_pos",
)
ACTION_DIMS = (6, 1, 6, 1)

indices_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26,<3")
    .add_local_dir(str(LOCAL_ROOT), str(PIPELINE_ROOT), copy=True)
    .env({"PYTHONPATH": str(PIPELINE_ROOT), "PYTHONUNBUFFERED": "1"})
)


def _episode_number(path: str | Path) -> int:
    match = re.fullmatch(r"episode_(\d+)\.(?:parquet|mp4)", Path(path).name)
    if match is None:
        raise ValueError(f"cannot parse episode number from {path}")
    return int(match.group(1))


def _index_episode_files(paths: list[str], kind: str) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path_text in paths:
        path = Path(path_text)
        episode = _episode_number(path)
        if episode in indexed:
            raise ValueError(
                f"duplicate {kind} for episode {episode}: {indexed[episode]}, {path}"
            )
        indexed[episode] = path
    return indexed


class _YAMDataset:
    """Read episode-addressed YAM parquet rows and single RGB video frames."""

    def __init__(self, root: Path):
        if not root.is_dir():
            raise FileNotFoundError(f"held-out DreamZero repo is absent: {root}")
        self.root = root
        parquet_paths = sorted(
            glob.glob(str(root / "data/**/episode_*.parquet"), recursive=True)
        )
        if not parquet_paths:
            raise FileNotFoundError(f"no episode parquet files found under {root / 'data'}")
        self.parquet_paths = _index_episode_files(parquet_paths, "parquet")
        self._tables: dict[int, Any] = {}

        # These folder names are meta/modality.json's original_key values for
        # the three DreamZero-facing video keys above.
        self.video_paths: dict[str, dict[int, Path]] = {}
        for video_key, folder_name in VIDEO_CAMERAS:
            paths = sorted(
                glob.glob(
                    str(root / f"videos/**/{folder_name}/episode_*.mp4"),
                    recursive=True,
                )
            )
            if not paths:
                raise FileNotFoundError(
                    f"no {folder_name} episode videos found under {root / 'videos'}"
                )
            self.video_paths[video_key] = _index_episode_files(paths, folder_name)

        tasks_path = root / "meta/tasks.jsonl"
        if not tasks_path.is_file():
            raise FileNotFoundError(f"held-out tasks metadata is absent: {tasks_path}")
        self.tasks: dict[int, str] = {}
        for line_number, line in enumerate(tasks_path.read_text().splitlines()):
            if not line:
                continue
            row = json.loads(line)
            task_index = int(row.get("task_index", line_number))
            task = row.get("task")
            if task_index in self.tasks:
                raise ValueError(f"duplicate task_index {task_index} in {tasks_path}")
            if not isinstance(task, str) or not task.strip():
                raise AssertionError(
                    f"empty/non-string task for task_index {task_index}: {task!r}"
                )
            self.tasks[task_index] = task

    def _table(self, episode: int) -> Any:
        if episode not in self.parquet_paths:
            raise KeyError(f"episode {episode} has no parquet file under {self.root / 'data'}")
        if episode not in self._tables:
            import pyarrow.parquet as pq

            self._tables[episode] = pq.read_table(
                self.parquet_paths[episode],
                columns=["action", "observation.state", "task_index", "frame_index"],
            )
        return self._tables[episode]

    def _read_frame(self, episode: int, frame: int, video_key: str) -> Any:
        import cv2

        try:
            path = self.video_paths[video_key][episode]
        except KeyError as exc:
            raise KeyError(
                f"episode {episode} has no video for {video_key!r}"
            ) from exc
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(f"failed to read frame {frame} from {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def sample(self, episode: int, frame: int) -> tuple[dict[str, Any], Any]:
        """Build upstream-style one-frame obs plus a raw 24-step GT chunk."""
        import numpy as np

        table = self._table(episode)
        if frame < 0 or frame + CHUNK_HORIZON > table.num_rows:
            raise IndexError(
                f"episode {episode} frame {frame} lacks {CHUNK_HORIZON} rows; "
                f"episode length is {table.num_rows}"
            )
        actual_frame = int(table.column("frame_index")[frame].as_py())
        if actual_frame != frame:
            raise AssertionError(
                f"episode {episode} row {frame} declares frame_index={actual_frame}"
            )

        state = np.asarray(
            table.column("observation.state")[frame].as_py(), dtype=np.float64
        )
        if state.shape != (14,):
            raise AssertionError(
                f"episode {episode} frame {frame} state is not 14-D: {state.shape}"
            )
        task_index = int(table.column("task_index")[frame].as_py())
        try:
            prompt = self.tasks[task_index]
        except KeyError as exc:
            raise KeyError(f"task_index {task_index} is absent from tasks.jsonl") from exc
        if not isinstance(prompt, str) or not prompt.strip():
            raise AssertionError(
                f"empty/non-string prompt at episode {episode} frame {frame}: {prompt!r}"
            )

        obs: dict[str, Any] = {}
        for video_key, _ in VIDEO_CAMERAS:
            rgb = self._read_frame(episode, frame, video_key)
            obs[video_key] = rgb[np.newaxis, ...].astype(np.uint8)
        for state_key, start, end in STATE_SLICES:
            obs[state_key] = state[start:end].reshape(1, -1).astype(np.float64)
        obs["annotation.task"] = prompt

        ground_truth = np.asarray(
            table.column("action")
            .slice(frame, CHUNK_HORIZON)
            .to_pylist(),
            dtype=np.float64,
        )
        if ground_truth.shape != (CHUNK_HORIZON, 14):
            raise AssertionError(
                f"episode {episode} frame {frame} GT has shape {ground_truth.shape}, "
                f"expected ({CHUNK_HORIZON}, 14)"
            )
        return obs, ground_truth


def _build_eval_transform(model: Any, checkpoint: Path) -> tuple[Any, Any]:
    """Mirror GrootSimPolicy's checkpoint metadata and eval-transform setup."""
    import numpy as np
    from groot.vla.data.schema import DatasetMetadata
    from groot.vla.data.transform import ComposedModalityTransform
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    metadata_payload = json.loads(
        (checkpoint / "experiment_cfg/metadata.json").read_text()
    )
    if "yam" not in metadata_payload:
        raise KeyError("checkpoint merged metadata has no 'yam' embodiment")
    metadata = DatasetMetadata.model_validate(metadata_payload["yam"])

    if hasattr(model, "action_head") and hasattr(model.action_head, "config"):
        action_head_cfg = model.action_head.config
        target_height = getattr(action_head_cfg, "target_video_height", None)
        target_width = getattr(action_head_cfg, "target_video_width", None)
        if (
            target_height is not None
            and target_width is not None
            and metadata.modalities.video
        ):
            for key in metadata.modalities.video:
                metadata.modalities.video[key].resolution = (
                    int(target_width),
                    int(target_height),
                )

    train_cfg = OmegaConf.load(checkpoint / "experiment_cfg/conf.yaml")
    if "yam" not in train_cfg.transforms:
        raise KeyError(f"checkpoint transforms have no 'yam': {train_cfg.transforms.keys()}")
    eval_transform = instantiate(train_cfg.transforms["yam"])
    if not isinstance(eval_transform, ComposedModalityTransform):
        raise AssertionError(f"expected ComposedModalityTransform, got {eval_transform!r}")
    eval_transform.set_metadata(metadata)

    if train_cfg.get("relative_action_per_horizon", False):
        per_horizon_stats = {}
        for action_key in metadata.statistics.action:
            stats = metadata.statistics.action[action_key].model_dump()
            q01 = stats.get("q01")
            is_2d = (
                isinstance(q01, list | np.ndarray)
                and len(q01) > 0
                and isinstance(q01[0], list | np.ndarray)
            )
            if is_2d:
                per_horizon_stats[action_key] = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in stats.items()
                }
        if per_horizon_stats:
            eval_transform.set_per_horizon_statistics(per_horizon_stats)
        else:
            print(
                "WARNING: relative_action_per_horizon is true but metadata has no "
                "per-horizon action statistics",
                flush=True,
            )

    eval_transform.eval()
    return train_cfg, eval_transform


def _state_is_batched(obs: dict[str, Any]) -> bool:
    for key, value in obs.items():
        if "state" in key and len(value.shape) < 3:
            return False
    return True


def _unsqueeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    unsqueezed = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            unsqueezed[key] = np.expand_dims(value, axis=0)
        elif isinstance(value, list):
            unsqueezed[key] = np.array(value)
        elif isinstance(value, torch.Tensor):
            unsqueezed[key] = value.unsqueeze(0)
        elif isinstance(value, str):
            unsqueezed[key] = np.array([value])
        else:
            unsqueezed[key] = value
    return unsqueezed


def _squeeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    squeezed = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            squeezed[key] = np.squeeze(value)
        elif isinstance(value, torch.Tensor):
            squeezed[key] = value.squeeze()
        else:
            squeezed[key] = value
    return squeezed


def _as_plain_dict(value: Any) -> dict[str, Any]:
    value_type = type(value)
    if value_type.__module__.startswith("tianshou."):
        value = value.__getstate__()
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"eval transform returned unsupported type {type(value)}")


def _decode_absolute_action(
    eval_transform: Any,
    train_cfg: Any,
    normalized_action: Any,
    original_obs: dict[str, Any],
) -> Any:
    """Invert eval transforms and restore raw absolute actions from raw state."""
    import numpy as np
    import torch

    decoded = eval_transform.unapply(dict(action=normalized_action.cpu()))
    relative_action = train_cfg.get("relative_action", False)
    relative_per_horizon = train_cfg.get("relative_action_per_horizon", False)
    relative_keys = train_cfg.get("relative_action_keys", [])
    if (relative_action or relative_per_horizon) and relative_keys:
        for key in relative_keys:
            action_key = f"action.{key}"
            state_key = f"state.{key}"
            if action_key not in decoded:
                continue
            if state_key not in original_obs:
                raise KeyError(
                    f"original observation lacks {state_key!r} for relative decode"
                )
            last_state = original_obs[state_key]
            if torch.is_tensor(last_state):
                last_state = last_state.cpu().numpy()
            if len(last_state.shape) >= 2:
                last_state = last_state[..., -1, :]
            action_value = decoded[action_key]
            if torch.is_tensor(action_value):
                action_value = action_value.cpu().numpy()
            if len(action_value.shape) > len(last_state.shape):
                last_state = np.expand_dims(last_state, axis=-2)
            decoded[action_key] = action_value + last_state

    decoded = _squeeze_dict_values(decoded)
    action_parts = []
    for key, width in zip(ACTION_KEYS, ACTION_DIMS, strict=True):
        action_key = f"action.{key}"
        if action_key not in decoded:
            raise KeyError(f"inverse transform omitted {action_key!r}")
        part = np.asarray(decoded[action_key], dtype=np.float64)
        # Upstream squeeze_dict_values collapses the singleton gripper width;
        # restore it before concatenating the four YAM keys.
        if part.ndim == 1 and width == 1 and part.shape[0] == CHUNK_HORIZON:
            part = part.reshape(CHUNK_HORIZON, 1)
        if part.shape != (CHUNK_HORIZON, width):
            raise AssertionError(
                f"decoded {action_key} has shape {part.shape}, expected "
                f"({CHUNK_HORIZON}, {width})"
            )
        action_parts.append(part)
    absolute = np.concatenate(action_parts, axis=-1, dtype=np.float64)
    if absolute.shape != (CHUNK_HORIZON, 14):
        raise AssertionError(f"decoded YAM action has unexpected shape {absolute.shape}")
    return absolute


def _predict_chunk(
    model: Any,
    train_cfg: Any,
    eval_transform: Any,
    obs: dict[str, Any],
) -> Any:
    """Mirror GrootSimPolicy.lazy_joint_forward_causal without its loader."""
    import torch

    original_obs = copy.deepcopy(obs)
    is_batched = _state_is_batched(obs)
    if not is_batched:
        obs = _unsqueeze_dict_values(obs)
        original_obs = _unsqueeze_dict_values(original_obs)

    normalized_input = _as_plain_dict(eval_transform(obs))
    if train_cfg.get("eval_bf16", False):
        for key, value in normalized_input.items():
            if torch.is_tensor(value) and value.dtype == torch.float32:
                normalized_input[key] = value.to(dtype=torch.bfloat16)

    # The head caches language and a causal frame cursor. A single video frame
    # also resets the cursor upstream, but forcing both values makes independence
    # explicit and prevents any prior caller state from affecting this sample.
    model.action_head.language = None
    model.action_head.current_start_frame = 0
    with torch.inference_mode():
        output = model.lazy_joint_video_action_causal(
            normalized_input, latent_video=None
        )
    normalized_action = output["action_pred"].float()
    absolute = _decode_absolute_action(
        eval_transform, train_cfg, normalized_action, original_obs
    )
    if not is_batched:
        return absolute
    raise AssertionError("one-frame DreamZero eval observations must be unbatched")


def run_open_loop_eval(
    model: Any,
    checkpoint: Path,
    dataset_root: Path,
    indices_path: str | Path,
    output_path: str | Path,
) -> dict[str, float]:
    """Predict one independent 24-action chunk for every frozen eval index."""
    from src.eval_spec import load_indices, raw_absolute_mse, write_runner_output

    checkpoint = Path(checkpoint)
    dataset = _YAMDataset(Path(dataset_root))
    indices = load_indices(indices_path)
    train_cfg, eval_transform = _build_eval_transform(model, checkpoint)
    predictions = {}
    ground_truth = {}
    mse_16 = []
    mse_24 = []
    model.eval()

    for sample_number, (episode, frame) in enumerate(indices, start=1):
        obs, target = dataset.sample(episode, frame)
        prediction = _predict_chunk(model, train_cfg, eval_transform, obs)
        index = (episode, frame)
        predictions[index] = prediction
        ground_truth[index] = target
        mse_16.append(raw_absolute_mse(prediction, target, horizon=16))
        mse_24.append(raw_absolute_mse(prediction, target, horizon=24))
        print(
            f"DreamZero eval [{sample_number}/{len(indices)}] "
            f"episode={episode} frame={frame}",
            flush=True,
        )

    if not mse_16:
        raise AssertionError("frozen eval index set produced no predictions")
    write_runner_output(
        runner="dreamzero",
        ground_truth=ground_truth,
        predictions=predictions,
        output_path=output_path,
        indices_path=indices_path,
    )
    return {
        "open_loop_mse": sum(mse_16) / len(mse_16),
        "open_loop_mse_24": sum(mse_24) / len(mse_24),
    }


def default_runner_output_path(checkpoint: str | Path) -> Path:
    """Return the durable canonical runner-output path for a checkpoint."""
    checkpoint = Path(checkpoint)
    experiment = (
        checkpoint.parent.parent.name
        if checkpoint.parent.name == "milestones"
        else checkpoint.parent.name
    )
    step = checkpoint.name.rsplit("-", 1)[-1]
    return CHECKPOINT_ROOT / "eval" / f"{experiment}-step-{step}-dreamzero.json"


def _copy_val_to_nvme() -> None:
    if not DATA_VAL.is_dir():
        raise FileNotFoundError(f"held-out DreamZero repo is absent: {DATA_VAL}")
    LOCAL_VAL.parent.mkdir(parents=True, exist_ok=True)
    if not LOCAL_VAL.exists():
        subprocess.run(["cp", "-aL", str(DATA_VAL), str(LOCAL_VAL.parent)], check=True)


@app.function(
    image=training_image,
    gpu="H200:1",
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    secrets=secrets,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=60.0),
    cpu=16,
    memory=196_608,
    ephemeral_disk=1_200_000,
)
def eval_checkpoint(
    checkpoint: str,
    indices_path: str = str(DEFAULT_INDICES_PATH),
    output_path: str = "",
) -> None:
    """Reconstruct one checkpoint and run the frozen open-loop evaluator."""
    # Upstream torch.compiles the UniPC scheduler update with one_graph=True;
    # per-sample scheduler instances recompile until FailOnRecompileLimitHit.
    # Eager math is identical for evaluation, so disable dynamo entirely in
    # this process (must precede the first torch import).
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import torch

    from src.reconstruct import reconstruct_model

    _copy_val_to_nvme()
    checkpoint_path = Path(checkpoint)
    destination = (
        Path(output_path) if output_path else default_runner_output_path(checkpoint_path)
    )
    model, _, _ = reconstruct_model(checkpoint_path, device="cuda")
    # Keep reconstruct.py's injected LoRA modules in place. GrootSimPolicy's
    # merge_and_unload is mathematically equivalent, but its loader selects the
    # wrong frozen DiT base for this project before reaching that step.
    model.to(dtype=torch.bfloat16)
    model.post_initialize()
    metrics = run_open_loop_eval(
        model,
        checkpoint_path,
        LOCAL_VAL,
        indices_path=indices_path,
        output_path=destination,
    )
    checkpoint_volume.commit()
    print(json.dumps(metrics, sort_keys=True), flush=True)


@app.function(
    image=indices_image,
    volumes={"/data": data_volume},
    cpu=2,
    memory=4_096,
    timeout=60 * 60,
)
def generate_eval_indices(dataset_root: str = str(DATA_VAL)) -> None:
    """Generate the frozen-index payload on CPU for review and source control."""
    from src.eval_spec import generate_indices

    payload = generate_indices(dataset_root, "/tmp/eval_indices.json")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


__all__ = [
    "app",
    "default_runner_output_path",
    "eval_checkpoint",
    "generate_eval_indices",
    "run_open_loop_eval",
]
