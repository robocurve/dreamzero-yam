"""Frozen cross-stack YAM evaluation indices and raw-absolute MSE contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

VAL_REPO = "allenai/19012026-block-13"
N_SAMPLES = 20
SEED = 0
MIN_REMAINING_FRAMES = 24
PRIMARY_HORIZON = 16
ACTION_DIM = 14
DEFAULT_INDICES_PATH = Path(__file__).resolve().parents[1] / "outputs/eval_indices.json"


def _read_episode_lengths(dataset_root: Path) -> list[tuple[int, int]]:
    episodes_path = dataset_root / "meta/episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(
            f"held-out episodes metadata is absent: {episodes_path}"
        )
    episodes = [
        json.loads(line) for line in episodes_path.read_text().splitlines() if line
    ]
    result = [(int(row["episode_index"]), int(row["length"])) for row in episodes]
    if not result:
        raise ValueError(f"no episodes in {episodes_path}")
    return result


def generate_indices(
    dataset_root: str | Path,
    output_path: str | Path = DEFAULT_INDICES_PATH,
) -> dict:
    """Freeze 20 seed-0 samples with enough room for every model's horizon."""
    dataset_root = Path(dataset_root)
    eligible = [
        (episode, frame)
        for episode, length in _read_episode_lengths(dataset_root)
        for frame in range(max(0, length - MIN_REMAINING_FRAMES + 1))
    ]
    if not eligible:
        raise ValueError(f"no frames have {MIN_REMAINING_FRAMES} remaining steps")
    rng = np.random.RandomState(SEED)
    chosen = [
        eligible[int(index)] for index in rng.randint(0, len(eligible), N_SAMPLES)
    ]
    payload = {
        "repo_id": VAL_REPO,
        "seed": SEED,
        "num_samples": N_SAMPLES,
        "minimum_remaining_frames": MIN_REMAINING_FRAMES,
        "indices": [{"episode": episode, "frame": frame} for episode, frame in chosen],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote frozen indices to {output_path}; review and commit that file",
        flush=True,
    )
    return payload


def load_indices(path: str | Path = DEFAULT_INDICES_PATH) -> list[tuple[int, int]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"frozen eval indices are absent: {path}. Generate them once on Modal CPU with "
            "`python -m src.eval_spec generate-indices --dataset <held-out-v2-path>`, "
            "then review and commit the JSON before running any evaluator."
        )
    payload = json.loads(path.read_text())
    if payload.get("repo_id") != VAL_REPO:
        raise ValueError(
            f"indices target {payload.get('repo_id')!r}, expected {VAL_REPO!r}"
        )
    rows = payload.get("indices", [])
    if len(rows) != N_SAMPLES:
        raise ValueError(f"expected {N_SAMPLES} frozen indices, found {len(rows)}")
    indices = [(int(row["episode"]), int(row["frame"])) for row in rows]
    if any(episode < 0 or frame < 0 for episode, frame in indices):
        raise ValueError("frozen indices contain negative episode/frame values")
    return indices


def raw_absolute_mse(
    prediction: np.ndarray | Sequence,
    ground_truth: np.ndarray | Sequence,
    horizon: int = PRIMARY_HORIZON,
) -> float:
    """MSE in raw absolute joint space over the first 16 predicted steps."""
    prediction64 = np.asarray(prediction, dtype=np.float64)
    ground_truth64 = np.asarray(ground_truth, dtype=np.float64)
    if prediction64.ndim < 2 or ground_truth64.ndim < 2:
        raise ValueError(
            f"prediction and GT must end in [time, action], got "
            f"{prediction64.shape} and {ground_truth64.shape}"
        )
    if prediction64.shape[-1] != ACTION_DIM or ground_truth64.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"raw YAM actions must have {ACTION_DIM} dims, got "
            f"{prediction64.shape[-1]} and {ground_truth64.shape[-1]}"
        )
    if prediction64.shape[-2] < horizon or ground_truth64.shape[-2] < horizon:
        raise ValueError(
            f"both chunks need at least {horizon} steps, got "
            f"{prediction64.shape[-2]} and {ground_truth64.shape[-2]}"
        )
    pred_chunk = prediction64[..., :horizon, :]
    gt_chunk = ground_truth64[..., :horizon, :]
    if pred_chunk.shape != gt_chunk.shape:
        raise ValueError(
            f"prediction/GT shapes differ: {pred_chunk.shape} vs {gt_chunk.shape}"
        )
    return float(np.mean(np.square(pred_chunk - gt_chunk), dtype=np.float64))


def write_runner_output(
    runner: str,
    ground_truth: Mapping[tuple[int, int], np.ndarray | Sequence],
    output_path: str | Path,
    predictions: Mapping[tuple[int, int], np.ndarray | Sequence] | None = None,
    indices_path: str | Path = DEFAULT_INDICES_PATH,
) -> None:
    """Emit the canonical GT/prediction interchange file for any model stack."""
    indices = load_indices(indices_path)
    missing = [index for index in indices if index not in ground_truth]
    if missing:
        raise ValueError(f"runner {runner!r} omitted GT for frozen indices: {missing}")
    if predictions is not None:
        pred_missing = [index for index in indices if index not in predictions]
        if pred_missing:
            raise ValueError(f"runner {runner!r} omitted predictions: {pred_missing}")
    samples = []
    for episode, frame in indices:
        row = {
            "episode": episode,
            "frame": frame,
            "ground_truth": np.asarray(ground_truth[(episode, frame)]).tolist(),
        }
        if predictions is not None:
            row["prediction"] = np.asarray(predictions[(episode, frame)]).tolist()
        samples.append(row)
    payload = {
        "runner": runner,
        "repo_id": VAL_REPO,
        "action_space": "raw_absolute_joint",
        "samples": samples,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_runner(path: Path, indices: list[tuple[int, int]]) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("repo_id") != VAL_REPO:
        raise ValueError(f"{path}: wrong repo_id {payload.get('repo_id')!r}")
    if payload.get("action_space") != "raw_absolute_joint":
        raise ValueError(f"{path}: runner did not declare raw absolute joint space")
    actual_indices = [
        (int(row["episode"]), int(row["frame"])) for row in payload.get("samples", [])
    ]
    if actual_indices != indices:
        raise AssertionError(f"{path}: frozen index order/identity differs")
    return payload


def assert_ground_truth_equal(
    runner_outputs: Iterable[str | Path],
    indices_path: str | Path = DEFAULT_INDICES_PATH,
) -> None:
    """Cast GT to float64, then prove all runners address identical samples."""
    paths = [Path(path) for path in runner_outputs]
    if len(paths) < 2:
        raise ValueError("GT equality requires at least two runner outputs")
    indices = load_indices(indices_path)
    runners = [_load_runner(path, indices) for path in paths]
    reference_name = runners[0]["runner"]
    reference = [
        np.asarray(row["ground_truth"], dtype=np.float64)
        for row in runners[0]["samples"]
    ]
    for runner in runners[1:]:
        for sample_index, (expected, row) in enumerate(
            zip(reference, runner["samples"], strict=True)
        ):
            actual = np.asarray(row["ground_truth"], dtype=np.float64)
            if expected.shape != actual.shape or not np.array_equal(expected, actual):
                max_delta = (
                    float(np.max(np.abs(expected - actual)))
                    if expected.shape == actual.shape
                    else float("inf")
                )
                episode, frame = indices[sample_index]
                raise AssertionError(
                    f"GT mismatch {reference_name!r} vs {runner['runner']!r} at "
                    f"episode={episode}, frame={frame}: shapes={expected.shape}/{actual.shape}, "
                    f"max_abs_delta={max_delta}"
                )


def runner_mean_mse(
    runner_output: str | Path,
    indices_path: str | Path = DEFAULT_INDICES_PATH,
) -> float:
    indices = load_indices(indices_path)
    payload = _load_runner(Path(runner_output), indices)
    errors = []
    for row in payload["samples"]:
        if "prediction" not in row:
            raise ValueError(f"runner {payload['runner']!r} emitted no predictions")
        errors.append(raw_absolute_mse(row["prediction"], row["ground_truth"]))
    return float(np.mean(np.asarray(errors, dtype=np.float64), dtype=np.float64))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-indices")
    generate.add_argument("--dataset", type=Path, required=True)
    generate.add_argument("--output", type=Path, default=DEFAULT_INDICES_PATH)

    compare = subparsers.add_parser("check-gt")
    compare.add_argument("runner_outputs", nargs="+", type=Path)
    compare.add_argument("--indices", type=Path, default=DEFAULT_INDICES_PATH)

    metric = subparsers.add_parser("metric")
    metric.add_argument("runner_output", type=Path)
    metric.add_argument("--indices", type=Path, default=DEFAULT_INDICES_PATH)

    args = parser.parse_args()
    if args.command == "generate-indices":
        generate_indices(args.dataset, args.output)
    elif args.command == "check-gt":
        assert_ground_truth_equal(args.runner_outputs, args.indices)
        print("GT EQUALITY PASSED", flush=True)
    elif args.command == "metric":
        print(f"mean_mse={runner_mean_mse(args.runner_output, args.indices):.10g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
