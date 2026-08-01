"""Phase-2 DreamZero smoke, preemption, resume, and reconstruction drills."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

DREAMZERO_ROOT = Path("/root/dreamzero")
PIPELINE_ROOT = Path("/root/dreamzero_yam")
DEFAULT_DATASET = Path("/local/dreamzero_gear/allenai/01122025-box-01")
DEFAULT_OUTPUT = Path("/checkpoints/yam/smoke")
FIRST_CHECKPOINT_STEP = 20
FINAL_STEP = 40
SAVE_STEPS = 20
EXPECTED_WORLD_SIZE = 2


def assert_complete_checkpoint(
    checkpoint: Path,
    expected_world_size: int = EXPECTED_WORLD_SIZE,
    require_resume_probes: bool = True,
) -> None:
    required = (
        checkpoint / "trainer_state.json",
        checkpoint / "config.json",
        checkpoint / "experiment_cfg/conf.yaml",
        checkpoint / "experiment_cfg/metadata.json",
    )
    absent = [str(path) for path in required if not path.is_file()]
    if not (
        (checkpoint / "model.safetensors").is_file()
        or (checkpoint / "model.safetensors.index.json").is_file()
    ):
        absent.append(f"{checkpoint}/model.safetensors[.index.json]")
    global_state_dirs = [
        path for path in checkpoint.glob("global_step*") if path.is_dir()
    ]
    optimizer_files = [
        path for root in global_state_dirs for path in root.glob("*optim_states.pt")
    ]
    model_state_files = [
        path for root in global_state_dirs for path in root.glob("*model_states.pt")
    ]
    if len(optimizer_files) < expected_world_size:
        absent.append(
            f"DeepSpeed optimizer shards ({len(optimizer_files)}/{expected_world_size})"
        )
    if len(model_state_files) < expected_world_size:
        absent.append(
            f"DeepSpeed model shards ({len(model_state_files)}/{expected_world_size})"
        )
    if require_resume_probes:
        for rank in range(expected_world_size):
            probe = checkpoint / f"resume_probe_rank_{rank}.json"
            if not probe.is_file():
                absent.append(str(probe))
    if absent:
        raise AssertionError(f"checkpoint is incomplete: {checkpoint}: {absent}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    step = int(checkpoint.name.rsplit("-", 1)[1])
    assert int(state["global_step"]) == step, (
        f"trainer_state step {state['global_step']} != checkpoint directory step {step}"
    )


def _training_command(dataset: Path, output: Path, num_gpus: int) -> list[str]:
    from src.reconstruct import AGIBOT_CKPT_DIR, UMT5_CKPT_DIR, WAN_CKPT_DIR

    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        str(PIPELINE_ROOT / "src/train_entry.py"),
        "report_to=wandb",
        "data=dreamzero/yam_relative",
        "wandb_project=dreamzero-yam",
        "train_architecture=lora",
        "num_frames=33",
        "action_horizon=24",
        "num_views=3",
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf",
        "model/dreamzero/transform=dreamzero_cotrain",
        "num_frame_per_block=2",
        "num_action_per_block=24",
        "num_state_per_block=1",
        "seed=42",
        "training_args.learning_rate=1e-5",
        "training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json",
        "training_args.warmup_ratio=0.01",
        f"output_dir={output}",
        "per_device_train_batch_size=1",
        f"max_steps={FINAL_STEP}",
        "weight_decay=1e-5",
        "save_total_limit=5",
        "upload_checkpoints=false",
        "bf16=true",
        "tf32=true",
        "eval_bf16=true",
        "dataloader_pin_memory=false",
        "dataloader_num_workers=1",
        "image_resolution_width=320",
        "image_resolution_height=176",
        "save_lora_only=true",
        "max_chunk_size=4",
        "frame_seqlen=880",
        "save_strategy=steps",
        f"save_steps={SAVE_STEPS}",
        "logging_steps=1",
        f"yam_data_root={dataset}",
        f"dit_version={WAN_CKPT_DIR}",
        f"text_encoder_pretrained_path={WAN_CKPT_DIR / 'models_t5_umt5-xxl-enc-bf16.pth'}",
        f"image_encoder_pretrained_path={WAN_CKPT_DIR / 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'}",
        f"vae_pretrained_path={WAN_CKPT_DIR / 'Wan2.1_VAE.pth'}",
        f"tokenizer_path={UMT5_CKPT_DIR}",
        f"pretrained_model_path={AGIBOT_CKPT_DIR}",
        "++action_head_cfg.config.skip_component_loading=true",
        "++action_head_cfg.config.defer_lora_injection=true",
        "action_head_cfg.config.lora_rank=4",
        "action_head_cfg.config.lora_alpha=4",
    ]


def _start_training(
    command: list[str], log_path: Path, assert_resume: bool
) -> tuple[subprocess.Popen, object]:
    env = {
        **os.environ,
        "DREAMZERO_SAVE_FIXED_SAMPLE": "1",
        "DREAMZERO_ASSERT_RESUME": "1" if assert_resume else "0",
        "DREAMZERO_GLOBAL_BATCH": "2",
        "DREAMZERO_STAGE": "smoke",
        "WANDB_PROJECT": "dreamzero-yam",
        "WANDB_NAME": "smoke",
        "WANDB_RUN_GROUP": "smoke",
        "PYTHONPATH": f"{PIPELINE_ROOT}:{DREAMZERO_ROOT}:{os.environ.get('PYTHONPATH', '')}",
        "HYDRA_FULL_ERROR": "1",
    }
    log_file = log_path.open("ab")
    process = subprocess.Popen(
        command,
        cwd=DREAMZERO_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_file


def _dump_log_tail(log_path: Path, lines: int = 150) -> None:
    """Echo the child log to container stdout — volume commits can be lost on
    crash-shutdown, but streamed stdout always reaches Modal app logs."""
    try:
        content = log_path.read_text(errors="replace").splitlines()[-lines:]
    except OSError as error:
        print(f"[child log unreadable: {error}]", flush=True)
        return
    print(f"===== child log tail ({log_path}) =====", flush=True)
    for line in content:
        print(line, flush=True)
    print("===== end child log tail =====", flush=True)


def _stop_process_group(
    process: subprocess.Popen, timeout: float = 180.0, hard: bool = False
) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGKILL if hard else signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def kill_and_resume_drill(
    dataset: Path,
    output: Path,
    num_gpus: int = EXPECTED_WORLD_SIZE,
    checkpoint_timeout: float = 4 * 60 * 60,
) -> Path:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    log_path = output / "smoke.log"
    command = _training_command(dataset, output, num_gpus)

    process, log_file = _start_training(command, log_path, assert_resume=False)
    first = output / f"checkpoint-{FIRST_CHECKPOINT_STEP}"
    deadline = time.monotonic() + checkpoint_timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                _dump_log_tail(log_path)
                raise RuntimeError(
                    f"initial smoke process exited {process.returncode} before checkpoint "
                    f"{FIRST_CHECKPOINT_STEP}; see {log_path}"
                )
            try:
                assert_complete_checkpoint(first, expected_world_size=num_gpus)
                break
            except (AssertionError, FileNotFoundError, json.JSONDecodeError):
                time.sleep(10)
        else:
            _dump_log_tail(log_path)
            raise TimeoutError(f"timed out waiting for complete checkpoint: {first}")
    finally:
        # Hard kill (SIGKILL, no graceful flush) models a real preemption; the
        # resume asserts must hold without any shutdown-time checkpointing.
        _stop_process_group(process, hard=True)
        log_file.close()

    # A preemption may leave a newer partial directory. Never let upstream's
    # latest-checkpoint discovery select one on the resumed launch.
    for checkpoint in output.glob("checkpoint-*"):
        if checkpoint == first:
            continue
        try:
            assert_complete_checkpoint(checkpoint, expected_world_size=num_gpus)
        except Exception:
            shutil.rmtree(checkpoint)

    resumed, resumed_log = _start_training(command, log_path, assert_resume=True)
    try:
        return_code = resumed.wait(timeout=checkpoint_timeout)
    except subprocess.TimeoutExpired:
        _stop_process_group(resumed)
        _dump_log_tail(log_path)
        raise TimeoutError("resumed smoke process did not finish")
    finally:
        resumed_log.close()
    if return_code != 0:
        _dump_log_tail(log_path)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    final = output / f"checkpoint-{FINAL_STEP}"
    assert_complete_checkpoint(final, expected_world_size=num_gpus)
    for rank in range(num_gpus):
        marker = output / f"resume_verified_rank_{rank}.json"
        if not marker.is_file():
            raise AssertionError(
                f"resume callback did not verify rank {rank}: {marker}"
            )
        assert int(json.loads(marker.read_text())["step"]) == FIRST_CHECKPOINT_STEP
    return final


def assert_resume_loss_continuity(
    output: Path, boundary: int = FIRST_CHECKPOINT_STEP
) -> None:
    path = output / "runtime_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    points = [(int(row["step"]), float(row["loss"])) for row in rows if "loss" in row]
    before = [loss for step, loss in points if step <= boundary][-10:]
    after = [loss for step, loss in points if boundary < step <= boundary + 20]
    if len(before) < 5 or len(after) < 15:
        raise AssertionError(
            f"insufficient loss points around resume: before={len(before)}, after={len(after)}"
        )
    if not all(math.isfinite(loss) for loss in before + after):
        raise AssertionError("non-finite loss around resume")
    before_center = sorted(before)[len(before) // 2]
    after_center = sorted(after)[len(after) // 2]
    if after_center > 1.3 * before_center:
        raise AssertionError(
            f"post-resume loss discontinuity: median {after_center:.6g} > "
            f"1.3 x pre-resume {before_center:.6g}"
        )


def assert_resume_markers(output: Path, num_gpus: int) -> None:
    required_hashes = {"lora_hash", "action_head_hash", "optimizer_hash"}
    for rank in range(num_gpus):
        marker = output / f"resume_verified_rank_{rank}.json"
        record = json.loads(marker.read_text())
        assert int(record["step"]) == FIRST_CHECKPOINT_STEP
        for name in required_hashes:
            value = record.get(name, "")
            assert len(value) == 64, f"invalid {name} for rank {rank}: {value!r}"


def settle_annotation_key(dataset: Path) -> str:
    from src.prepare_data import _loader_dry_run

    # This traverses the numeric parquet task_index -> tasks.jsonl resolution
    # in DreamZero's loader. A missing/subkey-mismatched/empty prompt fails.
    _loader_dry_run(dataset, samples=2, seed=0)
    return "annotation.task"


def settle_max_chunk_size(dataset: Path) -> int:
    from src.prepare_data import _loader_dry_run

    # _loader_dry_run asserts 24 x 4 action rows from the real sharded class.
    _loader_dry_run(dataset, samples=2, seed=1)
    return 4


def assert_pretrained_load_integrity(log_path: Path) -> None:
    log = log_path.read_text(errors="replace")
    expected = "Strictly loading pretrained weights from:"
    if expected not in log:
        raise AssertionError("instrumented aggregate AgiBot load did not run")
    if "AgiBot load integrity failed" in log:
        raise AssertionError(
            "AgiBot load-integrity assertion failed; inspect smoke.log"
        )


def assert_merged_metadata_ranges(checkpoint: Path, eps: float = 1e-8) -> None:
    metadata_path = checkpoint / "experiment_cfg/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    yam = metadata.get("yam")
    if yam is None:
        raise AssertionError(
            f"merged checkpoint metadata has no yam embodiment: {metadata.keys()}"
        )
    action_stats = yam["statistics"]["action"]
    expected_keys = {
        "left_joint_pos",
        "left_gripper_pos",
        "right_joint_pos",
        "right_gripper_pos",
    }
    if set(action_stats) != expected_keys:
        raise AssertionError(
            f"merged action-stat keys differ: actual={set(action_stats)} "
            f"expected={expected_keys}"
        )
    offenders = []
    for key, stats in action_stats.items():
        for dim, (q01, q99) in enumerate(zip(stats["q01"], stats["q99"], strict=True)):
            if not float(q99) - float(q01) > eps:
                offenders.append(f"{key}[{dim}] q01={q01} q99={q99}")
    if offenders:
        raise AssertionError("degenerate merged q99 ranges: " + "; ".join(offenders))


def assert_eval_reconstruction(final_checkpoint: Path, device: str = "cuda:0") -> None:
    import torch

    from src.reconstruct import assert_eval_reconstruction_parity, reconstruct_model

    model, base_integrity, overlay_integrity = reconstruct_model(
        final_checkpoint, device=device
    )
    model.to(dtype=torch.bfloat16)
    assert not base_integrity.missing_keys and not base_integrity.unexpected_keys
    assert not overlay_integrity.unexpected_keys
    assert (
        overlay_integrity.checkpoint_keys == overlay_integrity.trained_parameter_names
    )
    # Parity must run BEFORE post_initialize: the reference was captured from
    # the eager training module, and post_initialize torch.compiles the text/
    # image/vae encoders on the parity path — inductor output is not
    # guaranteed bit-identical to eager. val_monitor keeps post_initialize
    # (upstream eval convention); its metrics don't require bit identity.
    assert_eval_reconstruction_parity(model, final_checkpoint)


def _run_check(name: str, check: Callable[[], object]) -> None:
    print(f"[RUN ] {name}", flush=True)
    result = check()
    suffix = f": {result}" if result is not None else ""
    print(f"[PASS] {name}{suffix}", flush=True)


def run_smoke(dataset: Path, output: Path, num_gpus: int = EXPECTED_WORLD_SIZE) -> None:
    if num_gpus != EXPECTED_WORLD_SIZE:
        raise ValueError("Phase-2 smoke is defined for exactly 2 GPUs")
    _run_check("annotation key resolution", lambda: settle_annotation_key(dataset))
    _run_check(
        "max_chunk_size=4 / action length", lambda: settle_max_chunk_size(dataset)
    )
    final_holder: dict[str, Path] = {}

    def train_resume() -> Path:
        final = kill_and_resume_drill(dataset, output, num_gpus=num_gpus)
        final_holder["checkpoint"] = final
        return final

    _run_check("20-step kill + 20-step resume", train_resume)
    _run_check(
        "LoRA/head/optimizer bit-identical resume",
        lambda: assert_resume_markers(output, num_gpus),
    )
    _run_check(
        "20-step post-resume loss continuity",
        lambda: assert_resume_loss_continuity(output),
    )
    _run_check(
        "AgiBot pretrained-load integrity",
        lambda: assert_pretrained_load_integrity(output / "smoke.log"),
    )
    _run_check(
        "merged-metadata nondegenerate ranges",
        lambda: assert_merged_metadata_ranges(final_holder["checkpoint"]),
    )
    _run_check(
        "AgiBot+LoRA eval reconstruction parity",
        lambda: assert_eval_reconstruction(final_holder["checkpoint"]),
    )
    print("ALL PHASE-2 SMOKE CHECKS PASSED", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-gpus", type=int, default=EXPECTED_WORLD_SIZE)
    args = parser.parse_args()
    run_smoke(args.dataset, args.output, args.num_gpus)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
