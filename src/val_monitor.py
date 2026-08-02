"""Checkpoint sidecar for deterministic held-out loss and open-loop MSE."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

from src.modal_app import checkpoint_volume, data_volume, secrets, training_image
from src.reconstruct import reconstruct_model

app = modal.App("dreamzero-yam-val-monitor")
PIPELINE_ROOT = Path("/root/dreamzero_yam")
DATA_VAL = Path("/data/dreamzero_gear/allenai/19012026-block-13")
LOCAL_VAL = Path("/local/dreamzero_gear/allenai/19012026-block-13")
CHECKPOINT_ROOT = Path("/checkpoints/yam")
WANDB_PROJECT = "dreamzero-yam"
HF_MILESTONE_REPO = "robocurve/dreamzero-yam-molmoact2"
VAL_SEED = 20260706
FIXED_SAMPLE_COUNT = 8
ACTION_KEYS = (
    "left_joint_pos",
    "left_gripper_pos",
    "right_joint_pos",
    "right_gripper_pos",
)


def checkpoint_step(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[1])


def checkpoint_is_evaluable(path: Path) -> bool:
    required = (
        path / "trainer_state.json",
        path / "config.json",
        path / "experiment_cfg/conf.yaml",
        path / "experiment_cfg/metadata.json",
        path / "experiment_cfg/trained_parameter_names.json",
    )
    if not all(item.is_file() for item in required):
        return False
    if not (
        (path / "model.safetensors").is_file()
        or (path / "model.safetensors.index.json").is_file()
    ):
        return False
    try:
        state = json.loads((path / "trainer_state.json").read_text())
        return int(state["global_step"]) == checkpoint_step(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def discover_new_milestones(exp_name: str, processed: set[int]) -> list[Path]:
    root = CHECKPOINT_ROOT / exp_name / "milestones"
    candidates = [
        path
        for path in root.glob("checkpoint-*")
        if path.is_dir()
        and checkpoint_step(path) not in processed
        and checkpoint_is_evaluable(path)
    ]
    return sorted(candidates, key=checkpoint_step)


def _copy_val_to_nvme() -> None:
    if not DATA_VAL.is_dir():
        raise FileNotFoundError(f"held-out DreamZero repo is absent: {DATA_VAL}")
    LOCAL_VAL.parent.mkdir(parents=True, exist_ok=True)
    if not LOCAL_VAL.exists():
        subprocess.run(["cp", "-aL", str(DATA_VAL), str(LOCAL_VAL.parent)], check=True)


def _merged_metadata(checkpoint: Path) -> Any:
    from groot.vla.data.schema import DatasetMetadata

    payload = json.loads((checkpoint / "experiment_cfg/metadata.json").read_text())
    if "yam" not in payload:
        raise KeyError("checkpoint merged metadata has no 'yam' embodiment")
    return DatasetMetadata.model_validate(payload["yam"])


def build_fixed_val_batches(
    checkpoint: Path, sample_count: int = FIXED_SAMPLE_COUNT
) -> tuple[list[Any], Any]:
    """Build the same transformed held-out samples for every checkpoint."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(checkpoint / "experiment_cfg/conf.yaml")
    cfg.train_dataset.mixture_spec[0].dataset_path.yam = [str(LOCAL_VAL)]
    cfg.train_dataset.mixture_kwargs.training = False
    cfg.train_dataset.mixture_kwargs.seed = VAL_SEED
    dataset = instantiate(cfg.train_dataset)
    if len(dataset.datasets) != 1:
        raise AssertionError(
            f"expected one held-out dataset, got {len(dataset.datasets)}"
        )
    single_dataset = dataset.datasets[0]
    transform = single_dataset.transforms
    transform.set_metadata(_merged_metadata(checkpoint))
    # Keep the training transform path (including DreamTransform's unbatched
    # output contract) but freeze every augmentation/dropout draw so the
    # decoded held-out inputs are checkpoint-comparable.
    transform.train()
    if hasattr(dataset, "reset_seed"):
        dataset.reset_seed(VAL_SEED)
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(VAL_SEED)
        np.random.seed(VAL_SEED)
        torch.manual_seed(VAL_SEED)
        iterator = iter(dataset)
        samples = [next(iterator) for _ in range(sample_count)]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
    collator = instantiate(cfg.data_collator)
    # Batch samples in PAIRS: upstream's training forward (reused for the
    # deterministic val/loss) breaks at batch 1 (tuple modulation shapes).
    if len(samples) % 2:
        samples = samples[:-1]
    pairs = [samples[i : i + 2] for i in range(0, len(samples), 2)]
    return [collator(pair) for pair in pairs], transform


def _to_device(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def deterministic_flow_matching_loss(
    model: Any, batches: list[Any], seed: int = VAL_SEED
) -> float:
    """Evaluate the training forward objective with fixed noise/timestep draws."""
    import torch

    device = next(model.parameters()).device
    losses = []
    was_training = model.training
    model.eval()
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            for batch in batches:
                output = model(_to_device(batch, device))
                loss = output["loss"] if isinstance(output, dict) else output.loss
                losses.append(float(loss.detach().float().cpu()))
    if was_training:
        model.train()
    if not losses:
        raise AssertionError("fixed held-out set produced no flow-matching losses")
    return sum(losses) / len(losses)


def _decode_absolute_action(
    transform: Any,
    normalized_action: Any,
    normalized_state: Any,
) -> Any:
    """Invert merged-stat normalization, then undo all-14D relative action."""
    import numpy as np

    decoded = transform.unapply(
        {
            "action": normalized_action.detach().float().cpu().clone(),
            "state": normalized_state.detach().float().cpu().clone(),
        }
    )
    action_parts = []
    for key in ACTION_KEYS:
        action_key = f"action.{key}"
        state_key = f"state.{key}"
        if action_key not in decoded or state_key not in decoded:
            raise KeyError(f"inverse transform omitted {action_key!r} or {state_key!r}")
        relative = np.asarray(decoded[action_key], dtype=np.float64)
        state = np.asarray(decoded[state_key], dtype=np.float64)
        reference = state[..., -1, :]
        if relative.ndim > reference.ndim:
            reference = np.expand_dims(reference, axis=-2)
        action_parts.append(relative + reference)
    absolute = np.concatenate(action_parts, axis=-1)
    if absolute.shape[-1] != 14:
        raise AssertionError(f"decoded YAM action is not 14-D: {absolute.shape}")
    return absolute


def teacher_forced_open_loop_mse(
    model: Any, batches: list[Any], transform: Any
) -> float:
    """Run GT-video-conditioned action prediction and score raw absolute joints."""
    import torch

    from src.eval_spec import raw_absolute_mse

    device = next(model.parameters()).device
    errors = []
    model.eval()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ),
    ):
        for cpu_batch in batches:
            batch = _to_device(cpu_batch, device)
            # The head is stateful (language cache, current_start_frame, KV
            # caches) and latent_video=None alone does NOT reset it; upstream
            # only resets on language change / single-frame video / frame
            # overflow. Force the `language is None` reset branch so every
            # sample is evaluated stateless.
            model.action_head.language = None
            model.action_head.current_start_frame = 0
            output = model.lazy_joint_video_action_causal(batch, latent_video=None)
            prediction = output["action_pred"]
            if "action" not in batch or "state" not in batch:
                raise KeyError("held-out collated batch lacks action/state tensors")
            prediction_absolute = _decode_absolute_action(
                transform, prediction, batch["state"]
            )
            ground_truth_absolute = _decode_absolute_action(
                transform, batch["action"], batch["state"]
            )
            errors.append(
                raw_absolute_mse(prediction_absolute, ground_truth_absolute, horizon=16)
            )
    if not errors:
        raise AssertionError("fixed held-out set produced no open-loop predictions")
    return sum(errors) / len(errors)


def evaluate_checkpoint(checkpoint: Path) -> dict[str, float]:
    """Load only through explicit AgiBot reconstruction and compute both metrics."""
    import torch

    batches, transform = build_fixed_val_batches(checkpoint)
    model, _, _ = reconstruct_model(checkpoint, device="cuda")
    model.to(dtype=torch.bfloat16)
    model.post_initialize()
    metrics = {
        "val/loss": deterministic_flow_matching_loss(model, batches),
        "val/open_loop_mse": teacher_forced_open_loop_mse(model, batches, transform),
    }
    del model
    torch.cuda.empty_cache()
    return metrics


def _read_monitor_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"processed_steps": [], "uploaded_steps": [], "in_progress": None}
    payload = json.loads(path.read_text())
    payload.setdefault("processed_steps", [])
    payload.setdefault("uploaded_steps", [])
    payload.setdefault("in_progress", None)
    return payload


def _write_monitor_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def _upload_milestone_to_hf(checkpoint: Path, step: int) -> None:
    """Mirror an evaluated milestone to HF under checkpoints/step-N/."""
    from huggingface_hub import HfApi

    # hf-robocurve-write exposes HF_WRITE_TOKEN (verified via Modal preflight);
    # HF_TOKEN belongs to the read-only huggingface-token secret.
    write_token = os.environ.get("HF_WRITE_TOKEN")
    if not write_token:
        raise RuntimeError(
            "HF_WRITE_TOKEN missing: hf-robocurve-write secret not mounted or renamed"
        )
    api = HfApi(token=write_token)
    api.create_repo(HF_MILESTONE_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=HF_MILESTONE_REPO,
        repo_type="model",
        folder_path=str(checkpoint),
        path_in_repo=f"checkpoints/step-{step}",
        commit_message=f"Milestone checkpoint step {step}",
    )
    print(f"uploaded milestone step {step} to {HF_MILESTONE_REPO}", flush=True)


@app.function(
    image=training_image,
    gpu="H200:1",
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    # hf-robocurve-write exposes HF_WRITE_TOKEN, used only for milestone uploads;
    # HF_TOKEN (read) still comes from the shared huggingface-token secret.
    secrets=[*secrets, modal.Secret.from_name("hf-robocurve-write")],
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=60.0),
    cpu=16,
    memory=196_608,
    ephemeral_disk=1_200_000,
)
def val_monitor(
    exp_name: str = "tune-armA",
    heartbeat_seconds: int = 60,
    max_checkpoints: int = 0,
    upload_to_hf: bool = True,
) -> None:
    """Watch complete milestone copies, with durable heartbeat and duplicate guard."""
    import wandb

    _copy_val_to_nvme()
    state_path = CHECKPOINT_ROOT / "val-monitor" / f"{exp_name}.json"
    runtime_path = CHECKPOINT_ROOT / "val-monitor" / f"{exp_name}-runtime.jsonl"
    state = _read_monitor_state(state_path)
    processed = {int(step) for step in state["processed_steps"]}
    uploaded = {int(step) for step in state["uploaded_steps"]}
    run = wandb.init(
        project=WANDB_PROJECT,
        id=f"{exp_name}-val-monitor",
        name="val-monitor",
        group=exp_name,
        job_type="validation-monitor",
        resume="allow",
    )
    telemetry_env = {
        **os.environ,
        "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_NAME": "val-monitor-gpu-metrics",
        "WANDB_RUN_GROUP": exp_name,
    }
    telemetry = subprocess.Popen(
        [
            sys.executable,
            str(PIPELINE_ROOT / "src/gpu_metrics.py"),
            "--stage",
            "val-monitor",
            "--gpu-spec",
            "H200:1",
            "--runtime-metrics",
            str(runtime_path),
        ],
        cwd=PIPELINE_ROOT,
        env=telemetry_env,
    )
    evaluated_this_invocation = 0
    try:
        while True:
            checkpoint_volume.reload()
            pending = discover_new_milestones(exp_name, processed)
            for checkpoint in pending:
                step = checkpoint_step(checkpoint)
                state["in_progress"] = step
                state["heartbeat_unix"] = time.time()
                _write_monitor_state(state_path, state)
                checkpoint_volume.commit()
                started = time.monotonic()
                metrics = evaluate_checkpoint(checkpoint)
                elapsed = time.monotonic() - started
                with runtime_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "samples_per_second": FIXED_SAMPLE_COUNT / elapsed,
                            }
                        )
                        + "\n"
                    )
                run.log(metrics, step=step)
                if upload_to_hf and step not in uploaded:
                    _upload_milestone_to_hf(checkpoint, step)
                    uploaded.add(step)
                    state["uploaded_steps"] = sorted(uploaded)
                processed.add(step)
                state["processed_steps"] = sorted(processed)
                state["in_progress"] = None
                state["heartbeat_unix"] = time.time()
                state["last_metrics"] = {"step": step, **metrics}
                _write_monitor_state(state_path, state)
                checkpoint_volume.commit()
                print(f"VAL checkpoint={step} metrics={metrics}", flush=True)
                evaluated_this_invocation += 1
                if max_checkpoints and evaluated_this_invocation >= max_checkpoints:
                    return
            state["heartbeat_unix"] = time.time()
            _write_monitor_state(state_path, state)
            checkpoint_volume.commit()
            time.sleep(max(10, heartbeat_seconds))
    finally:
        telemetry.terminate()
        try:
            telemetry.wait(timeout=20)
        except subprocess.TimeoutExpired:
            telemetry.kill()
            telemetry.wait()
        run.finish()
