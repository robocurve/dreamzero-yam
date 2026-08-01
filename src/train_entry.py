"""Instrumented DreamZero training entrypoint.

The upstream experiment is retained, but pretrained loading is made strict at
the aggregate shard level and callbacks record the artifacts needed to prove
resume and evaluation reconstruction parity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

RISING_TAIL_FRACTION = 0.2
RISING_TAIL_MIN_POINTS = 5


def _scalar_token(value: Any) -> str | None:
    if value is None:
        return "none:None"
    if isinstance(value, bool):
        return f"bool:{value!r}"
    if isinstance(value, int):
        return f"int:{value!r}"
    if isinstance(value, float):
        return f"float:{value!r}"
    if isinstance(value, str):
        return f"str:{value!r}"
    return None


def _hash_value(value: Any, digest: Any, prefix: str = "root") -> None:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(prefix.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        keyed_values = [
            (token, item)
            for item in value
            if (token := _scalar_token(item)) is not None
        ]
        for token, key in sorted(keyed_values):
            _hash_value(value[key], digest, f"{prefix}.{token}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _hash_value(item, digest, f"{prefix}.{index}")
    elif (token := _scalar_token(value)) is not None:
        digest.update(prefix.encode())
        digest.update(token.encode())


def canonical_hash(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(value, digest)
    return digest.hexdigest()


def normalized_rising_tail(losses: list[float]) -> float:
    """Return the fitted fractional loss change over the trailing arm window."""
    if len(losses) < RISING_TAIL_MIN_POINTS:
        return math.nan
    count = max(RISING_TAIL_MIN_POINTS, math.ceil(len(losses) * RISING_TAIL_FRACTION))
    tail = losses[-count:]
    x_center = (count - 1) / 2
    y_center = statistics.fmean(tail)
    denominator = sum((index - x_center) ** 2 for index in range(count))
    slope = sum(
        (index - x_center) * (loss - y_center) for index, loss in enumerate(tail)
    ) / max(denominator, 1.0)
    scale = statistics.median(abs(loss) for loss in tail)
    return slope * (count - 1) / scale if scale else math.nan


def _unwrap(trainer: Any) -> Any:
    try:
        return trainer.accelerator.unwrap_model(trainer.model)
    except Exception:
        return getattr(trainer.model, "module", trainer.model)


def _category_state(model: Any) -> dict[str, dict[str, Any]]:
    named_parameters = dict(model.named_parameters())
    categories = {"lora": {}, "action_head": {}}
    for name, parameter in named_parameters.items():
        if "lora_A" in name or "lora_B" in name:
            categories["lora"][name] = parameter
        elif (
            name.startswith("action_head.")
            and parameter.requires_grad
            and "lora_" not in name
        ):
            categories["action_head"][name] = parameter
    if not categories["lora"] or not categories["action_head"]:
        raise AssertionError(
            f"resume probes need LoRA and action-head tensors, got sizes "
            f"{ {key: len(value) for key, value in categories.items()} }"
        )
    return categories


def _first_batch_item(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value[:1].detach().cpu()
    if isinstance(value, np.ndarray):
        return value[:1].copy()
    if isinstance(value, dict):
        return {key: _first_batch_item(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_first_batch_item(item) for item in value)
    if isinstance(value, list):
        return value[:1]
    return value


def _save_fixed_sample_reference(trainer: Any, checkpoint_dir: Path) -> None:
    import torch
    from safetensors.torch import save_file

    from src.reconstruct import (
        FIXED_SAMPLE_INPUT,
        FIXED_SAMPLE_OUTPUT,
        fixed_sample_policy_call,
        flatten_tensor_tree,
    )

    sample = getattr(trainer, "_dreamzero_fixed_sample", None)
    if sample is None:
        raise AssertionError(
            "no fixed training sample was captured before checkpoint save"
        )
    model = _unwrap(trainer)
    device_sample = trainer._prepare_inputs(sample)
    was_training = model.training
    model.eval()
    devices = [model.device] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        if devices:
            torch.cuda.manual_seed_all(0)
        with torch.inference_mode():
            output = fixed_sample_policy_call(model, device_sample)
    if was_training:
        model.train()
    tensors = flatten_tensor_tree(output)
    if not tensors:
        raise AssertionError("fixed sample produced no tensor policy outputs")
    torch.save(sample, checkpoint_dir / FIXED_SAMPLE_INPUT)
    save_file(tensors, str(checkpoint_dir / FIXED_SAMPLE_OUTPUT))


def _save_parity_reference(model: Any, checkpoint_dir: Path) -> None:
    from safetensors.torch import save_file

    from src.reconstruct import PARITY_REFERENCE_NAMES, PARITY_REFERENCE_TENSORS

    state = model.state_dict()
    categories = {
        "base_layer": next((name for name in state if ".base_layer." in name), None),
        "lora": next(
            (name for name in state if "lora_A" in name or "lora_B" in name), None
        ),
        "action_head": next(
            (
                name
                for name, parameter in model.named_parameters()
                if name.startswith("action_head.")
                and parameter.requires_grad
                and "lora_" not in name
            ),
            None,
        ),
    }
    missing = [category for category, name in categories.items() if name is None]
    if missing:
        raise AssertionError(f"could not select training parity tensors: {missing}")
    names = {category: str(name) for category, name in categories.items()}
    tensors = {
        category: state[name].detach().cpu().contiguous()
        for category, name in names.items()
    }
    save_file(tensors, str(checkpoint_dir / PARITY_REFERENCE_TENSORS))
    (checkpoint_dir / PARITY_REFERENCE_NAMES).write_text(
        json.dumps(names, indent=2) + "\n"
    )


class IntegrityCallback(TrainerCallback):
    """Record and verify model/optimizer state at checkpoint boundaries."""

    def __init__(self, trainer: Any):
        super().__init__()
        self.trainer = trainer

    def on_save(
        self, args: Any, state: Any, control: Any, optimizer: Any = None, **_: Any
    ) -> None:
        import torch.distributed as dist

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        model = _unwrap(self.trainer)
        categories = _category_state(model)
        if optimizer is None:
            optimizer = self.trainer.optimizer
        probe = {
            "step": int(state.global_step),
            "rank": rank,
            "lora_hash": canonical_hash(categories["lora"]),
            "action_head_hash": canonical_hash(categories["action_head"]),
            "optimizer_hash": canonical_hash(optimizer.state_dict()),
        }
        (checkpoint_dir / f"resume_probe_rank_{rank}.json").write_text(
            json.dumps(probe, indent=2) + "\n"
        )
        if state.is_world_process_zero:
            _save_parity_reference(model, checkpoint_dir)
            if os.environ.get("DREAMZERO_SAVE_FIXED_SAMPLE", "0") == "1":
                _save_fixed_sample_reference(self.trainer, checkpoint_dir)

    def on_train_begin(
        self, args: Any, state: Any, control: Any, optimizer: Any = None, **_: Any
    ) -> None:
        if (
            os.environ.get("DREAMZERO_ASSERT_RESUME") != "1"
            or int(state.global_step) == 0
        ):
            return
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        reference_path = checkpoint_dir / f"resume_probe_rank_{rank}.json"
        if not reference_path.is_file():
            raise FileNotFoundError(f"resume reference is absent: {reference_path}")
        reference = json.loads(reference_path.read_text())
        model = _unwrap(self.trainer)
        categories = _category_state(model)
        if optimizer is None:
            optimizer = self.trainer.optimizer
        actual = {
            "lora_hash": canonical_hash(categories["lora"]),
            "action_head_hash": canonical_hash(categories["action_head"]),
            "optimizer_hash": canonical_hash(optimizer.state_dict()),
        }
        mismatches = {
            key: {"saved": reference[key], "resumed": actual[key]}
            for key in actual
            if reference.get(key) != actual[key]
        }
        if mismatches:
            raise AssertionError(
                f"resume state is not bit-identical on rank {rank}: {mismatches}"
            )
        marker = Path(args.output_dir) / f"resume_verified_rank_{rank}.json"
        marker.write_text(
            json.dumps({"step": int(state.global_step), **actual}, indent=2) + "\n"
        )


class _TimedDataLoader:
    """Delegating loader that accrues blocking dataloader-wait seconds."""

    def __init__(self, loader: Any, trainer: Any) -> None:
        self._loader = loader
        self._trainer = trainer

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def __iter__(self) -> Any:
        iterator = iter(self._loader)
        while True:
            started = time.monotonic()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            waited = time.monotonic() - started
            self._trainer._dreamzero_dataloader_wait_s = (
                getattr(self._trainer, "_dreamzero_dataloader_wait_s", 0.0) + waited
            )
            yield batch


class StageMetricsCallback(TrainerCallback):
    """Log tune instability screens and measured end-to-end throughput."""

    def __init__(self, trainer: Any = None) -> None:
        super().__init__()
        self.trainer = trainer
        self.losses: list[float] = []
        self.grad_norms: list[float] = []
        self.last_time: float | None = None
        self.last_step: int | None = None
        self.last_wait_total = 0.0

    def on_log(
        self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any
    ) -> None:
        if not state.is_world_process_zero or not logs:
            return
        loss_value = logs.get("loss", logs.get("train/loss"))
        if loss_value is None:
            return
        loss = float(loss_value)
        grad_raw = logs.get("grad_norm", logs.get("train/grad_norm"))
        grad_norm = float(grad_raw) if grad_raw is not None else math.nan
        loss_median = statistics.median(self.losses[-100:]) if self.losses else loss
        finite_gradients = [
            value for value in self.grad_norms[-100:] if math.isfinite(value)
        ]
        grad_median = (
            statistics.median(finite_gradients) if finite_gradients else grad_norm
        )
        finite_losses = [*self.losses, loss] if math.isfinite(loss) else self.losses
        now = time.monotonic()
        step = int(state.global_step)
        global_batch = int(os.environ.get("DREAMZERO_GLOBAL_BATCH", "1"))
        samples_per_second = 0.0
        if (
            self.last_time is not None
            and self.last_step is not None
            and step > self.last_step
        ):
            elapsed = now - self.last_time
            if elapsed > 0:
                samples_per_second = (step - self.last_step) * global_batch / elapsed
        wait_total = float(
            getattr(self.trainer, "_dreamzero_dataloader_wait_s", 0.0) or 0.0
        )
        wait_delta = max(0.0, wait_total - self.last_wait_total)
        self.last_wait_total = wait_total
        dataloader_wait_seconds_per_step = 0.0
        dataloader_wait_fraction = 0.0
        if (
            self.last_time is not None
            and self.last_step is not None
            and step > self.last_step
            and now > self.last_time
        ):
            dataloader_wait_seconds_per_step = wait_delta / (step - self.last_step)
            dataloader_wait_fraction = wait_delta / (now - self.last_time)
        metrics = {
            "instability/nan_flag": float(
                not math.isfinite(loss)
                or (grad_raw is not None and not math.isfinite(grad_norm))
            ),
            "instability/loss_spike_ratio": (
                loss / loss_median if loss_median and math.isfinite(loss) else math.inf
            ),
            "instability/grad_norm_ratio": (
                grad_norm / grad_median
                if grad_median and math.isfinite(grad_norm)
                else math.nan
            ),
            "instability/rising_tail": normalized_rising_tail(finite_losses),
            "perf/samples_per_second": samples_per_second,
            "perf/dataloader_wait_seconds_per_step": dataloader_wait_seconds_per_step,
            "perf/dataloader_wait_fraction": dataloader_wait_fraction,
        }
        row = {
            "time_monotonic": now,
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "samples_per_second": samples_per_second,
            **metrics,
        }
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "runtime_metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=True) + "\n")
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"[stage metrics] WandB log failed: {type(exc).__name__}: {exc}")
        if math.isfinite(loss):
            self.losses.append(loss)
        if math.isfinite(grad_norm):
            self.grad_norms.append(grad_norm)
        self.last_time = now
        self.last_step = step


def _install_training_patches() -> None:
    from groot.vla.experiment import base as base_module
    from groot.vla.experiment import experiment as experiment_module
    from groot.vla.experiment.utils import mprint
    from hydra.utils import instantiate

    from src.reconstruct import (
        TRAINED_PARAMETER_NAMES,
        agibot_allowed_missing,
        load_agibot_shards,
    )

    def create_model_with_integrity(self: Any, cfg: Any, training_args: Any) -> Any:
        model = instantiate(cfg.model)
        if cfg.pretrained_model_path is not None:
            mprint(
                f"Strictly loading pretrained weights from: {cfg.pretrained_model_path}"
            )
            load_agibot_shards(
                model,
                cfg.pretrained_model_path,
                allowed_missing=agibot_allowed_missing(cfg.pretrained_model_path),
            )
            if (
                hasattr(model, "action_head")
                and hasattr(model.action_head, "inject_lora_after_loading")
                and model.action_head.config.defer_lora_injection
            ):
                model.action_head.inject_lora_after_loading()
        warm_start = os.environ.get("DREAMZERO_WARM_START_CHECKPOINT")
        if warm_start:
            # Config-portable restart across a world-size change: reload the
            # trained LoRA/head overlay, then let the trainer build a fresh
            # optimizer (the archived DeepSpeed shards cannot be resharded).
            from src.reconstruct import load_overlay_with_integrity

            mprint(f"Warm-starting trained overlay from: {warm_start}")
            load_overlay_with_integrity(model, Path(warm_start))
        model.config.resume_path = model.config._name_or_path = training_args.output_dir
        trainable = sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if not trainable:
            raise AssertionError(
                "model has no trainable parameters after LoRA injection"
            )
        exp_cfg = Path(training_args.output_dir) / "experiment_cfg"
        exp_cfg.mkdir(parents=True, exist_ok=True)
        (exp_cfg / TRAINED_PARAMETER_NAMES).write_text(
            json.dumps(trainable, indent=2) + "\n"
        )
        mprint(f"Recorded {len(trainable)} trainable parameter names")
        return model

    original_create_trainer = base_module.BaseExperiment.create_trainer

    def create_trainer_with_probes(self: Any, *args: Any, **kwargs: Any) -> Any:
        trainer = original_create_trainer(self, *args, **kwargs)
        stage = os.environ.get("DREAMZERO_STAGE")
        if stage:
            trainer.args.run_name = stage
        original_training_step = trainer.training_step

        def training_step(
            model: Any, inputs: Any, *step_args: Any, **step_kwargs: Any
        ) -> Any:
            if not hasattr(trainer, "_dreamzero_fixed_sample"):
                trainer._dreamzero_fixed_sample = _first_batch_item(inputs)
            return original_training_step(model, inputs, *step_args, **step_kwargs)

        trainer.training_step = training_step
        original_get_train_dataloader = trainer.get_train_dataloader

        def get_train_dataloader(*loader_args: Any, **loader_kwargs: Any) -> Any:
            return _TimedDataLoader(
                original_get_train_dataloader(*loader_args, **loader_kwargs), trainer
            )

        trainer.get_train_dataloader = get_train_dataloader
        trainer.add_callback(IntegrityCallback(trainer))
        trainer.add_callback(StageMetricsCallback(trainer))
        return trainer

    base_module.BaseExperiment.create_model = create_model_with_integrity
    base_module.BaseExperiment.create_trainer = create_trainer_with_probes
    # VLAExperiment inherits both methods, so patching the base before Hydra's
    # decorated main constructs it preserves the official command-line API.
    assert experiment_module.VLAExperiment.create_model is create_model_with_integrity


def main() -> None:
    _install_training_patches()
    from groot.vla.experiment.experiment import main as dreamzero_main

    dreamzero_main()


if __name__ == "__main__":
    main()
