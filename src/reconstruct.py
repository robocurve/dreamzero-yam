"""Correct DreamZero AgiBot-base + YAM LoRA reconstruction and parity checks.

Never use ``VLA.load_lora`` for this project: it instantiates the frozen DiT
from vanilla Wan2.1. This module deliberately mirrors the training sequence.
"""

from __future__ import annotations

import copy
import json
import re
import types
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_VOLUME_ROOT = Path("/checkpoints")
AGIBOT_CKPT_DIR = CHECKPOINT_VOLUME_ROOT / "DreamZero-AgiBot"
WAN_CKPT_DIR = CHECKPOINT_VOLUME_ROOT / "Wan2.1-I2V-14B-480P"
UMT5_CKPT_DIR = CHECKPOINT_VOLUME_ROOT / "umt5-xxl"

WAN_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
AGIBOT_REPO_ID = "GEAR-Dreams/DreamZero-AgiBot"
UMT5_REPO_ID = "google/umt5-xxl"

TRAINED_PARAMETER_NAMES = "trained_parameter_names.json"
PARITY_REFERENCE_TENSORS = "parity_reference.safetensors"
PARITY_REFERENCE_NAMES = "parity_reference_names.json"
FIXED_SAMPLE_INPUT = "fixed_sample_input.pt"
FIXED_SAMPLE_OUTPUT = "fixed_sample_output.safetensors"
AGIBOT_COMPONENT_PREFIXES = "agibot_component_prefixes.json"
OPTIONAL_COMPONENT_PREFIXES = ("text_encoder", "image_encoder", "vae")


@dataclass(frozen=True)
class LoadIntegrity:
    loaded_keys: frozenset[str]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


@dataclass(frozen=True)
class OverlayIntegrity:
    checkpoint_keys: frozenset[str]
    trained_parameter_names: frozenset[str]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_path = checkpoint_dir / "model.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        filenames = sorted(set(index["weight_map"].values()))
        paths = [checkpoint_dir / name for name in filenames]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"checkpoint index references absent shards: {missing}"
            )
        return paths
    if single_path.is_file():
        return [single_path]
    raise FileNotFoundError(
        f"no model.safetensors or model.safetensors.index.json in {checkpoint_dir}"
    )


def record_agibot_component_prefixes(checkpoint_dir: str | Path) -> Path:
    """Persist which optional Wan component prefixes the AgiBot index supplies."""
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"AgiBot safetensors index is absent: {index_path}")
    index = json.loads(index_path.read_text())
    keys = index["weight_map"]
    present = sorted(
        prefix
        for prefix in OPTIONAL_COMPONENT_PREFIXES
        if any(re.search(rf"(?:^|\.){re.escape(prefix)}\.", key) for key in keys)
    )
    destination = checkpoint_dir / AGIBOT_COMPONENT_PREFIXES
    destination.write_text(
        json.dumps(
            {
                "source": index_path.name,
                "present_prefixes": present,
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def agibot_allowed_missing(checkpoint_dir: str | Path) -> Callable[[str], bool]:
    """Allow missing Wan components only when the AgiBot index omits them."""
    metadata_path = Path(checkpoint_dir) / AGIBOT_COMPONENT_PREFIXES
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"AgiBot component metadata is absent: {metadata_path}; rerun "
            "download_checkpoints"
        )
    payload = json.loads(metadata_path.read_text())
    present = set(payload["present_prefixes"])
    unknown = present - set(OPTIONAL_COMPONENT_PREFIXES)
    if unknown:
        raise ValueError(f"unknown AgiBot component prefixes: {sorted(unknown)}")
    absent = set(OPTIONAL_COMPONENT_PREFIXES) - present
    if not absent:
        return lambda _name: False
    pattern = re.compile(rf"(?:^|\.)(?:{'|'.join(sorted(map(re.escape, absent)))})\.")
    return lambda name: pattern.search(name) is not None


def load_checkpoint_state_dict(checkpoint_dir: str | Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    state: dict[str, Any] = {}
    for path in _checkpoint_files(checkpoint_dir):
        shard = load_file(str(path))
        overlap = state.keys() & shard.keys()
        if overlap:
            raise AssertionError(
                f"duplicate keys across checkpoint shards: {sorted(overlap)[:10]}"
            )
        state.update(shard)
    return state


def _action_head_inner(config_dict: dict[str, Any]) -> dict[str, Any]:
    action_head = config_dict["action_head_cfg"]
    inner = action_head.get("config", action_head)
    if not isinstance(inner, dict):
        raise TypeError("checkpoint action_head_cfg.config is not a mapping")
    return inner


def training_identical_config(config_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Return a config with stable component paths and deferred LoRA injection."""
    config = copy.deepcopy(dict(config_dict))
    inner = _action_head_inner(config)
    inner["defer_lora_injection"] = True
    inner["skip_component_loading"] = True
    inner["diffusion_model_cfg"]["diffusion_model_pretrained_path"] = str(WAN_CKPT_DIR)
    inner["text_encoder_cfg"]["text_encoder_pretrained_path"] = str(
        WAN_CKPT_DIR / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    inner["image_encoder_cfg"]["image_encoder_pretrained_path"] = str(
        WAN_CKPT_DIR / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    )
    inner["vae_cfg"]["vae_pretrained_path"] = str(WAN_CKPT_DIR / "Wan2.1_VAE.pth")
    return config


def load_agibot_shards(
    model: Any,
    checkpoint_dir: str | Path = AGIBOT_CKPT_DIR,
    allowed_missing: Callable[[str], bool] | None = None,
) -> LoadIntegrity:
    """Load all AgiBot shards and validate the aggregate key set.

    Checking ``missing_keys`` after each individual shard is invalid because
    every other shard is necessarily missing. This compares the union of loaded
    keys with the pre-injection model only after all shards have been applied.
    """
    import gc

    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    model_keys = set(model.state_dict())
    loaded: set[str] = set()
    unexpected: set[str] = set()
    for path in _checkpoint_files(checkpoint_dir):
        shard = load_file(str(path))
        duplicate = loaded & shard.keys()
        if duplicate:
            raise AssertionError(
                f"duplicate AgiBot shard keys: {sorted(duplicate)[:10]}"
            )
        result = model.load_state_dict(shard, strict=False)
        loaded.update(shard)
        unexpected.update(result.unexpected_keys)
        del shard
        gc.collect()

    missing = model_keys - loaded
    if allowed_missing is not None:
        missing = {name for name in missing if not allowed_missing(name)}
    integrity = LoadIntegrity(
        loaded_keys=frozenset(loaded),
        missing_keys=tuple(sorted(missing)),
        unexpected_keys=tuple(sorted(unexpected)),
    )
    if integrity.missing_keys or integrity.unexpected_keys:
        raise AssertionError(
            "AgiBot load integrity failed: "
            f"missing={list(integrity.missing_keys)[:30]} "
            f"unexpected={list(integrity.unexpected_keys)[:30]}"
        )
    return integrity


def _find_trained_parameter_file(checkpoint_dir: Path) -> Path:
    candidates = (
        checkpoint_dir / "experiment_cfg" / TRAINED_PARAMETER_NAMES,
        checkpoint_dir / TRAINED_PARAMETER_NAMES,
        checkpoint_dir.parent / "experiment_cfg" / TRAINED_PARAMETER_NAMES,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{TRAINED_PARAMETER_NAMES} is required for overlay parity; checked {candidates}"
    )


@contextmanager
def _capture_load_state_dict(model: Any) -> Iterator[list[Any]]:
    """Capture ``load_state_dict`` results used inside load_lora_weight()."""
    original = model.load_state_dict
    records: list[Any] = []

    def wrapped(
        _self: Any, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        result = original(state_dict, *args, **kwargs)
        records.append((frozenset(state_dict), result))
        return result

    model.load_state_dict = types.MethodType(wrapped, model)
    try:
        yield records
    finally:
        model.load_state_dict = original


def load_overlay_with_integrity(
    model: Any, checkpoint_dir: str | Path
) -> OverlayIntegrity:
    checkpoint_dir = Path(checkpoint_dir)
    raw_checkpoint_keys = frozenset(load_checkpoint_state_dict(checkpoint_dir))
    trained_path = _find_trained_parameter_file(checkpoint_dir)
    trained_names = frozenset(json.loads(trained_path.read_text()))
    if raw_checkpoint_keys != trained_names:
        raise AssertionError(
            "LoRA checkpoint keys differ from the training-time trainable set: "
            f"missing_from_checkpoint={sorted(trained_names - raw_checkpoint_keys)[:30]} "
            f"unrecorded_checkpoint_keys={sorted(raw_checkpoint_keys - trained_names)[:30]}"
        )

    with _capture_load_state_dict(model) as records:
        returned = model.load_lora_weight(str(checkpoint_dir))
    if returned is not None:
        raise AssertionError(
            "DreamZero load_lora_weight API unexpectedly started returning a value"
        )
    if len(records) != 1:
        raise AssertionError(
            f"expected one overlay load_state_dict call, captured {len(records)}"
        )
    _, result = records[0]
    integrity = OverlayIntegrity(
        checkpoint_keys=raw_checkpoint_keys,
        trained_parameter_names=trained_names,
        missing_keys=tuple(sorted(result.missing_keys)),
        unexpected_keys=tuple(sorted(result.unexpected_keys)),
    )
    # Missing keys are expected: this is a trainable-only state dict being
    # loaded into the full 14B model. Unexpected keys are never acceptable.
    if integrity.unexpected_keys:
        raise AssertionError(
            f"unexpected LoRA/head keys: {list(integrity.unexpected_keys)[:30]}"
        )
    return integrity


def reconstruct_model(
    checkpoint_dir: str | Path,
    agibot_dir: str | Path = AGIBOT_CKPT_DIR,
    device: str | None = None,
) -> tuple[Any, LoadIntegrity, OverlayIntegrity]:
    """Reconstruct exactly: VLA(deferred) -> AgiBot -> inject -> YAM overlay."""
    from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig

    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config is absent: {config_path}")
    config = VLAConfig(**training_identical_config(json.loads(config_path.read_text())))
    model = VLA(config)
    base_integrity = load_agibot_shards(
        model,
        agibot_dir,
        allowed_missing=agibot_allowed_missing(agibot_dir),
    )
    if not hasattr(model.action_head, "inject_lora_after_loading"):
        raise AttributeError("DreamZero action head has no inject_lora_after_loading()")
    model.action_head.inject_lora_after_loading()
    overlay_integrity = load_overlay_with_integrity(model, checkpoint_dir)
    model.eval()
    model.requires_grad_(False)
    if device is not None:
        model.to(device=device)
    return model, base_integrity, overlay_integrity


def assert_tensor_bit_identity(model: Any, reference_dir: str | Path) -> dict[str, str]:
    """Check one base_layer, one LoRA tensor, and one trained action-head tensor."""
    import torch
    from safetensors.torch import load_file

    reference_dir = Path(reference_dir)
    names = json.loads((reference_dir / PARITY_REFERENCE_NAMES).read_text())
    required = {"base_layer", "lora", "action_head"}
    if set(names) != required:
        raise AssertionError(f"parity reference must name {required}, got {set(names)}")
    reference = load_file(str(reference_dir / PARITY_REFERENCE_TENSORS))
    current = model.state_dict()
    for category, parameter_name in names.items():
        if parameter_name not in current:
            raise KeyError(
                f"parity tensor is absent after reconstruction: {parameter_name}"
            )
        expected = reference[category].cpu()
        actual = current[parameter_name].detach().cpu()
        if not torch.equal(actual, expected):
            raise AssertionError(
                f"{category} tensor is not bit-identical: {parameter_name}"
            )
    return names


def flatten_tensor_tree(value: Any, prefix: str = "output") -> dict[str, Any]:
    """Flatten tensor-bearing policy output containers for safetensors."""
    import torch

    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu().contiguous()}
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key in sorted(value):
            output.update(flatten_tensor_tree(value[key], f"{prefix}.{key}"))
        return output
    if hasattr(value, "items"):
        return flatten_tensor_tree(dict(value.items()), prefix)
    if isinstance(value, (tuple, list)):
        output = {}
        for index, item in enumerate(value):
            output.update(flatten_tensor_tree(item, f"{prefix}.{index}"))
        return output
    return {}


def _tree_to_device(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    return value


def fixed_sample_policy_call(model: Any, sample: Mapping[str, Any]) -> Any:
    """Call the policy heads on a collated training sample.

    Upstream ``VLA.get_action`` validates pre-collation NumPy inputs, while the
    parity artifact is deliberately captured after collation as tensors. This
    reproduces its post-validation preparation, including VLA's device/dtype
    tree cast, and exercises the same policy output without weakening the
    public input validation.
    """
    import torch
    import tree

    backbone_inputs = model.backbone.prepare_input(sample)
    action_inputs = model.action_head.prepare_input(sample)

    def to_device_with_maybe_dtype(value: Any) -> Any:
        if torch.is_floating_point(value):
            return value.to(model.device, dtype=model.action_head.dtype)
        return value.to(model.device)

    backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
    action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
    backbone_outputs = model.backbone(backbone_inputs)
    return model.action_head.get_action(backbone_outputs, action_inputs)


def assert_policy_output_bit_identity(
    model: Any,
    reference_dir: str | Path,
    policy_call: Callable[[Any, Any], Any] | None = None,
) -> None:
    """Re-run the fixed training sample and require bit-identical policy output."""
    import torch
    from safetensors.torch import load_file

    reference_dir = Path(reference_dir)
    sample_path = reference_dir / FIXED_SAMPLE_INPUT
    output_path = reference_dir / FIXED_SAMPLE_OUTPUT
    if not sample_path.is_file() or not output_path.is_file():
        raise FileNotFoundError(
            f"fixed-sample parity artifacts are absent: {sample_path}, {output_path}"
        )
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    sample = _tree_to_device(sample, model.device)
    devices = [model.device] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        if devices:
            torch.cuda.manual_seed_all(0)
        with torch.inference_mode():
            actual_output = (
                policy_call(model, sample)
                if policy_call
                else fixed_sample_policy_call(model, sample)
            )
    actual = flatten_tensor_tree(actual_output)
    expected = load_file(str(output_path))
    if set(actual) != set(expected):
        raise AssertionError(
            f"policy output keys differ: actual={sorted(actual)} expected={sorted(expected)}"
        )
    for key in actual:
        if not torch.equal(actual[key].cpu(), expected[key].cpu()):
            raise AssertionError(
                f"fixed-sample policy output is not bit-identical at {key}"
            )


def assert_eval_reconstruction_parity(model: Any, checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    assert_tensor_bit_identity(model, checkpoint_dir)
    assert_policy_output_bit_identity(model, checkpoint_dir)


__all__ = [
    "AGIBOT_COMPONENT_PREFIXES",
    "AGIBOT_CKPT_DIR",
    "WAN_CKPT_DIR",
    "UMT5_CKPT_DIR",
    "AGIBOT_REPO_ID",
    "WAN_REPO_ID",
    "UMT5_REPO_ID",
    "agibot_allowed_missing",
    "assert_eval_reconstruction_parity",
    "assert_policy_output_bit_identity",
    "assert_tensor_bit_identity",
    "flatten_tensor_tree",
    "fixed_sample_policy_call",
    "load_agibot_shards",
    "load_overlay_with_integrity",
    "reconstruct_model",
    "record_agibot_component_prefixes",
    "training_identical_config",
]
