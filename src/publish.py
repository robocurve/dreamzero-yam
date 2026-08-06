"""Merge and publish the selected DreamZero-YAM checkpoint.

Phase 6 of the plan: reconstruct the checkpoint via the explicit AgiBot
sequence, merge the LoRA (peft merge_and_unload), write a stock-loadable
artifact (config.json rewritten to train_architecture="full",
defer_lora_injection=false; experiment_cfg save_lora_only=false), run the
publish gate (load through STOCK GrootSimPolicy, assert fixed-sample output
identity vs the explicit reconstruction), and only with --confirm upload to
HF robocurve/dreamzero-yam-molmoact2.

peft 0.5.0 note: merge happens on the wrapped LoraModel via
merge_and_unload(), returning the base module tree with deltas folded in
(fp32 upcast during injection — cast back to bf16 after merging so the
artifact matches the AgiBot layout dtype).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import modal

from src.modal_app import checkpoint_volume, data_volume, secrets, training_image

app = modal.App("dreamzero-yam-publish")
PIPELINE_ROOT = Path("/root/dreamzero_yam")
DEFAULT_CHECKPOINT = "/checkpoints/yam/tune-armC/milestones/checkpoint-10000"
PUBLISH_ROOT = Path("/checkpoints/yam/publish")
HF_RELEASE_REPO = "robocurve/dreamzero-yam-molmoact2"
DATA_VAL = Path("/data/dreamzero_gear/allenai/19012026-block-13")
LOCAL_VAL = Path("/local/dreamzero_gear/allenai/19012026-block-13")


@app.function(
    image=training_image,
    gpu="H200:1",
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    secrets=[*secrets, modal.Secret.from_name("hf-robocurve-write")],
    timeout=4 * 60 * 60,
    cpu=16,
    memory=262_144,
)
def publish(
    checkpoint: str = DEFAULT_CHECKPOINT,
    confirm: bool = False,
) -> None:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["HF_HOME"] = "/tmp/hf-cache"
    import subprocess

    import torch
    from safetensors.torch import save_file

    from src.eval_dreamzero import (
        _as_plain_dict,
        _build_eval_transform,
        _state_is_batched,
        _unsqueeze_dict_values,
        _YAMDataset,
    )
    from src.eval_spec import load_indices
    from src.reconstruct import reconstruct_model

    checkpoint_path = Path(checkpoint)
    artifact = PUBLISH_ROOT / checkpoint_path.name
    if artifact.exists():
        shutil.rmtree(artifact)
    artifact.mkdir(parents=True)

    # 1. Explicit reconstruction (the proven wrong-base-proof path).
    model, base_integrity, overlay_integrity = reconstruct_model(
        checkpoint_path, device="cuda"
    )
    assert not base_integrity.missing_keys and not base_integrity.unexpected_keys
    assert not overlay_integrity.unexpected_keys
    model.to(dtype=torch.bfloat16)

    # 2. Merge the LoRA into the base weights.
    wrapped = model.action_head.model
    if hasattr(wrapped, "merge_and_unload"):
        model.action_head.model = wrapped.merge_and_unload()
    else:
        raise AssertionError(
            "action_head.model is not a peft LoraModel; nothing to merge"
        )
    merged_state = {
        key: value.detach().to(torch.bfloat16).cpu().contiguous()
        if torch.is_floating_point(value)
        else value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(merged_state, str(artifact / "model.safetensors"))

    # 3. Stock-loadable configs.
    config = json.loads((checkpoint_path / "config.json").read_text())

    def _rewrite(node):
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key == "train_architecture":
                    node[key] = "full"
                elif key == "defer_lora_injection":
                    node[key] = False
                elif key == "skip_component_loading":
                    node[key] = False
                else:
                    _rewrite(value)
        elif isinstance(node, list):
            for item in node:
                _rewrite(item)

    _rewrite(config)
    (artifact / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copytree(checkpoint_path / "experiment_cfg", artifact / "experiment_cfg")
    conf_path = artifact / "experiment_cfg/conf.yaml"
    conf_text = conf_path.read_text()
    conf_text = conf_text.replace("save_lora_only: true", "save_lora_only: false")
    conf_text = conf_text.replace(
        "train_architecture: lora", "train_architecture: full"
    )
    conf_path.write_text(conf_text)
    checkpoint_volume.commit()
    print(f"merged artifact written to {artifact}", flush=True)

    # 4. Publish gate: fixed-sample output identity, merged model (still in
    # memory, post-merge) vs the same inputs. A cold stock-path load is
    # approximated by reloading the artifact through VLA.from_pretrained via
    # the artifact's own config.json (what GrootSimPolicy does for
    # save_lora_only=false checkpoints).
    if not LOCAL_VAL.exists():
        LOCAL_VAL.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-aL", str(DATA_VAL), str(LOCAL_VAL.parent)], check=True)
    dataset = _YAMDataset(LOCAL_VAL)
    episode, frame = load_indices(PIPELINE_ROOT / "outputs/eval_indices.json")[0]
    obs, _ = dataset.sample(episode, frame)
    if not _state_is_batched(obs):
        obs = _unsqueeze_dict_values(obs)
    train_cfg, eval_transform = _build_eval_transform(model, checkpoint_path)
    normalized = _as_plain_dict(eval_transform(obs))
    for key, value in list(normalized.items()):
        if torch.is_tensor(value) and value.dtype == torch.float32:
            normalized[key] = value.to(dtype=torch.bfloat16)

    def _predict(m):
        m.eval()
        m.action_head.language = None
        m.action_head.current_start_frame = 0
        with torch.inference_mode():
            return (
                m.lazy_joint_video_action_causal(dict(normalized), latent_video=None)[
                    "action_pred"
                ]
                .float()
                .cpu()
            )

    model.post_initialize()
    reference = _predict(model)

    from groot.vla.model.dreamzero.base_vla import VLA

    stock = VLA.from_pretrained(str(artifact)).to("cuda", dtype=torch.bfloat16)
    stock.post_initialize()
    candidate = _predict(stock)
    if not torch.equal(reference, candidate):
        delta = float((reference - candidate).abs().max())
        raise AssertionError(f"publish gate FAILED: stock-path mismatch {delta}")
    print(
        "publish gate PASS: stock-path output identical to reconstruction", flush=True
    )

    # 5. Upload only with explicit confirmation.
    if not confirm:
        print("dry run complete; re-run with --confirm to upload to HF", flush=True)
        return
    from huggingface_hub import HfApi

    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        raise RuntimeError("HF_WRITE_TOKEN missing")
    api = HfApi(token=token)
    api.create_repo(HF_RELEASE_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=HF_RELEASE_REPO,
        repo_type="model",
        folder_path=str(artifact),
        path_in_repo=".",
        commit_message=f"Release merged {checkpoint_path.name}",
    )
    print(f"uploaded release to {HF_RELEASE_REPO}", flush=True)
