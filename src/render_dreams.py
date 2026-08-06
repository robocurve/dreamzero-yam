"""Render imagined-vs-real videos from a DreamZero-YAM checkpoint.

For a handful of frozen eval indices, run the same single-chunk causal call
as the eval runner, VAE-decode the imagined latent video, and write
side-by-side MP4s (imagined | real) to /checkpoints/yam/renders/.

Upstream mapping (pinned commit ab790c1):
- video_pred is latent-space: wan_flow_matching_action_tf.py:1310-1338 returns
  the chunk's latents (B, C, T=3, H_lat, W_lat) including the conditioning
  frame.
- Decode: action_head.vae.decode(latents) -> B C T H W in [-1, 1]
  (eval_utils/serve_dreamzero_wan22.py:286-295); 3 latent frames -> 9 RGB
  frames (1 + 2x4 temporal upsampling).
- The model imagines a 2x2 grid of the three cameras
  (transform/dreamzero_cotrain.py:309-381) at 352x640.
- Time alignment: video delta indices stride 3 -> imagined frame i
  corresponds to dataset frame anchor + 3*i.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from src.modal_app import checkpoint_volume, data_volume, secrets, training_image

app = modal.App("dreamzero-yam-render")
PIPELINE_ROOT = Path("/root/dreamzero_yam")
DATA_VAL = Path("/data/dreamzero_gear/allenai/19012026-block-13")
LOCAL_VAL = Path("/local/dreamzero_gear/allenai/19012026-block-13")
RENDER_ROOT = Path("/checkpoints/yam/renders")
DEFAULT_CHECKPOINT = "/checkpoints/yam/tune-armC/milestones/checkpoint-10000"
IMAGINED_FRAMES = 9  # 1 conditioning + 2 latent frames x 4 temporal upsample
FRAME_STRIDE = 3  # dataset frames per imagined frame


def _label(frame, text):
    import cv2

    frame = frame.copy()
    cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(
        frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
    )
    return frame


def _camera_grid(top, left, right):
    """Replicate the training-time 2x2 packing: top spans the upper half."""
    import numpy as np

    height, width = top.shape[:2]
    canvas = np.zeros((height * 2, width, 3), dtype=top.dtype)
    canvas[:height] = top
    half = width // 2
    import cv2

    canvas[height:, :half] = cv2.resize(left, (half, height))
    canvas[height:, half:] = cv2.resize(right, (width - half, height))
    return canvas


@app.function(
    image=training_image,
    gpu="H200:1",
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    secrets=secrets,
    timeout=2 * 60 * 60,
    cpu=16,
    memory=131_072,
    ephemeral_disk=400_000,
)
def render(
    checkpoint: str = DEFAULT_CHECKPOINT,
    sample_count: int = 6,
    fps: int = 10,
) -> None:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["HF_HOME"] = "/tmp/hf-cache"
    import cv2
    import imageio
    import numpy as np
    import torch
    from einops import rearrange

    from src.eval_dreamzero import (
        _as_plain_dict,
        _build_eval_transform,
        _state_is_batched,
        _unsqueeze_dict_values,
        _YAMDataset,
    )
    from src.eval_spec import load_indices
    from src.reconstruct import reconstruct_model

    if not LOCAL_VAL.exists():
        LOCAL_VAL.parent.mkdir(parents=True, exist_ok=True)
        import subprocess

        subprocess.run(["cp", "-aL", str(DATA_VAL), str(LOCAL_VAL.parent)], check=True)

    checkpoint_path = Path(checkpoint)
    dataset = _YAMDataset(LOCAL_VAL)
    indices = load_indices(PIPELINE_ROOT / "outputs/eval_indices.json")[:sample_count]
    model, _, _ = reconstruct_model(checkpoint_path, device="cuda")
    model.to(dtype=torch.bfloat16)
    model.post_initialize()
    train_cfg, eval_transform = _build_eval_transform(model, checkpoint_path)
    model.eval()
    RENDER_ROOT.mkdir(parents=True, exist_ok=True)

    for episode, frame in indices:
        obs, _ = dataset.sample(episode, frame)
        if not _state_is_batched(obs):
            obs = _unsqueeze_dict_values(obs)
        normalized = _as_plain_dict(eval_transform(obs))
        if train_cfg.get("eval_bf16", False):
            for key, value in normalized.items():
                if torch.is_tensor(value) and value.dtype == torch.float32:
                    normalized[key] = value.to(dtype=torch.bfloat16)
        model.action_head.language = None
        model.action_head.current_start_frame = 0
        with torch.inference_mode():
            output = model.lazy_joint_video_action_causal(normalized, latent_video=None)
            latents = output["video_pred"]
            if latents.dim() == 4:
                latents = latents.unsqueeze(0)
            decoded = model.action_head.vae.decode(latents.to(torch.bfloat16))
        video = rearrange(decoded[0].float().cpu(), "c t h w -> t h w c").numpy()
        imagined = ((video + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        frames_out = []
        for i in range(min(IMAGINED_FRAMES, imagined.shape[0])):
            dataset_frame = frame + FRAME_STRIDE * i
            try:
                real_obs, _ = dataset.sample(episode, dataset_frame)
            except Exception:
                break
            real_grid = _camera_grid(
                *(
                    np.asarray(real_obs[f"video.{name}"])[0]
                    for name in (
                        "top_camera-images-rgb",
                        "left_camera-images-rgb",
                        "right_camera-images-rgb",
                    )
                )
            )
            dream = imagined[i]
            if real_grid.shape[:2] != dream.shape[:2]:
                real_grid = cv2.resize(real_grid, (dream.shape[1], dream.shape[0]))
            combined = np.concatenate(
                [
                    _label(dream, f"imagined t+{FRAME_STRIDE * i}"),
                    _label(real_grid, f"real t+{FRAME_STRIDE * i}"),
                ],
                axis=1,
            )
            frames_out.append(combined)

        out_path = RENDER_ROOT / f"dream-ep{episode}-f{frame}.mp4"
        imageio.mimwrite(str(out_path), frames_out, fps=fps, quality=8)
        print(f"rendered {out_path} ({len(frames_out)} frames)", flush=True)
    checkpoint_volume.commit()
    manifest = RENDER_ROOT / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"checkpoint": str(checkpoint_path), "indices": indices[:sample_count]},
            indent=2,
        )
    )
    checkpoint_volume.commit()
