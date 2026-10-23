"""Live WandB replay of the trainer's runtime metrics.

The training process's own WandB run drops post-resume points (inherited
internal step counter exceeds global_step). This CPU sidecar tails
runtime_metrics.jsonl from the checkpoint volume and replays every row into a
fresh run indexed by the true training step, giving a live, complete curve
without restarting the trainer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from src.image_security import secure_image
from src.modal_app import checkpoint_volume, secrets

app = modal.App("dreamzero-yam-loss-replay")
CHECKPOINT_ROOT = Path("/checkpoints/yam")
WANDB_PROJECT = "dreamzero-yam"

replay_image = (
    secure_image(modal.Image.debian_slim(python_version="3.11"))
    .pip_install("wandb")
    .add_local_dir(
        str(Path(__file__).resolve().parents[1]), "/root/dreamzero_yam", copy=True
    )
    .env({"PYTHONPATH": "/root/dreamzero_yam", "PYTHONUNBUFFERED": "1"})
)


@app.function(
    image=replay_image,
    volumes={"/checkpoints": checkpoint_volume},
    secrets=secrets,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=8, initial_delay=30.0),
    cpu=2,
)
def loss_replay(exp_name: str = "tune-armC", poll_seconds: int = 60) -> None:
    import wandb

    metrics_path = CHECKPOINT_ROOT / exp_name / "runtime_metrics.jsonl"
    run = wandb.init(
        project=WANDB_PROJECT,
        id=f"{exp_name}-train-live",
        name="train-live",
        group=exp_name,
        job_type="loss-replay",
        resume="allow",
    )
    # Index every series by the true training step so replay order and
    # WandB's internal counter are irrelevant.
    run.define_metric("step")
    run.define_metric("*", step_metric="step")
    last_step = int(run.summary.get("replay/last_step", 0) or 0)
    print(f"replaying from step {last_step}", flush=True)
    while True:
        checkpoint_volume.reload()
        if metrics_path.is_file():
            logged = 0
            for line in metrics_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = row.get("step")
                if not isinstance(step, int) or step <= last_step:
                    continue
                payload = {
                    key: value
                    for key, value in row.items()
                    if isinstance(value, (int, float)) and key != "time_monotonic"
                }
                payload["step"] = step
                run.log(payload)
                last_step = step
                logged += 1
            if logged:
                run.summary["replay/last_step"] = last_step
                print(f"replayed {logged} rows up to step {last_step}", flush=True)
        time.sleep(poll_seconds)
