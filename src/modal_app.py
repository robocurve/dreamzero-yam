"""Modal scaffolding for DreamZero-YAM data-independent checkpoint and training stages."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import modal

from src.gpu_metrics import parse_gpu_spec, throughput_metrics
from src.reconstruct import (
    AGIBOT_CKPT_DIR,
    AGIBOT_REPO_ID,
    PARITY_REFERENCE_NAMES,
    PARITY_REFERENCE_TENSORS,
    UMT5_CKPT_DIR,
    UMT5_REPO_ID,
    WAN_CKPT_DIR,
    WAN_REPO_ID,
    record_agibot_component_prefixes,
)

DREAMZERO_REF = "ab790c198fbce33503358efbbd4187ce9a89adf3"
LOCAL_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = Path("/root/dreamzero_yam")
DREAMZERO_ROOT = Path("/root/dreamzero")
DATA_VOLUME_ROOT = Path("/data/dreamzero_gear")
LOCAL_DATA_ROOT = Path("/local/dreamzero_gear")
CHECKPOINT_ROOT = Path("/checkpoints/yam")
REPOS_FILE = PIPELINE_ROOT / "configs/repos.json"
DATA_CONFIG_SOURCE = PIPELINE_ROOT / "configs/yam_dataset.yaml"
DATA_CONFIG_TARGET = (
    DREAMZERO_ROOT / "groot/vla/configs/data/dreamzero/yam_dataset.yaml"
)
COMMIT_INTERVAL_SECONDS = 150
MILESTONE_INTERVAL_STEPS = 2_500
# Config-portable restarts keep the training recipe identical across GPU
# allocations: the global batch is pinned and per-device batch must divide it.
GLOBAL_BATCH_SIZE = 32
# Young-Daly inputs: the ~2 GB LoRA+head payload plus DeepSpeed shards take
# about a minute to write, and the Modal 24 h function cap bounds the mean
# time between interruptions even when spot preemptions are rare.
CHECKPOINT_WRITE_SECONDS = 60.0
INTERRUPTION_MTBF_SECONDS = 24 * 60 * 60.0
# Plan Phase 3/4 research estimate (~0.3 samples/s/GPU on 8 GPUs); the
# benchmark stage replaces this with a measured value.
ESTIMATED_AGGREGATE_SAMPLES_PER_SECOND = 2.4
DEFAULT_GPU_SPEC = "H100:8"
WANDB_PROJECT = "dreamzero-yam"


app = modal.App("dreamzero-yam")

training_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git",
        "git-lfs",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "build-essential",
        "ninja-build",
        "nvtop",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu129 "
        "torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0",
        f"git clone https://github.com/dreamzero0/dreamzero.git {DREAMZERO_ROOT}",
        f"cd {DREAMZERO_ROOT} && git checkout {DREAMZERO_REF}",
        # --no-deps: the full upstream tree drags in sim/teleop/serving deps
        # (ray, mujoco, evdev — which fails to build — PyQt6, tensorrt, ...).
        # Training-path deps are curated below from upstream pyproject +
        # import scan of groot/vla (ftfy is the only borderline import).
        f"python -m pip install --no-deps -e {DREAMZERO_ROOT}",
        "python -m pip install "
        "av==15.0.0 scipy==1.15.3 numpy==1.26.4 matplotlib hydra-core "
        "loguru peft==0.5.0 datasets==3.6.0 pandas dm_tree "
        "transformers==4.51.3 albumentations==1.4.18 einops==0.8.1 "
        "imageio==2.34.2 imageio-ffmpeg wandb opencv-python-headless==4.9.0.80 "
        "diffusers==0.30.2 ftfy huggingface_hub decord2 deepspeed "
        "tiktoken sentencepiece termcolor click tyro timm safetensors "
        "nvidia-ml-py pynvml h5py gymnasium",
        "MAX_JOBS=8 python -m pip install flash-attn --no-build-isolation",
    )
    .add_local_dir(str(LOCAL_ROOT), str(PIPELINE_ROOT), copy=True)
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": "/checkpoints/hf-cache",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONPATH": f"{PIPELINE_ROOT}:{DREAMZERO_ROOT}",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub", "wandb", "safetensors")
    # This module does `from src....` at import time, so every image that
    # executes a function defined here needs the repo source + PYTHONPATH.
    .add_local_dir(str(LOCAL_ROOT), str(PIPELINE_ROOT), copy=True)
    .env({"PYTHONPATH": str(PIPELINE_ROOT), "PYTHONUNBUFFERED": "1"})
)

data_volume = modal.Volume.from_name("gr00t-yam-data", create_if_missing=False)
checkpoint_volume = modal.Volume.from_name("dreamzero-yam-ckpt", create_if_missing=True)
secrets = [
    modal.Secret.from_name("huggingface-token"),
    modal.Secret.from_name("wandb"),
]


def _read_repos() -> dict:
    repos = json.loads(REPOS_FILE.read_text())
    assert len(repos["train"]) == 124
    assert repos["val"] not in repos["train"]
    return repos


def _copy_repo_to_nvme(repo_id: str) -> Path:
    source = DATA_VOLUME_ROOT / repo_id
    target = LOCAL_DATA_ROOT / repo_id
    if not source.is_dir():
        raise FileNotFoundError(f"prepared DreamZero dataset is absent: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        # -L follows the hardlink/symlink sharing layer on the FUSE volume. The
        # resulting local tree has ordinary files and all random video reads hit
        # container NVMe during torchrun. Copy to a temp name and rename so a
        # mid-copy failure on a warm container can never be mistaken for a
        # complete repo by the exists() skip above.
        staging = target.parent / f".{target.name}.copying"
        if staging.exists():
            shutil.rmtree(staging)
        subprocess.run(["cp", "-aL", str(source), str(staging)], check=True)
        staging.rename(target)
    return target


def _copy_training_data_to_nvme(
    include_val: bool = False, repo_ids: list[str] | None = None
) -> None:
    repos = _read_repos()
    selected = list(repo_ids if repo_ids is not None else repos["train"])
    if include_val:
        selected.append(repos["val"])
    started = time.monotonic()
    for index, repo_id in enumerate(selected, 1):
        print(f"NVMe copy [{index}/{len(selected)}] {repo_id}", flush=True)
        _copy_repo_to_nvme(repo_id)
    print(
        f"NVMe copy complete in {(time.monotonic() - started) / 60:.1f} min", flush=True
    )


def _install_data_config(repo_ids: list[str] | None = None) -> None:
    DATA_CONFIG_TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA_CONFIG_SOURCE, DATA_CONFIG_TARGET)
    if repo_ids is not None:
        from omegaconf import OmegaConf

        config = OmegaConf.load(DATA_CONFIG_TARGET)
        config.train_dataset.mixture_spec[0].dataset_path.yam = [
            str(LOCAL_DATA_ROOT / repo_id) for repo_id in repo_ids
        ]
        OmegaConf.save(config, DATA_CONFIG_TARGET)
        # OmegaConf drops comments, including the `# @package _global_` Hydra
        # directive on line 1 — without it the config lands under the group
        # package and upstream conf.yaml's top-level train_dataset stays ???.
        body = DATA_CONFIG_TARGET.read_text()
        if not body.startswith("# @package _global_"):
            DATA_CONFIG_TARGET.write_text("# @package _global_\n" + body)


def young_daly_save_steps(
    seconds_per_step: float,
    checkpoint_write_seconds: float = CHECKPOINT_WRITE_SECONDS,
    mtbf_seconds: float = INTERRUPTION_MTBF_SECONDS,
) -> int:
    """Derive save_steps from the Young-Daly optimum ``sqrt(2 * delta * MTBF)``.

    The optimum is snapped to the divisor of ``MILESTONE_INTERVAL_STEPS``
    nearest in log space so milestone steps always land on a saved checkpoint.
    """
    optimum_steps = math.sqrt(2.0 * checkpoint_write_seconds * mtbf_seconds) / max(
        seconds_per_step, 1e-9
    )
    divisors = [
        value
        for value in range(1, MILESTONE_INTERVAL_STEPS + 1)
        if MILESTONE_INTERVAL_STEPS % value == 0
    ]
    return min(divisors, key=lambda value: abs(math.log(value / optimum_steps)))


def _checkpoint_step(checkpoint: Path) -> int:
    return int(checkpoint.name.rsplit("-", 1)[1])


def _checkpoint_world_size(checkpoint: Path) -> int:
    state_dirs = [path for path in checkpoint.glob("global_step*") if path.is_dir()]
    return len([path for root in state_dirs for path in root.glob("*optim_states.pt")])


def _checkpoint_complete(checkpoint: Path, expected_world_size: int | None) -> bool:
    try:
        required = (
            checkpoint / "trainer_state.json",
            checkpoint / "config.json",
            checkpoint / "experiment_cfg/conf.yaml",
            checkpoint / "experiment_cfg/metadata.json",
            checkpoint / "experiment_cfg/trained_parameter_names.json",
            checkpoint / PARITY_REFERENCE_NAMES,
            checkpoint / PARITY_REFERENCE_TENSORS,
        )
        if not all(path.is_file() for path in required):
            return False
        if not (
            (checkpoint / "model.safetensors").is_file()
            or (checkpoint / "model.safetensors.index.json").is_file()
        ):
            return False
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if int(state["global_step"]) != _checkpoint_step(checkpoint):
            return False
        state_dirs = [path for path in checkpoint.glob("global_step*") if path.is_dir()]
        optimizer = [
            path for root in state_dirs for path in root.glob("*optim_states.pt")
        ]
        model_state = [
            path for root in state_dirs for path in root.glob("*model_states.pt")
        ]
        if expected_world_size is None:
            # Self-consistency for resume discovery: a checkpoint written by a
            # different world size is still complete; the launcher decides
            # whether its DeepSpeed shards are loadable on this allocation.
            if not optimizer or len(model_state) != len(optimizer):
                return False
        elif (
            len(optimizer) < expected_world_size
            or len(model_state) < expected_world_size
        ):
            return False
        probe_world_size = expected_world_size or len(optimizer)
        if not all(
            (checkpoint / f"resume_probe_rank_{rank}.json").is_file()
            for rank in range(probe_world_size)
        ):
            return False
        if (checkpoint / "model.safetensors.index.json").is_file():
            index = json.loads(
                (checkpoint / "model.safetensors.index.json").read_text()
            )
            if not all(
                (checkpoint / name).is_file()
                for name in set(index["weight_map"].values())
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _complete_checkpoints(output: Path, expected_world_size: int | None) -> list[Path]:
    checkpoints = [
        path
        for path in output.glob("checkpoint-*")
        if path.is_dir() and _checkpoint_complete(path, expected_world_size)
    ]
    return sorted(checkpoints, key=_checkpoint_step)


def _prune_incomplete_checkpoints(
    output: Path, expected_world_size: int | None
) -> None:
    if not output.exists():
        return
    removed = False
    for checkpoint in output.glob("checkpoint-*"):
        if checkpoint.is_dir() and not _checkpoint_complete(
            checkpoint, expected_world_size
        ):
            print(f"pruning incomplete checkpoint {checkpoint}", flush=True)
            shutil.rmtree(checkpoint)
            removed = True
    if removed:
        checkpoint_volume.commit()


def _copy_milestones(
    output: Path, expected_world_size: int, interval_steps: int
) -> None:
    milestone_root = output / "milestones"
    for checkpoint in _complete_checkpoints(output, expected_world_size):
        step = _checkpoint_step(checkpoint)
        if step % interval_steps:
            continue
        destination = milestone_root / checkpoint.name
        if destination.exists():
            continue
        print(f"preserving milestone checkpoint {step}", flush=True)
        milestone_root.mkdir(parents=True, exist_ok=True)
        temporary = milestone_root / f".{checkpoint.name}.partial"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(checkpoint, temporary)
        temporary.rename(destination)


def _run_with_checkpoint_commits(
    command: list[str],
    output: Path,
    expected_world_size: int,
    process_env: dict[str, str],
    milestone_interval_steps: int,
    planned_timeout_seconds: float | None = None,
) -> bool:
    """Run torchrun with 150-second volume commits and planned-stop support."""
    process = subprocess.Popen(
        command,
        cwd=DREAMZERO_ROOT,
        env=process_env,
        start_new_session=True,
    )
    stop = threading.Event()

    def commit_loop() -> None:
        while not stop.wait(COMMIT_INTERVAL_SECONDS):
            try:
                _copy_milestones(output, expected_world_size, milestone_interval_steps)
                checkpoint_volume.commit()
            except Exception as exc:
                print(f"[checkpoint committer] {type(exc).__name__}: {exc}", flush=True)

    thread = threading.Thread(target=commit_loop, daemon=True)
    thread.start()
    planned_stop = False
    try:
        try:
            return_code = process.wait(timeout=planned_timeout_seconds)
        except subprocess.TimeoutExpired:
            planned_stop = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()
        if not planned_stop and return_code:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        stop.set()
        thread.join(timeout=10)
        _copy_milestones(output, expected_world_size, milestone_interval_steps)
        checkpoint_volume.commit()
    return planned_stop


def _training_command(
    output: Path,
    num_gpus: int,
    max_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    per_device_batch_size: int,
    lora_rank: int,
    lora_alpha: int,
    save_steps: int,
    dataloader_num_workers: int,
) -> list[str]:
    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        str(PIPELINE_ROOT / "src/train_entry.py"),
        "report_to=wandb",
        "data=dreamzero/yam_dataset",
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
        f"training_args.learning_rate={learning_rate}",
        "training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json",
        f"training_args.warmup_ratio={warmup_ratio}",
        f"output_dir={output}",
        f"per_device_train_batch_size={per_device_batch_size}",
        f"global_batch_size={per_device_batch_size * num_gpus}",
        f"max_steps={max_steps}",
        "weight_decay=1e-5",
        "save_total_limit=5",
        "upload_checkpoints=false",
        "bf16=true",
        "tf32=true",
        "eval_bf16=true",
        "dataloader_pin_memory=false",
        f"dataloader_num_workers={dataloader_num_workers}",
        "image_resolution_width=320",
        "image_resolution_height=176",
        "save_lora_only=true",
        "max_chunk_size=4",
        "frame_seqlen=880",
        "save_strategy=steps",
        f"save_steps={save_steps}",
        "logging_steps=1",
        f"dit_version={WAN_CKPT_DIR}",
        f"text_encoder_pretrained_path={WAN_CKPT_DIR / 'models_t5_umt5-xxl-enc-bf16.pth'}",
        f"image_encoder_pretrained_path={WAN_CKPT_DIR / 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'}",
        f"vae_pretrained_path={WAN_CKPT_DIR / 'Wan2.1_VAE.pth'}",
        f"tokenizer_path={UMT5_CKPT_DIR}",
        f"pretrained_model_path={AGIBOT_CKPT_DIR}",
        "++action_head_cfg.config.skip_component_loading=true",
        "++action_head_cfg.config.defer_lora_injection=true",
        f"action_head_cfg.config.lora_rank={lora_rank}",
        f"action_head_cfg.config.lora_alpha={lora_alpha}",
    ]


def _train_impl(
    *,
    stage: str,
    gpu_spec: str,
    exp_name: str,
    max_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    per_device_batch_size: int,
    lora_rank: int,
    lora_alpha: int,
    save_steps: int,
    dataloader_num_workers: int,
    fresh: bool,
    fresh_token: str = "",
    allow_fresh_optimizer: bool = False,
    planned_timeout_seconds: float | None = None,
    prepare_data: bool = True,
) -> dict[str, float | str | int]:
    _, num_gpus = parse_gpu_spec(gpu_spec)
    if per_device_batch_size * num_gpus != GLOBAL_BATCH_SIZE:
        raise ValueError(
            f"stage {stage!r} must keep the global batch at {GLOBAL_BATCH_SIZE} "
            f"for config-portable restarts; got {per_device_batch_size} x "
            f"{num_gpus} GPUs. Set per_device_batch_size to "
            f"{GLOBAL_BATCH_SIZE} // num_gpus."
        )
    if save_steps <= 0:
        save_steps = young_daly_save_steps(
            seconds_per_step=(
                per_device_batch_size
                * num_gpus
                / ESTIMATED_AGGREGATE_SAMPLES_PER_SECOND
            )
        )
        print(f"Young-Daly derived save_steps={save_steps}", flush=True)
    milestone_interval_steps = (
        save_steps if stage.startswith("tune") else MILESTONE_INTERVAL_STEPS
    )
    output = CHECKPOINT_ROOT / exp_name
    fresh_marker = output / ".fresh_launch_token"
    initialized_token = (
        fresh_marker.read_text().strip() if fresh_marker.is_file() else ""
    )
    if fresh and initialized_token != fresh_token:
        if not fresh_token:
            raise ValueError("fresh launches require a stable non-empty fresh_token")
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        fresh_marker.write_text(fresh_token + "\n")
        checkpoint_volume.commit()
    output.mkdir(parents=True, exist_ok=True)
    _prune_incomplete_checkpoints(output, expected_world_size=None)
    checkpoints = _complete_checkpoints(output, expected_world_size=None)
    warm_start_checkpoint = ""
    if checkpoints:
        latest = checkpoints[-1]
        checkpoint_world_size = _checkpoint_world_size(latest)
        if checkpoint_world_size == num_gpus:
            print(f"auto-resume source: {latest}", flush=True)
        elif not allow_fresh_optimizer:
            raise ValueError(
                f"latest checkpoint {latest.name} carries DeepSpeed state for "
                f"{checkpoint_world_size} ranks but this launch has {num_gpus} "
                "GPUs. Relaunch on the original world size, or pass "
                "--allow-fresh-optimizer to archive the old lineage and "
                "warm-start the trained weights with a fresh optimizer "
                "(resets the step counter)."
            )
        else:
            archive = output / (
                f"world_size_{checkpoint_world_size}_archive_{int(time.time())}"
            )
            archive.mkdir(parents=True, exist_ok=True)
            for entry in sorted(output.glob("checkpoint-*")):
                entry.rename(archive / entry.name)
            milestones = output / "milestones"
            if milestones.is_dir():
                milestones.rename(archive / "milestones")
            warm_start_checkpoint = str(archive / latest.name)
            print(
                "world-size change accepted: archived old lineage to "
                f"{archive}; warm-starting overlay weights from "
                f"{warm_start_checkpoint} with a fresh optimizer",
                flush=True,
            )
            checkpoint_volume.commit()
    if prepare_data:
        _install_data_config()
        _copy_training_data_to_nvme(include_val=False)
    command = _training_command(
        output=output,
        num_gpus=num_gpus,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        per_device_batch_size=per_device_batch_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        save_steps=save_steps,
        dataloader_num_workers=dataloader_num_workers,
    )
    runtime_metrics = output / "runtime_metrics.jsonl"
    process_env = {
        **os.environ,
        "DREAMZERO_GLOBAL_BATCH": str(per_device_batch_size * num_gpus),
        "DREAMZERO_STAGE": stage,
        "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_NAME": stage,
        "WANDB_RUN_GROUP": stage,
        "WANDB_RUN_ID": f"{exp_name}-{stage}",
        "WANDB_RESUME": "allow",
    }
    if warm_start_checkpoint:
        process_env["DREAMZERO_WARM_START_CHECKPOINT"] = warm_start_checkpoint
    telemetry = subprocess.Popen(
        [
            sys.executable,
            str(PIPELINE_ROOT / "src/gpu_metrics.py"),
            "--stage",
            stage,
            "--gpu-spec",
            gpu_spec,
            "--runtime-metrics",
            str(runtime_metrics),
        ],
        cwd=PIPELINE_ROOT,
        env=process_env,
    )
    try:
        _run_with_checkpoint_commits(
            command,
            output,
            expected_world_size=num_gpus,
            process_env=process_env,
            milestone_interval_steps=milestone_interval_steps,
            planned_timeout_seconds=planned_timeout_seconds,
        )
    finally:
        telemetry.terminate()
        try:
            telemetry.wait(timeout=20)
        except subprocess.TimeoutExpired:
            telemetry.kill()
            telemetry.wait()
    rates = []
    if runtime_metrics.is_file():
        for line in runtime_metrics.read_text().splitlines():
            row = json.loads(line)
            rate = float(row.get("samples_per_second", 0.0))
            if rate > 0:
                rates.append(rate)
    samples_per_second = statistics.median(rates[10:] or rates) if rates else 0.0
    return {
        "stage": stage,
        "gpu_spec": gpu_spec,
        "num_gpus": num_gpus,
        "dataloader_num_workers": dataloader_num_workers,
        **throughput_metrics(samples_per_second, gpu_spec),
    }


@app.function(
    image=download_image,
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=24 * 60 * 60,
    cpu=8,
    memory=32_768,
)
def download_checkpoints() -> None:
    from huggingface_hub import snapshot_download

    for repo_id, destination in (
        (AGIBOT_REPO_ID, AGIBOT_CKPT_DIR),
        (WAN_REPO_ID, WAN_CKPT_DIR),
        (UMT5_REPO_ID, UMT5_CKPT_DIR),
    ):
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=destination)
        if repo_id == AGIBOT_REPO_ID:
            metadata_path = record_agibot_component_prefixes(destination)
            print(f"recorded AgiBot component prefixes in {metadata_path}", flush=True)
        print(f"downloaded {repo_id} -> {destination}", flush=True)
        checkpoint_volume.commit()


@app.function(
    image=training_image,
    gpu="H100:2",
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    secrets=secrets,
    timeout=8 * 60 * 60,
    cpu=16,
    memory=131_072,
    ephemeral_disk=1_200_000,
)
def smoke_test() -> None:
    repos = _read_repos()
    local_train = _copy_repo_to_nvme(sorted(repos["train"])[0])
    output = CHECKPOINT_ROOT / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    process_env = {
        **os.environ,
        "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_NAME": "smoke",
        "WANDB_RUN_GROUP": "smoke",
    }
    telemetry = subprocess.Popen(
        [
            sys.executable,
            str(PIPELINE_ROOT / "src/gpu_metrics.py"),
            "--stage",
            "smoke",
            "--gpu-spec",
            "H100:2",
            "--runtime-metrics",
            str(output / "runtime_metrics.jsonl"),
        ],
        cwd=PIPELINE_ROOT,
        env=process_env,
    )
    try:
        subprocess.run(
            [
                sys.executable,
                str(PIPELINE_ROOT / "src/smoke.py"),
                "--dataset",
                str(local_train),
                "--output",
                str(output),
                "--num-gpus",
                "2",
            ],
            cwd=PIPELINE_ROOT,
            env=process_env,
            check=True,
        )
    finally:
        telemetry.terminate()
        try:
            telemetry.wait(timeout=20)
        except subprocess.TimeoutExpired:
            telemetry.kill()
            telemetry.wait()
    checkpoint_volume.commit()


@app.function(
    image=training_image,
    gpu="H100:1",
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    secrets=secrets,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=8, initial_delay=60.0),
    cpu=32,
    memory=262_144,
    ephemeral_disk=1_200_000,
)
def _gpu_stage_worker(
    stage: str,
    gpu_spec: str,
    exp_name: str,
    max_steps: int = 30_000,
    learning_rate: float = 1e-5,
    warmup_ratio: float = 0.01,
    per_device_batch_size: int = 4,
    lora_rank: int = 4,
    lora_alpha: int = 4,
    save_steps: int = 0,
    dataloader_num_workers: int = 8,
    fresh: bool = False,
    fresh_token: str = "",
    allow_fresh_optimizer: bool = False,
    planned_timeout_seconds: float | None = None,
) -> dict[str, float | str | int]:
    return _train_impl(
        stage=stage,
        gpu_spec=gpu_spec,
        exp_name=exp_name,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        per_device_batch_size=per_device_batch_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        save_steps=save_steps,
        dataloader_num_workers=dataloader_num_workers,
        fresh=fresh,
        fresh_token=fresh_token,
        allow_fresh_optimizer=allow_fresh_optimizer,
        planned_timeout_seconds=planned_timeout_seconds,
    )


@app.function(
    image=training_image,
    gpu="H100:1",
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    secrets=secrets,
    timeout=24 * 60 * 60,
    cpu=32,
    memory=262_144,
    ephemeral_disk=1_200_000,
    # A preempted matrix cell restarts alone instead of aborting the sweep.
    retries=modal.Retries(max_retries=2, initial_delay=60.0),
)
def _benchmark_worker(
    gpu_spec: str,
    container_cpus: int,
    dataloader_workers: list[int],
    steps: int,
) -> list[dict[str, float | str | int]]:
    architecture, num_gpus = parse_gpu_spec(gpu_spec)
    if GLOBAL_BATCH_SIZE % num_gpus:
        raise ValueError(
            f"benchmark GPU count {num_gpus} must divide global batch "
            f"{GLOBAL_BATCH_SIZE}"
        )
    per_device_batch_size = GLOBAL_BATCH_SIZE // num_gpus
    # A fixed first-10 sorted train subset keeps every matrix cell comparable
    # while limiting the one-time FUSE-to-NVMe copy in each GPU/CPU container.
    benchmark_repos = sorted(_read_repos()["train"])[:10]
    _install_data_config(benchmark_repos)
    _copy_training_data_to_nvme(repo_ids=benchmark_repos)

    rows = []
    for workers in dataloader_workers:
        stage = (
            f"benchmark-{architecture.lower()}x{num_gpus}-w{workers}-c{container_cpus}"
        )
        row = _train_impl(
            stage=stage,
            gpu_spec=gpu_spec,
            exp_name=stage,
            max_steps=steps,
            learning_rate=1e-5,
            warmup_ratio=0.01,
            per_device_batch_size=per_device_batch_size,
            lora_rank=4,
            lora_alpha=4,
            save_steps=steps + 1,
            dataloader_num_workers=workers,
            fresh=True,
            fresh_token=uuid.uuid4().hex,
            prepare_data=False,
        )
        row["container_cpus"] = container_cpus
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


@app.function(
    image=download_image,
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60,
)
def _report_benchmark(rows: list[dict[str, float | str | int]], steps: int) -> None:
    import wandb

    columns = [
        "gpu_spec",
        "container_cpus",
        "dataloader_workers",
        "samples_per_second",
        "model_tflops_per_second_per_gpu",
        "mfu",
        "cost_per_1k_samples_usd",
    ]
    table_rows = [
        [
            row["gpu_spec"],
            row["container_cpus"],
            row["dataloader_num_workers"],
            row["perf/samples_per_second"],
            row["perf/model_tflops_per_second_per_gpu"],
            row["perf/mfu"],
            row["perf/cost_per_1k_samples_usd"],
        ]
        for row in rows
    ]
    run = wandb.init(
        project=WANDB_PROJECT,
        name="benchmark",
        job_type="benchmark-summary",
        config={"benchmark_steps": steps},
    )
    run.log({"benchmark/results": wandb.Table(columns=columns, data=table_rows)})
    run.finish()
    print(json.dumps(rows, indent=2), flush=True)


def _arm_config(arm: str) -> dict[str, str | float | int]:
    normalized = arm.lower().removeprefix("arm")
    if normalized not in {"a", "b", "c"}:
        raise ValueError(f"arm must be A, B, or C, got {arm!r}")
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/arms"
        / f"arm{normalized.upper()}.json"
    )
    return json.loads(path.read_text())


@app.local_entrypoint()
def train(
    gpu_spec: str = DEFAULT_GPU_SPEC,
    container_cpus: int = 32,
    exp_name: str = "tune-armA",
    max_steps: int = 30_000,
    learning_rate: float = 1e-5,
    warmup_ratio: float = 0.01,
    per_device_batch_size: int = 4,
    lora_rank: int = 4,
    lora_alpha: int = 4,
    save_steps: int = 0,
    dataloader_num_workers: int = 8,
    fresh: bool = False,
    allow_fresh_optimizer: bool = False,
) -> None:
    parse_gpu_spec(gpu_spec)
    stage = "train"
    _gpu_stage_worker.with_options(gpu=gpu_spec, cpu=container_cpus).remote(
        stage=stage,
        gpu_spec=gpu_spec,
        exp_name=exp_name,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        per_device_batch_size=per_device_batch_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        save_steps=save_steps,
        dataloader_num_workers=dataloader_num_workers,
        fresh=fresh,
        fresh_token=uuid.uuid4().hex if fresh else "",
        allow_fresh_optimizer=allow_fresh_optimizer,
    )


@app.local_entrypoint()
def tune(
    arm: str = "A",
    gpu_spec: str = DEFAULT_GPU_SPEC,
    container_cpus: int = 32,
    trial_minutes: int = 120,
    max_steps: int = 30_000,
    per_device_batch_size: int = 4,
    save_steps: int = 100,
    dataloader_num_workers: int = 8,
    fresh: bool = False,
    allow_fresh_optimizer: bool = False,
) -> None:
    parse_gpu_spec(gpu_spec)
    choice = _arm_config(arm)
    stage = str(choice["stage"])
    _gpu_stage_worker.with_options(gpu=gpu_spec, cpu=container_cpus).remote(
        stage=stage,
        gpu_spec=gpu_spec,
        exp_name=stage,
        max_steps=max_steps,
        learning_rate=float(choice["learning_rate"]),
        warmup_ratio=float(choice["warmup_ratio"]),
        per_device_batch_size=per_device_batch_size,
        lora_rank=int(choice["lora_rank"]),
        lora_alpha=int(choice["lora_alpha"]),
        save_steps=save_steps,
        dataloader_num_workers=dataloader_num_workers,
        fresh=fresh,
        fresh_token=uuid.uuid4().hex if fresh else "",
        allow_fresh_optimizer=allow_fresh_optimizer,
        planned_timeout_seconds=trial_minutes * 60,
    )


@app.local_entrypoint()
def benchmark(
    gpu_specs: str = "H100:8,H100:4,B200:8,H200:8",
    dataloader_workers: str = "1,8",
    container_cpus: str = "16,32",
    steps: int = 100,
) -> None:
    specs = [item.strip().upper() for item in gpu_specs.split(",") if item.strip()]
    worker_values = [int(item) for item in dataloader_workers.split(",")]
    cpu_values = [int(item) for item in container_cpus.split(",")]
    rows: list[dict[str, float | str | int]] = []
    for gpu_spec in specs:
        _, count = parse_gpu_spec(gpu_spec)
        if GLOBAL_BATCH_SIZE % count:
            raise ValueError(
                f"benchmark GPU count {count} must divide global batch "
                f"{GLOBAL_BATCH_SIZE}"
            )
        for cpu_count in cpu_values:
            container_rows = _benchmark_worker.with_options(
                gpu=gpu_spec, cpu=cpu_count
            ).remote(
                gpu_spec=gpu_spec,
                container_cpus=cpu_count,
                dataloader_workers=worker_values,
                steps=steps,
            )
            rows.extend(container_rows)
    print("GPU benchmark results:", flush=True)
    print(json.dumps(rows, indent=2), flush=True)
    _report_benchmark.remote(rows, steps)
