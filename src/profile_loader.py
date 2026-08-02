"""Ad-hoc CPU-only profiling of the DreamZero sharded dataloader.

Not part of the training pipeline. Drives the real mixture dataset exactly as
training does (same hydra config composition and overrides as
``src.modal_app._training_command``) on a CPU container over the 10-repo
benchmark subset copied to NVMe, and reports where dataloader time goes.

Run:
    modal run src/profile_loader.py::probe
    modal run src/profile_loader.py::profile
"""

from __future__ import annotations

import json
import math
import subprocess
import time

import modal

from src.modal_app import (
    DATA_VOLUME_ROOT,
    checkpoint_volume,
    data_volume,
    secrets,
    training_image,
)

profile_app = modal.App("dreamzero-yam-profile")


@profile_app.function(
    image=training_image,
    volumes={"/data": data_volume},
    cpu=2.0,
    memory=4096,
    timeout=900,
)
def probe() -> dict:
    """Report per-repo frame counts, resolutions, and shard sizing."""
    root = DATA_VOLUME_ROOT / "allenai"
    repos = sorted(path.name for path in root.iterdir() if path.is_dir())
    out: dict = {"repos": {}}
    total_frames_all = 0
    total_shards_all = 0
    for name in repos:
        info = json.loads((root / name / "meta/info.json").read_text())
        frames = info["total_frames"]
        total_frames_all += frames
        total_shards_all += math.ceil(frames / 1e4)
        out["repos"][name] = {
            "total_frames": frames,
            "total_episodes": info["total_episodes"],
            "fps": info.get("fps"),
        }
    out["total_frames_all"] = total_frames_all
    out["total_shards_all"] = total_shards_all
    first = repos[0]
    out["first_repo"] = first
    out["features_video"] = {
        key: value
        for key, value in json.loads((root / first / "meta/info.json").read_text())[
            "features"
        ].items()
        if value.get("dtype") == "video"
    }
    video = next((root / first / "videos").rglob("*.mp4"))
    ffprobe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,codec_name",
            "-of",
            "json",
            str(video),
        ]
    )
    out["ffprobe_sample"] = json.loads(ffprobe)["streams"][0]
    du = subprocess.check_output(["du", "-sm", str(root / first)])
    out["first_repo_mb"] = int(du.split()[0])
    bench_repos = repos[:10]
    du10 = 0
    for name in bench_repos:
        du10 += int(subprocess.check_output(["du", "-sm", str(root / name)]).split()[0])
    out["bench_10repo_mb"] = du10
    out["bench_repos"] = bench_repos
    print(json.dumps(out, indent=2))
    return out


PROFILE_OVERRIDES = [
    # Mirror src.modal_app._training_command data-path overrides exactly.
    "data=dreamzero/yam_dataset",
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
    "output_dir=/tmp/profile-out",
    "per_device_train_batch_size=2",
    "gradient_accumulation_steps=2",
    "global_batch_size=32",
    "max_steps=1000",
    "save_total_limit=5",
    "dataloader_pin_memory=false",
    "dataloader_num_workers=1",
    "image_resolution_width=320",
    "image_resolution_height=176",
    "max_chunk_size=4",
    "frame_seqlen=880",
    "tokenizer_path=/checkpoints/umt5-xxl",
]


def _build_dataset(rate_override: float | None = None, extra: list[str] | None = None):
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    overrides = list(PROFILE_OVERRIDES)
    if rate_override is not None:
        overrides.append(f"dataset_shard_sampling_rate={rate_override}")
    if extra:
        overrides.extend(extra)
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        config_dir="/root/dreamzero/groot/vla/configs", version_base=None
    ):
        cfg = hydra.compose(config_name="conf", overrides=overrides)
    dataset = instantiate(cfg.train_dataset)
    collator = instantiate(cfg.data_collator)
    return dataset, collator


class _Prof:
    """Aggregate timing sink shared via module global (per process)."""

    def __init__(self) -> None:
        self.records: dict[str, list[float]] = {}
        self.events: list[dict] = []

    def add(self, key: str, seconds: float) -> None:
        self.records.setdefault(key, []).append(seconds)

    def event(self, **kwargs) -> None:
        kwargs["t"] = time.monotonic()
        self.events.append(kwargs)
        print(f"[PROF-EVENT] {json.dumps(kwargs)}", flush=True)

    def summary(self) -> dict:
        out = {}
        for key, values in sorted(self.records.items()):
            out[key] = {
                "n": len(values),
                "total_s": round(sum(values), 3),
                "mean_s": round(sum(values) / len(values), 4),
                "max_s": round(max(values), 3),
            }
        return out


PROF = _Prof()


def _instrument_classes() -> None:
    """Patch shard cache + per-sample stages with timers (idempotent)."""
    from groot.vla.data.dataset import lerobot_sharded as ls

    cls = ls.ShardedLeRobotSubLangSingleActionChunkDatasetDROID
    if getattr(cls, "_prof_patched", False):
        return
    raw = cls.__dict__["get_shard"]
    original_get_shard = raw.__func__ if isinstance(raw, staticmethod) else raw

    def timed_get_shard(*args, **kwargs):
        started = time.monotonic()
        result = original_get_shard(*args, **kwargs)
        elapsed = time.monotonic() - started
        cached_frames = result[0]
        n_frames = sum(int(v.shape[0]) for v in cached_frames.values())
        n_bytes = sum(int(v.nbytes) for v in cached_frames.values())
        PROF.add("shard_cache_total", elapsed)
        PROF.event(
            kind="shard_cache",
            seconds=round(elapsed, 2),
            trajectories=len(args[0]),
            frames=n_frames,
            gib=round(n_bytes / 2**30, 2),
        )
        return result

    cls.get_shard = staticmethod(timed_get_shard)

    original_finish = cls.finish_cache_shard

    def timed_finish(self):
        started = time.monotonic()
        original_finish(self)
        waited = time.monotonic() - started
        PROF.add("shard_wait", waited)
        PROF.event(kind="shard_wait", seconds=round(waited, 3))

    cls.finish_cache_shard = timed_finish

    original_get_step_data = cls.get_step_data

    def timed_get_step_data(self, trajectory_id, indices):
        started = time.monotonic()
        result = original_get_step_data(self, trajectory_id, indices)
        PROF.add("get_step_data", time.monotonic() - started)
        if result is None:
            PROF.add("sample_skipped", 1.0)
        count = len(PROF.records["get_step_data"])
        if count % 100 == 0:
            print(f"[PROF-WORKER-SUMMARY] {json.dumps(PROF.summary())}", flush=True)
        return result

    cls.get_step_data = timed_get_step_data

    original_get_video = cls.get_video

    def timed_get_video(self, trajectory_id, key, step_indices):
        started = time.monotonic()
        result = original_get_video(self, trajectory_id, key, step_indices)
        PROF.add("get_video", time.monotonic() - started)
        return result

    cls.get_video = timed_get_video

    original_get_traj = cls.get_trajectory_data

    def timed_get_traj(self, trajectory_id):
        started = time.monotonic()
        result = original_get_traj(self, trajectory_id)
        PROF.add("get_trajectory_data", time.monotonic() - started)
        return result

    cls.get_trajectory_data = timed_get_traj
    cls._prof_patched = True

    # Per-sub-transform timing on the composed pipeline.
    from groot.vla.data.transform.base import ComposedModalityTransform

    if not getattr(ComposedModalityTransform, "_prof_patched", False):
        original_call = ComposedModalityTransform.__call__

        def timed_call(self, data):
            started = time.monotonic()
            result = original_call(self, data)
            PROF.add("transforms_total", time.monotonic() - started)
            return result

        ComposedModalityTransform.__call__ = timed_call

        _ = ComposedModalityTransform.apply

        def timed_apply(self, data):
            for transform in self.transforms:
                name = type(transform).__name__
                started = time.monotonic()
                data = transform(data)
                PROF.add(f"transform.{name}", time.monotonic() - started)
            return data

        ComposedModalityTransform.apply = timed_apply
        ComposedModalityTransform._prof_patched = True


def _iterate_inprocess(dataset, collator, num_batches: int, batch_size: int) -> dict:
    """Phase A: iterate the mixture in-process with fine-grained timers.

    torch intra-op threads are pinned to 1 to match what a DataLoader worker
    process gets (PyTorch forces num_threads=1 inside workers).
    """
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    iterator = iter(dataset)
    batch_times = []
    started_all = time.monotonic()
    for batch_index in range(num_batches):
        samples = []
        batch_start = time.monotonic()
        for _ in range(batch_size):
            sample_start = time.monotonic()
            samples.append(next(iterator))
            PROF.add("sample_total", time.monotonic() - sample_start)
        collate_start = time.monotonic()
        collator(samples)
        PROF.add("collate", time.monotonic() - collate_start)
        batch_times.append(time.monotonic() - batch_start)
        if batch_index % 10 == 0:
            print(
                f"[PROF] in-process batch {batch_index}: {batch_times[-1]:.2f}s",
                flush=True,
            )
    elapsed = time.monotonic() - started_all
    torch.set_num_threads(previous_threads)
    return {
        "batches": num_batches,
        "batch_size": batch_size,
        "wall_s": round(elapsed, 2),
        "mean_batch_s": round(sum(batch_times) / len(batch_times), 3),
        "p50_batch_s": round(sorted(batch_times)[len(batch_times) // 2], 3),
        "max_batch_s": round(max(batch_times), 3),
    }


def _worker_init(_worker_id: int) -> None:
    _instrument_classes()


def _iterate_dataloader(
    dataset, collator, num_batches: int, batch_size: int, num_workers: int
) -> dict:
    """Phase B: measure batch arrival times through a real torch DataLoader."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )
    iterator = iter(loader)
    arrival = []
    first_batch_s = None
    for batch_index in range(num_batches):
        started = time.monotonic()
        next(iterator)
        waited = time.monotonic() - started
        if batch_index == 0:
            first_batch_s = waited
        else:
            arrival.append(waited)
        if batch_index % 10 == 0:
            print(f"[PROF] loader batch {batch_index}: {waited:.2f}s", flush=True)
    del iterator, loader
    arrival_sorted = sorted(arrival)
    return {
        "batches": num_batches,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "first_batch_s": round(first_batch_s, 2),
        "steady_mean_s": round(sum(arrival) / len(arrival), 3),
        "steady_p50_s": round(arrival_sorted[len(arrival) // 2], 3),
        "steady_p90_s": round(arrival_sorted[int(len(arrival) * 0.9)], 3),
        "steady_max_s": round(max(arrival), 3),
    }


def _decode_benchmark(dataset) -> dict:
    """Phase D: raw shard decode cost vs decord thread count."""
    from groot.vla.data.dataset import lerobot_sharded as ls

    del ls
    single = dataset.datasets[0]
    trajectory_ids = single.sharded_trajectories[0]
    # Cap the work: decode only the first few trajectories per config.
    subset = trajectory_ids[: min(3, len(trajectory_ids))]
    results = {}
    for label, kwargs in [
        ("default", {}),
        ("threads1", {"num_threads": 1}),
        ("threads2", {"num_threads": 2}),
        ("threads4", {"num_threads": 4}),
        ("threads8", {"num_threads": 8}),
    ]:
        started = time.monotonic()
        cached_frames, _, _ = single.get_shard(
            subset,
            single.modality_keys,
            single.all_video_paths,
            single.all_parquet_paths,
            "decord",
            kwargs,
            single.fps,
        )
        elapsed = time.monotonic() - started
        n_frames = sum(int(v.shape[0]) for v in cached_frames.values())
        results[label] = {
            "seconds": round(elapsed, 2),
            "frames": n_frames,
            "fps": round(n_frames / elapsed, 1),
        }
        print(f"[PROF] decode {label}: {results[label]}", flush=True)
        del cached_frames
    return results


@profile_app.function(
    image=training_image,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    secrets=secrets,
    cpu=16.0,
    memory=120 * 1024,
    timeout=3 * 3600,
)
def profile(
    batches_inprocess: int = 40,
    batches_loader: int = 50,
    batch_size: int = 2,
) -> dict:
    from src.modal_app import (
        _copy_training_data_to_nvme,
        _install_data_config,
        _read_repos,
    )

    report: dict = {}
    benchmark_repos = sorted(_read_repos()["train"])[:10]
    _install_data_config(benchmark_repos)
    _copy_training_data_to_nvme(repo_ids=benchmark_repos)

    _instrument_classes()

    print("[PROF] building dataset (rate=0.1)", flush=True)
    build_start = time.monotonic()
    dataset, collator = _build_dataset(rate_override=0.1)
    report["dataset_build_s"] = round(time.monotonic() - build_start, 1)
    total_shards = sum(single.num_shards for single in dataset.datasets)
    report["total_shards_10repo"] = total_shards

    import gc

    # Phase A: in-process iteration with fine-grained timers.
    PROF.records.clear()
    report["phaseA_inprocess_rate0.1"] = _iterate_inprocess(
        dataset, collator, batches_inprocess, batch_size
    )
    report["phaseA_breakdown"] = PROF.summary()
    print(
        "[PROF-PARTIAL] phaseA: "
        + json.dumps(
            {
                "summary": report["phaseA_inprocess_rate0.1"],
                "breakdown": report["phaseA_breakdown"],
            }
        ),
        flush=True,
    )

    # Phase D: decode scaling on the same dataset (uses only static members).
    try:
        report["decode_benchmark"] = _decode_benchmark(dataset)
    except Exception as exc:  # noqa: BLE001
        report["decode_benchmark"] = {"error": f"{type(exc).__name__}: {exc}"}
    print(
        "[PROF-PARTIAL] decode: " + json.dumps(report["decode_benchmark"]), flush=True
    )
    del dataset, collator
    gc.collect()

    # Phase B: torch DataLoader with 1 worker, as trained.
    dataset_b, collator_b = _build_dataset(rate_override=0.1)
    report["phaseB_loader_w1_rate0.1"] = _iterate_dataloader(
        dataset_b, collator_b, batches_loader, batch_size, num_workers=1
    )
    del dataset_b

    def _partial(key: str) -> None:
        print(f"[PROF-PARTIAL] {key}: {json.dumps(report.get(key))}", flush=True)

    _partial("phaseB_loader_w1_rate0.1")

    # Phase C: higher sampling rates through the loader.
    for rate in (0.5, 1.0):
        dataset_c, collator_c = _build_dataset(rate_override=rate)
        key = f"phaseC_loader_w1_rate{rate}"
        try:
            report[key] = _iterate_dataloader(
                dataset_c, collator_c, 30, batch_size, num_workers=1
            )
        except Exception as exc:  # noqa: BLE001
            report[key] = {"error": f"{type(exc).__name__}: {exc}"}
        _partial(key)
        del dataset_c
        gc.collect()

    # Phase E/F: worker scaling under the smaller-shard candidate fix (the
    # stock 1e4-step shard is ~21 GiB decoded, so multiple workers per rank
    # blow past the training container's memory; smaller shards keep the
    # identical 10%-of-shard sampling semantics with proportionally less RAM).
    for key, workers, shard_steps in (
        ("phaseE_loader_w2_shard2500_rate0.1", 2, 2500),
        ("phaseF_loader_w4_shard2500_rate0.1", 4, 2500),
    ):
        dataset_x, collator_x = _build_dataset(
            rate_override=0.1,
            extra=[f"+train_dataset.dataset_kwargs.num_steps_per_shard={shard_steps}"],
        )
        try:
            report[key] = _iterate_dataloader(
                dataset_x, collator_x, batches_loader, batch_size, num_workers=workers
            )
        except Exception as exc:  # noqa: BLE001
            report[key] = {"error": f"{type(exc).__name__}: {exc}"}
        _partial(key)
        del dataset_x
        gc.collect()

    print("[PROF-REPORT] " + json.dumps(report, indent=2), flush=True)
    return report


@profile_app.function(
    image=training_image,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    secrets=secrets,
    timeout=4 * 60 * 60,
    cpu=16,
    memory=131_072,
    ephemeral_disk=1_200_000,
)
def shard_check() -> dict:
    """Build the FULL 124-repo dataset with 2500-step shards.

    Guards the latent generate_shards cutoff assert before the real run;
    also smoke-reads a few batches through the small-shard config.
    """
    from src.modal_app import (
        _copy_training_data_to_nvme,
        _install_data_config,
        _read_repos,
    )

    train_repos = sorted(_read_repos()["train"])
    _install_data_config()
    _copy_training_data_to_nvme(repo_ids=train_repos)
    dataset, collator = _build_dataset(
        rate_override=0.1,
        extra=[
            "+train_dataset.dataset_kwargs.num_steps_per_shard=2500",
            "+train_dataset.dataset_kwargs.video_backend_kwargs.num_threads=4",
        ],
    )
    total = sum(single.num_shards for single in dataset.datasets)
    iterator = iter(dataset)
    for _ in range(4):
        next(iterator)
    print(f"[SHARD-CHECK] repos={len(dataset.datasets)} shards={total}", flush=True)
    return {"repos": len(dataset.datasets), "shards": total}
