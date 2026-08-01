"""Profiling-grade GPU telemetry for every DreamZero GPU stage.

Acquisition deliberately degrades without affecting the workload:
DCGM profiling counters -> NVML GPM -> ordinary NVML utilization/VRAM.
Every sample carries ``dcgm/source_tier`` and is emitted per GPU plus as a
node mean.  The standalone process uses a separate WandB run in the same
stage group so concurrent logging cannot corrupt Trainer's local run files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

FIELD_SM = 1002  # DCGM_FI_PROF_SM_ACTIVE
FIELD_TENSOR = 1004  # DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
FIELD_DRAM = 1005  # DCGM_FI_PROF_DRAM_ACTIVE
FIELD_PCIE_TX = 1009
FIELD_PCIE_RX = 1010
FIELD_NVLINK_TX = 1011
FIELD_NVLINK_RX = 1012
DCGM_FIELDS = (
    FIELD_TENSOR,
    FIELD_DRAM,
    FIELD_SM,
    FIELD_PCIE_TX,
    FIELD_PCIE_RX,
    FIELD_NVLINK_TX,
    FIELD_NVLINK_RX,
)
DCGM_NAMES = (
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "DCGM_FI_PROF_DRAM_ACTIVE",
    "DCGM_FI_PROF_SM_ACTIVE",
    "DCGM_FI_PROF_PCIE_TX_BYTES",
    "DCGM_FI_PROF_PCIE_RX_BYTES",
    "DCGM_FI_PROF_NVLINK_TX_BYTES",
    "DCGM_FI_PROF_NVLINK_RX_BYTES",
)

GPU_PRICE_PER_HOUR = {"H100": 3.95, "H200": 4.54, "B200": 6.25}
DENSE_BF16_PEAK_TFLOPS = {"H100": 989.0, "H200": 989.0, "B200": 2250.0}


def parse_gpu_spec(spec: str) -> tuple[str, int]:
    try:
        architecture, count_text = spec.upper().split(":", 1)
        count = int(count_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"GPU spec must look like H100:8, got {spec!r}") from exc
    if architecture not in DENSE_BF16_PEAK_TFLOPS or count < 1:
        raise ValueError(
            f"unsupported GPU spec {spec!r}; architecture must be one of "
            f"{sorted(DENSE_BF16_PEAK_TFLOPS)} and count must be positive"
        )
    return architecture, count


def wan_dit_training_flops_per_sample(
    *,
    dim: int = 5120,
    ffn_dim: int = 13824,
    num_layers: int = 40,
    tokens_per_latent_frame: int = 880,
    num_latent_frames: int = 9,
) -> float:
    """Approximate one 14B Wan DiT training sample in floating-point ops.

    Per transformer layer, counting multiply and add separately:
      QKV + output projections: 8*S*d^2
      attention score/value products: 4*S^2*d
      gated FFN (gate, up, down): 6*S*d*f
    The factor 3 is the standard forward + backward training approximation.
    Dimensions come from DreamZero's Wan2.1 14B config. Its 33 input frames
    become ``(33-1)//4+1 == 9`` VAE-latent frames, each with the configured
    ``frame_seqlen == 880`` tokens, so S=7,920 (action/state tokens are tiny
    beside the video sequence and are omitted from this approximation).
    This intentionally mirrors the analytic SO-101 accounting: it is a stable
    architecture-level comparison metric, not a profiler's operator total.
    """
    sequence_length = tokens_per_latent_frame * num_latent_frames
    per_layer = (
        8 * sequence_length * dim**2
        + 4 * sequence_length**2 * dim
        + 6 * sequence_length * dim * ffn_dim
    )
    return float(3 * num_layers * per_layer)


def throughput_metrics(samples_per_second: float, gpu_spec: str) -> dict[str, float]:
    architecture, count = parse_gpu_spec(gpu_spec)
    per_gpu = (
        wan_dit_training_flops_per_sample() * float(samples_per_second) / count / 1e12
    )
    mfu = per_gpu / DENSE_BF16_PEAK_TFLOPS[architecture]
    cost_per_hour = GPU_PRICE_PER_HOUR[architecture] * count
    cost_per_1k = (
        math.inf
        if samples_per_second <= 0
        else cost_per_hour * 1000.0 / (samples_per_second * 3600.0)
    )
    return {
        "perf/samples_per_second": float(samples_per_second),
        "perf/model_tflops_per_second_per_gpu": per_gpu,
        "perf/mfu": mfu,
        "perf/cost_per_1k_samples_usd": cost_per_1k,
    }


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _parse_dmon_line(line: str) -> tuple[int, list[float | None]] | None:
    parts = line.split()
    if len(parts) < 3 or parts[0] != "GPU":
        return None
    try:
        gpu_index = int(parts[1])
    except ValueError:
        return None
    values = [_number(item) for item in parts[2 : 2 + len(DCGM_FIELDS)]]
    values.extend([None] * (len(DCGM_FIELDS) - len(values)))
    return gpu_index, values


def _aggregate_per_gpu(per_gpu: dict[int, dict[str, float]]) -> dict[str, float]:
    output: dict[str, float] = {}
    by_name: dict[str, list[float]] = {}
    for gpu_index, values in sorted(per_gpu.items()):
        for name, value in values.items():
            output[f"gpu/{gpu_index}/{name}"] = value
            by_name.setdefault(name, []).append(value)
    for name, values in by_name.items():
        output[f"gpu/mean/{name}"] = sum(values) / len(values)
    return output


class GpuMetricsSampler:
    """Tiered sampler. Sampling errors demote tiers and never reach training."""

    TIERS = ("dcgm", "gpm", "nvml-util")

    def __init__(
        self,
        interval_s: float = 10.0,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.interval_s = interval_s
        self.sink = sink or (lambda values: print(json.dumps(values), flush=True))
        self.source: str | None = None
        self.samples_taken = 0
        self.last_values: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dmon_proc: subprocess.Popen[str] | None = None
        self._dmon_latest: dict[int, dict[str, float]] = {}
        self._dmon_timestamp = 0.0
        self._dmon_lock = threading.Lock()

    def _try_dcgm(self) -> bool:
        try:
            subprocess.run(
                ["nv-hostengine", "-b", "127.0.0.1"],
                capture_output=True,
                timeout=60,
                check=False,
            )
            delay_ms = max(1000, min(int(self.interval_s * 1000), 10_000))
            self._dmon_proc = subprocess.Popen(
                [
                    "dcgmi",
                    "dmon",
                    "-e",
                    ",".join(str(field) for field in DCGM_FIELDS),
                    "-d",
                    str(delay_ms),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )

            def reader() -> None:
                assert self._dmon_proc is not None
                assert self._dmon_proc.stdout is not None
                for line in self._dmon_proc.stdout:
                    parsed = _parse_dmon_line(line)
                    if parsed is None:
                        continue
                    gpu_index, raw_values = parsed
                    values = {
                        name: value
                        for name, value in zip(DCGM_NAMES, raw_values, strict=True)
                        if value is not None
                    }
                    if values:
                        with self._dmon_lock:
                            self._dmon_latest[gpu_index] = values
                            self._dmon_timestamp = time.time()

            threading.Thread(target=reader, daemon=True).start()
            deadline = time.time() + max(15.0, 3 * self.interval_s)
            while time.time() < deadline:
                if self._dmon_proc.poll() is not None:
                    return False
                with self._dmon_lock:
                    if self._dmon_latest:
                        self.source = "dcgm"
                        return True
                time.sleep(1.0)
            self._kill_dmon()
            return False
        except Exception as exc:  # noqa: BLE001
            print(f"[gpu_metrics] DCGM unavailable: {type(exc).__name__}: {exc}")
            self._kill_dmon()
            return False

    def _kill_dmon(self) -> None:
        if self._dmon_proc is not None:
            try:
                self._dmon_proc.kill()
            except Exception:  # noqa: BLE001
                pass
            self._dmon_proc = None

    def _sample_dcgm(self) -> dict[str, float]:
        if self._dmon_proc is None or self._dmon_proc.poll() is not None:
            raise RuntimeError("dcgmi dmon process died")
        with self._dmon_lock:
            fresh = self._dmon_timestamp > time.time() - 3 * self.interval_s - 15
            values = {index: dict(row) for index, row in self._dmon_latest.items()}
        return _aggregate_per_gpu(values) if fresh else {}

    def _try_gpm(self) -> bool:
        try:
            import pynvml as nv

            nv.nvmlInit()
            handles = [
                nv.nvmlDeviceGetHandleByIndex(index)
                for index in range(nv.nvmlDeviceGetCount())
            ]
            if not handles or not all(
                nv.nvmlGpmQueryDeviceSupport(handle).isSupportedDevice
                for handle in handles
            ):
                return False
            self._nv = nv
            self._handles = handles
            if not self._sample_gpm():
                return False
            self.source = "gpm"
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[gpu_metrics] NVML GPM unavailable: {type(exc).__name__}: {exc}")
            return False

    def _sample_gpm(self) -> dict[str, float]:
        nv = self._nv
        per_gpu: dict[int, dict[str, float]] = {}
        for gpu_index, handle in enumerate(self._handles):
            first = nv.nvmlGpmSampleAlloc()
            second = nv.nvmlGpmSampleAlloc()
            try:
                nv.nvmlGpmSampleGet(handle, first)
                time.sleep(1.0)
                nv.nvmlGpmSampleGet(handle, second)
                request = nv.c_nvmlGpmMetricsGet_t()
                request.version = nv.NVML_GPM_METRICS_GET_VERSION
                request.numMetrics = 2
                request.sample1 = first
                request.sample2 = second
                request.metrics[0].metricId = nv.NVML_GPM_METRIC_ANY_TENSOR_UTIL
                request.metrics[1].metricId = nv.NVML_GPM_METRIC_DRAM_BW_UTIL
                nv.nvmlGpmMetricsGet(request)
                per_gpu[gpu_index] = {
                    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": (
                        request.metrics[0].value / 100.0
                    ),
                    "DCGM_FI_PROF_DRAM_ACTIVE": request.metrics[1].value / 100.0,
                }
            finally:
                nv.nvmlGpmSampleFree(first)
                nv.nvmlGpmSampleFree(second)
        return _aggregate_per_gpu(per_gpu)

    def _try_nvml_util(self) -> bool:
        try:
            import pynvml as nv

            nv.nvmlInit()
            self._nv = nv
            self._handles = [
                nv.nvmlDeviceGetHandleByIndex(index)
                for index in range(nv.nvmlDeviceGetCount())
            ]
            if not self._sample_nvml_util():
                return False
            self.source = "nvml-util"
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[gpu_metrics] NVML unavailable: {type(exc).__name__}: {exc}")
            return False

    def _sample_nvml_util(self) -> dict[str, float]:
        per_gpu: dict[int, dict[str, float]] = {}
        for gpu_index, handle in enumerate(self._handles):
            utilization = self._nv.nvmlDeviceGetUtilizationRates(handle)
            memory = self._nv.nvmlDeviceGetMemoryInfo(handle)
            per_gpu[gpu_index] = {
                "util_percent": float(utilization.gpu),
                "memory_util_percent": float(utilization.memory),
                "vram_used_bytes": float(memory.used),
                "vram_total_bytes": float(memory.total),
            }
        return _aggregate_per_gpu(per_gpu)

    def _setup_from(self, tier_index: int) -> bool:
        setup = {
            "dcgm": self._try_dcgm,
            "gpm": self._try_gpm,
            "nvml-util": self._try_nvml_util,
        }
        for tier in self.TIERS[tier_index:]:
            if setup[tier]():
                self.source = tier
                print(
                    f"[gpu_metrics] source_tier={self.TIERS.index(tier) + 1} ({tier})"
                )
                return True
        return False

    def _loop(self) -> None:
        sample = {
            "dcgm": self._sample_dcgm,
            "gpm": self._sample_gpm,
            "nvml-util": self._sample_nvml_util,
        }
        empty_streak = 0
        while not self._stop.is_set() and self.source is not None:
            try:
                values = sample[self.source]()
                if values:
                    empty_streak = 0
                    values["dcgm/source_tier"] = self.TIERS.index(self.source) + 1
                    self.last_values = dict(values)
                    self.samples_taken += 1
                    self.sink(values)
                else:
                    empty_streak += 1
            except Exception as exc:  # noqa: BLE001
                empty_streak += 1
                if empty_streak <= 2:
                    print(f"[gpu_metrics] sample failed: {type(exc).__name__}: {exc}")
            if empty_streak >= 10:
                next_tier = self.TIERS.index(self.source) + 1
                self._kill_dmon()
                if next_tier >= len(self.TIERS) or not self._setup_from(next_tier):
                    print("[gpu_metrics] all tiers exhausted; sampler disabled")
                    return
                empty_streak = 0
            self._stop.wait(self.interval_s)

    def start(self) -> GpuMetricsSampler:
        try:
            if self._setup_from(0):
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
            else:
                print("[gpu_metrics] no metrics source available; disabled")
        except Exception as exc:  # noqa: BLE001
            print(f"[gpu_metrics] start failed, disabled: {type(exc).__name__}: {exc}")
        return self

    def stop(self) -> None:
        self._stop.set()
        self._kill_dmon()
        if self._thread is not None:
            self._thread.join(timeout=10)


def _last_runtime_rate(path: Path) -> float | None:
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        for line in reversed(lines):
            row = json.loads(line)
            value = float(row.get("samples_per_second", 0.0))
            if value > 0 and math.isfinite(value):
                return value
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return None


def run_collector(
    *, stage: str, gpu_spec: str, runtime_metrics: Path, interval_s: float
) -> None:
    import wandb

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dreamzero-yam"),
        id=f"{stage}-gpu-metrics",
        name=f"{stage}-gpu-metrics",
        group=stage,
        job_type="gpu-telemetry",
        resume="allow",
        config={
            "gpu_spec": gpu_spec,
            "source_tiers": "dcgm,nvml-gpm,nvml-util",
            "wan_dit_flops_per_sample": wan_dit_training_flops_per_sample(),
        },
    )

    def sink(values: dict[str, Any]) -> None:
        rate = _last_runtime_rate(runtime_metrics)
        if rate is not None:
            values.update(throughput_metrics(rate, gpu_spec))
        run.log(values)

    sampler = GpuMetricsSampler(interval_s=interval_s, sink=sink).start()
    stop = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while not stop.wait(1.0):
            pass
    finally:
        sampler.stop()
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--gpu-spec", required=True)
    parser.add_argument("--runtime-metrics", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    parse_gpu_spec(args.gpu_spec)
    run_collector(
        stage=args.stage,
        gpu_spec=args.gpu_spec,
        runtime_metrics=args.runtime_metrics,
        interval_s=args.interval_seconds,
    )


if __name__ == "__main__":
    main()
