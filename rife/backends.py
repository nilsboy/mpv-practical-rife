"""Inference backends running the exported RIFE ONNX graph.

The graph takes
  * ``x``        float32 ``[1, 6, H, W]``  (image0 RGB stacked on image1 RGB, [0, 1])
  * ``timestep`` float32 ``[1, 1, 1, 1]``
and returns the merged intermediate RGB frame ```` ``[1, 3, H, W]``.
"""

from __future__ import annotations

import os

import numpy as np


class InferenceError(RuntimeError):
    pass


def _resolve_device(backend: str, device: str | None) -> str:
    if device:
        return device
    if backend == "ov":
        try:
            import openvino as ov
        except ImportError:
            raise InferenceError("openvino is not installed")
        devices = ov.Core().available_devices
        for candidate in ("GPU", "GPU.1"):
            for existing in devices:
                if existing.startswith(candidate):
                    return candidate
        return "CPU"
    if backend == "ort":
        return "CPU"
    raise InferenceError(f"unknown backend {backend!r}")


class OrtBackend:
    name = "ort"
    device_label = "CPU"

    def __init__(self, onnx_path: str, device: str | None = None, threads: int = 0) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = ["CPUExecutionProvider"]
        device = _resolve_device("ort", device)
        if device and device.upper().startswith("CUDA"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.device_label = device or "CPU"
        self.session = ort.InferenceSession(onnx_path, options, providers=providers)

    def run(self, pair: "np.ndarray", timestep: float) -> "np.ndarray":
        result = self.session.run(
            None,
            {"x": pair, "timestep": _scalar(timestep)},
        )
        return np.asarray(result[0])


class OvBackend:
    name = "ov"

    def __init__(self, onnx_path: str, device: str | None = None, threads: int = 0) -> None:
        import openvino as ov

        self.device = _resolve_device("ov", device)
        self.device_label = self.device
        config = {"PERFORMANCE_HINT": "LATENCY"}
        if self.device.upper() == "CPU" and threads > 0:
            config["INFERENCE_NUM_THREADS"] = threads
        core = ov.Core()
        if self.device.upper().startswith("GPU"):
            config["CACHE_DIR"] = os.environ.get("MPV_RIFE_OV_CACHE", "/tmp/opencode-rife-ov-cache")
        compiled = core.compile_model(onnx_path, self.device, config)
        self.output = compiled.output(0)
        self.request = compiled.create_infer_request()

    def run(self, pair: "np.ndarray", timestep: float) -> "np.ndarray":
        result = self.request.infer({0: pair, 1: _scalar(timestep)})
        return np.asarray(result[self.output])


_BACKENDS = {"ort": OrtBackend, "ov": OvBackend}


def create_backend(onnx_path: str, backend: str = "auto", device: str = "", threads: int = 0):
    """Instantiate the first working backend, ordered by preference.

    Preference: openvino GPU (if available) > openvino CPU > onnxruntime CPU.
    ``backend`` pins a choice ("ort"/"ov"); ``device`` pins the runtime's
    device (e.g. "CPU"/"GPU"/"CUDA").
    """
    if not onnx_path or not os.path.exists(onnx_path):
        raise InferenceError(f"model not found: {onnx_path!r}")

    def _candidates() -> list:
        if backend == "auto":
            try:
                import openvino  # noqa: F401

                yield ("ov", "GPU")
                yield ("ov", "")
            except ImportError:
                pass
            yield ("ort", "")
            return
        yield (backend, device)

    last_error: Exception | None = None
    for backend_name, device_name in _candidates():
        try:
            return _BACKENDS[backend_name](onnx_path, device_name, threads)
        except Exception as exc:
            last_error = exc
    raise InferenceError(f"no usable inference backend (last error: {last_error})")


def _scalar(timestep: float) -> "np.ndarray":
    return np.full((1, 1, 1, 1), float(timestep), np.float32)
