"""VapourSynth frame interpolation filter using the exported Practical-RIFE
ONNX model.

``rife_interpolate(clip, ...)`` returns a clip whose frame rate is multiplied
by ``factor_num / factor_den``. Output frames that coincide with a source
frame are pass-throughs; every other output frame is synthesized by the
network between the nearest source neighbours.

The filter is lazy (``std.FrameEval`` + ``std.ModifyFrame``): frames are only
inferred on demand, which keeps memory use bounded during playback even when
the network is slower than realtime.

Every output frame carries ``_DurationNum``/``_DurationDen`` frame properties
so that mpv can pace the result correctly through its ``vf_vapoursynth``
bridge.
"""

from __future__ import annotations

import threading
from fractions import Fraction

import numpy as np
import vapoursynth as vs

from .backends import create_backend
from .config import load_settings

_INFERENCE_LOCK = threading.Lock()


def _rgbs_format() -> vs.VideoFormat:
    rgbs = getattr(vs, "RGBS", None)
    if rgbs is None:
        return vs.core.query_video_format(vs.ColorFamily.RGB, vs.SampleType.FLOAT, 32)
    return vs.core.get_video_format(rgbs)


_RGBS_ID = None


def _rgbs_id() -> int:
    global _RGBS_ID
    if _RGBS_ID is None:
        _RGBS_ID = _rgbs_format().id
    return _RGBS_ID


def _to_rgb(clip: vs.VideoNode, matrix: str) -> vs.VideoNode:
    if clip.format.id == _rgbs_id():
        return clip
    if clip.format.color_family not in (vs.ColorFamily.YUV, vs.ColorFamily.GRAY, vs.ColorFamily.RGB):
        raise ValueError(f"unsupported color family {clip.format.name}")
    return clip.resize.Bicubic(matrix_in_s=matrix, format=_rgbs_format())


def _frame_diff(a: "np.ndarray", b: "np.ndarray") -> float:
    return float(np.mean(np.abs(a - b)))


class _InterpState:
    """Holds the per-playback state referenced from the VS filter callbacks."""

    def __init__(
        self,
        backend,
        src: vs.VideoNode,
        factor_num: int,
        factor_den: int,
        sc_threshold: float,
        height: int,
        width: int,
        pad_height: int,
        pad_width: int,
        duration: tuple[int, int],
    ) -> None:
        self.backend = backend
        self.src = src
        self.factor_num = factor_num
        self.factor_den = factor_den
        self.sc_threshold = sc_threshold
        self.height = height
        self.width = width
        self.pad_height = pad_height
        self.pad_width = pad_width
        self.duration = duration

    def _loop(self, index: int) -> vs.VideoNode:
        trimmed = vs.core.std.Trim(self.src, index, index)
        return vs.core.std.Loop(trimmed, times=max(2, self.src.num_frames))

    def _with_duration(self, frame: vs.VideoFrame) -> vs.VideoFrame:
        frame.props["_DurationNum"] = int(self.duration[0])
        frame.props["_DurationDen"] = int(self.duration[1])
        return frame

    def build(self, n: int) -> vs.VideoNode:
        core = vs.core
        i, r = divmod(n * self.factor_den, self.factor_num)

        if r == 0:
            loop = self._loop(i)

            def passthrough(n, f):
                frame = f[0] if isinstance(f, (list, tuple)) else f
                return self._with_duration(frame.copy())

            return core.std.ModifyFrame(self.src, loop, selector=passthrough)

        loop_i = self._loop(i)
        loop_j = self._loop(i + 1)

        def interpolate(n, f):
            a0 = np.stack([np.asarray(plane) for plane in f[0]])
            a1 = np.stack([np.asarray(plane) for plane in f[1]])
            if self.sc_threshold > 0.0 and _frame_diff(a0, a1) > self.sc_threshold:
                data = a1 if 2 * r > self.factor_num else a0
            else:
                x = self._pack(a0, a1)
                timestep = r / self.factor_num
                with _INFERENCE_LOCK:
                    y = self.backend.run(x, timestep)
                data = y[0, :, : self.height, : self.width]
            out = f[0].copy()
            for plane, plane_data in enumerate(data):
                arr = np.asarray(out[plane])
                if arr.shape == plane_data.shape:
                    arr[:] = plane_data
                else:
                    arr[:, : plane_data.shape[1]] = plane_data
            return self._with_duration(out)

        return core.std.ModifyFrame(self.src, [loop_i, loop_j], selector=interpolate)

    def _pack(self, a0: "np.ndarray", a1: "np.ndarray") -> "np.ndarray":
        pad_v = self.pad_height - self.height
        pad_h = self.pad_width - self.width
        if pad_v or pad_h:
            a0 = np.pad(a0, ((0, 0), (0, pad_v), (0, pad_h)))
            a1 = np.pad(a1, ((0, 0), (0, pad_v), (0, pad_h)))
        return np.concatenate((a0, a1), axis=0)[np.newaxis].astype(np.float32, copy=False)


def rife_interpolate(
    clip: vs.VideoNode,
    model: str = "",
    backend: str = "auto",
    device: str = "",
    threads: int = 0,
    factor_num: int = 2,
    factor_den: int = 1,
    sc: float = 0.15,
    pad: int = 128,
    source_fps: Fraction | None = None,
    user_data: str | None = None,
    matrix: str | None = None,
) -> vs.VideoNode:
    """Interpolate ``clip`` to ``fps * factor_num / factor_den`` with RIFE."""
    settings = load_settings(
        user_data,
        {
            "model": model,
            "backend": backend,
            "device": device,
            "threads": threads,
            "factor_num": factor_num,
            "factor_den": factor_den,
            "sc": sc,
            "pad": pad,
        },
    )
    model_path = str(settings["model"])
    backend_obj = create_backend(
        model_path,
        str(settings["backend"]),
        str(settings["device"]),
        int(settings["threads"]),
    )
    return _build_clip(
        clip,
        backend_obj,
        factor_num=int(settings["factor_num"]),
        factor_den=int(settings["factor_den"]),
        sc_threshold=float(settings["sc"]),
        pad=int(settings["pad"]),
        source_fps=source_fps,
        matrix=str(settings["matrix"]) if matrix is None else matrix,
    )


def _build_clip(
    clip: vs.VideoNode,
    backend,
    factor_num: int,
    factor_den: int,
    sc_threshold: float,
    pad: int,
    source_fps: Fraction | None,
    matrix: str,
) -> vs.VideoNode:
    core = vs.core
    if factor_num < 1 or factor_den < 1:
        raise ValueError("fps factor numerator/denominator must be positive")
    if pad < 1:
        raise ValueError("pad must be positive")

    src_rgb = _to_rgb(clip, matrix)
    fps = source_fps if source_fps is not None else src_rgb.fps
    fps = Fraction(fps)
    if fps <= 0:
        raise ValueError("source clip has no fps and no source_fps override given")

    out_fps = fps * Fraction(factor_num, factor_den)
    pad_height = max(pad, (src_rgb.height - 1) // pad * pad + pad)
    pad_width = max(pad, (src_rgb.width - 1) // pad * pad + pad)

    duration = Fraction(1) / out_fps

    if src_rgb.num_frames > 1:
        out_len = (src_rgb.num_frames - 1) * factor_num // factor_den + 1
    else:
        out_len = src_rgb.num_frames
    base = core.std.BlankClip(
        src_rgb, length=out_len, fpsnum=out_fps.numerator, fpsden=out_fps.denominator
    )

    state = _InterpState(
        backend,
        src_rgb,
        factor_num,
        factor_den,
        sc_threshold,
        src_rgb.height,
        src_rgb.width,
        pad_height,
        pad_width,
        (duration.numerator, duration.denominator),
    )

    def eval_(n: int) -> vs.VideoNode:
        return state.build(n)

    out = core.std.FrameEval(base, eval_, clip_src=[src_rgb])

    # mpv's vf_vapoursynth only propagates YUV planar formats in both
    # directions, so return the clip in the caller's colorspace unless the
    # caller (e.g. offline VapourSynth scripts) handed us RGB directly.
    if clip.format.id != out.format.id and clip.format.color_family == vs.ColorFamily.YUV:
        out = out.resize.Bicubic(matrix_s=matrix, format=clip.format)
    return out

