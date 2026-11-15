"""Offline benchmark for the mpv-practical-rife VapourSynth filter.

Examples:
  python3 tools/bench.py --model rife.onnx --frames 50
  python3 tools/bench.py --model rife.onnx --synthetic 1274x714 --frames 30

Benchmarks interpolation over a synthetic moving-pattern source (an RGB
gradients scrolling every frame, real spatial content for the network).
Prints per-frame latency and achievable interpolated fps; realtime needs,
for a factor of N, N x source_fps interpolated frames per second. Exits 2 if
that requirement is not met.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import vapoursynth as vs

from rife.vsfilter import rife_interpolate


def parse_size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


def moving_pattern(core, width: int, height: int, frames: int, fps: Fraction):
    fmt = core.query_video_format(vs.ColorFamily.RGB, vs.SampleType.FLOAT, 32)
    base = core.std.BlankClip(format=fmt, width=width, height=height,
                              fpsnum=fps.numerator, fpsden=fps.denominator, length=frames)

    def fill(n, f):
        out = f[1].copy()
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        shift = n * 7.0
        r = (xx + yy) / (width + height) * 0.9 + (0.05 + 0.001 * shift)
        g = (yy / height) * 0.7 + 0.001 * n
        b = (xx / width) * 0.6 + 0.002 * n
        data = np.stack([r, g, b]).astype(np.float32)
        for plane in range(3):
            arr = np.asarray(out[plane])
            arr[:, :] = data[plane]
        return out

    blank = core.std.BlankClip(base, length=frames)
    return core.std.ModifyFrame(base, [base, blank], selector=fill)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--synthetic", default="1274x714", help="WxH of the synthetic source")
    parser.add_argument("--source-fps", default="24", help="fps of the synthetic source")
    parser.add_argument("--frames", type=int, default=20, help="number of output inferences to time")
    parser.add_argument("--factor-num", type=int, default=2)
    parser.add_argument("--factor-den", type=int, default=1)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default="")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sc", type=float, default=0.0)
    args = parser.parse_args()

    core = vs.core
    core.num_threads = 2
    width, height = parse_size(args.synthetic)
    src_fps = Fraction(args.source_fps)
    src = moving_pattern(core, width, height, max(args.frames, 8) * args.factor_den // args.factor_den + 4, src_fps)
    out = rife_interpolate(
        src,
        model=args.model,
        backend=args.backend,
        device=args.device,
        threads=args.threads,
        factor_num=args.factor_num,
        factor_den=args.factor_den,
        sc=args.sc,
        source_fps=src_fps,
    )
    out_fps_needed = float(src_fps * args.factor_num / args.factor_den)
    print(f"backend={args.backend!r} device={args.device!r} source {width}x{height}@{src_fps}"
          f" -> target output {out_fps_needed:.3f} fps")

    start_n = 3
    count = 0
    total = 0.0
    values = []
    for n in range(start_n, start_n + args.frames + args.warmup):
        if n >= out.num_frames:
            break
        t0 = time.perf_counter()
        frame = out.get_frame(n)
        ms = (time.perf_counter() - t0) * 1000
        count += 1
        if count > args.warmup:
            total += ms
        values.append(float(np.asarray(frame[1])[height // 4, width // 4]))
    frames_timed = count - args.warmup
    if frames_timed <= 0:
        print("no frames timed")
        return 1
    avg_ms = total / frames_timed
    ips = 1000.0 / avg_ms
    realtime = ips >= out_fps_needed
    print(f"timed {frames_timed} interpolated frames, avg {avg_ms:.1f} ms/frame"
          f" -> {ips:.2f} interp/sec (need {out_fps_needed:.2f} for realtime)")
    print("values", [f"{v:.3f}" for v in values[:8]])
    print("REALTIME OK" if realtime else "REALTIME NOT MET")
    return 0 if realtime else 2


if __name__ == "__main__":
    sys.exit(main())
