"""mpv vf_vapoursynth script: Practical-RIFE realtime frame interpolation.

Usage:

    mpv --vf=vapoursynth=/path/to/mpv/rife_video.py:buffered_frames=4 \
        video.mkv

Configuration reaches this script through the filter's ``user-data``
suboption (JSON or ``b64:`` JSON, produced by mpv/rife.lua) or through plain
``MPV_RIFE_*`` environment variables. See ``rife/config.py`` for the keys.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

package_root = Path(__file__).resolve().parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

import vapoursynth as vs  # noqa: E402

from rife.config import load_settings  # noqa: E402
from rife.vsfilter import rife_interpolate  # noqa: E402

core = vs.core

video_in = globals().get("video_in")
if video_in is None:
    raise RuntimeError("rife_video.py must run through mpv's vapoursynth filter")

user_data = globals().get("user_data", "") or ""
settings = load_settings(user_data)
core.num_threads = max(1, int(settings["vs_threads"]))

container_fps = float(globals().get("container_fps", 0.0) or 0.0)
fps_num = int(settings.get("fps_num") or 0)
fps_den = int(settings.get("fps_den") or 0)
if fps_num > 0 and fps_den > 0:
    source_fps = Fraction(fps_num, fps_den)
elif container_fps > 0:
    source_fps = Fraction(container_fps).limit_denominator(1000000)
else:
    source_fps = None

out_clip = rife_interpolate(
    video_in,
    source_fps=source_fps,
    user_data=user_data,
    matrix=str(settings["matrix"]),
)
out_clip.set_output(0)
