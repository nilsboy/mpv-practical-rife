import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import vapoursynth as vs
    import onnx

    HAS_VS = True
except ImportError:
    HAS_VS = False

from blend_onnx import make_blend_onnx


def rgbs_format(core):
    rgbs = getattr(vs, "RGBS", None)
    if rgbs is None:
        return core.query_video_format(vs.ColorFamily.RGB, vs.SampleType.FLOAT, 32)
    return core.get_video_format(rgbs)


@unittest.skipIf(not HAS_VS, "vapoursynth or onnx missing")
class VsFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join(tempfile.mkdtemp(dir="/tmp/opencode"), "blend.onnx")
        make_blend_onnx(cls.model_path)
        cls.core = vs.core

    def build_src(self, values, fps=(24, 1), width=320, height=256):
        core = self.core
        fmt = rgbs_format(core)
        segments = [
            core.std.BlankClip(format=fmt, width=width, height=height, length=1, color=[v, v, v])
            for v in values
        ]
        return core.std.Splice(segments, mismatch=False)

    def test_frame_count_and_fps(self):
        from rife.vsfilter import rife_interpolate

        src = self.build_src([0.2, 0.4, 0.6])
        out = rife_interpolate(src, model=self.model_path, sc=0, pad=128)
        self.assertEqual(out.num_frames, 5)
        self.assertEqual(out.fps.numerator, 48)

    def test_values_interpolated(self):
        from rife.vsfilter import rife_interpolate

        core = self.core
        src = self.build_src([0.2, 0.4, 0.6])
        out = rife_interpolate(src, model=self.model_path, sc=0, pad=128)
        values = [float(np.asarray(f[1])[128, 160]) for f in out.frames()]
        self.assertAlmostEqual(values[0], 0.2, places=5)
        self.assertAlmostEqual(values[2], 0.4, places=5)
        self.assertAlmostEqual(values[4], 0.6, places=5)
        self.assertAlmostEqual(values[1], 0.3, places=5)
        self.assertAlmostEqual(values[3], 0.5, places=5)

    def test_duration_props(self):
        from rife.vsfilter import rife_interpolate

        src = self.build_src([0.2, 0.4, 0.6])
        out = rife_interpolate(src, model=self.model_path, sc=0, pad=128)
        for f in out.frames():
            self.assertEqual(f.props.get("_DurationNum"), 1)
            self.assertEqual(f.props.get("_DurationDen"), 48)

    def test_scene_change_hold(self):
        from rife.vsfilter import rife_interpolate

        src = self.build_src([0.2, 0.6, 0.7])
        out = rife_interpolate(src, model=self.model_path, sc=0.15, pad=128)
        values = [float(np.asarray(f[1])[128, 160]) for f in out.frames()]
        self.assertAlmostEqual(values[0], 0.2, places=5)
        self.assertAlmostEqual(values[1], 0.2, places=5, msg="scene change should hold frame 0")
        self.assertAlmostEqual(values[2], 0.6, places=5)
        self.assertAlmostEqual(values[3], 0.65, places=5)
        self.assertAlmostEqual(values[4], 0.7, places=5)

    def test_factor(self):
        from rife.vsfilter import rife_interpolate

        src = self.build_src([0.1, 0.5])
        out = rife_interpolate(src, model=self.model_path, factor_num=4, factor_den=1, sc=0, pad=128)
        self.assertEqual(out.num_frames, 5)
        self.assertEqual(out.fps.numerator, 96)
        values = [float(np.asarray(f[1])[128, 160]) for f in out.frames()]
        # blend model returns the pair average for every interpolated frame
        expected = [0.1, 0.3, 0.3, 0.3, 0.5]
        for got, want in zip(values, expected):
            self.assertAlmostEqual(got, want, places=5)

    def test_yuv_roundtrip(self):
        from rife.vsfilter import rife_interpolate

        core = self.core
        yuv = core.std.BlankClip(
            format=core.query_video_format(vs.ColorFamily.YUV, vs.SampleType.INTEGER, 8),
            width=320, height=256, length=2, fpsnum=24, fpsden=1,
            color=[100, 128, 128],
        )
        out = rife_interpolate(yuv, model=self.model_path, sc=0, pad=128)
        self.assertEqual(out.format.color_family, vs.ColorFamily.YUV)
        self.assertEqual(out.num_frames, 3)
        frame_values = [float(np.asarray(f[0])[128, 160]) for f in out.frames()]
        for value in frame_values:
            self.assertGreaterEqual(value, 0.0)
            self.assertGreaterEqual(255.0, value)

    def test_unpadded_base64_user_data(self):
        import base64

        from rife.config import decode_user_data

        raw = '{"model": "m.onnx", "pad": 64}'
        data = decode_user_data(base64.b64encode(raw.encode()).decode().rstrip("="))
        self.assertEqual(data["model"], "m.onnx")
        self.assertEqual(data["pad"], 64)


if __name__ == "__main__":
    unittest.main()
