"""End-to-end parity: exported ONNX vs the original torch model.

Requires a Practical-RIFE checkout (model/warplayer.py) and an unpacked
model dir (flownet.pkl...). Skipped automatically when any of these or torch
are unavailable. Point RIFE_TEST_MODEL_DIR at a train_log dir to run it in a
fresh environment.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODEL_DIRS = [
    "tmp/models/rife-4.25/train_log",
    "tmp/models/rife-4.25-lite/train_log",
]

def _model_dir() -> str:
    env = os.environ.get("RIFE_TEST_MODEL_DIR")
    if env:
        return env
    for candidate in MODEL_DIRS:
        if os.path.exists(os.path.join(ROOT, candidate, "flownet.pkl")):
            return candidate
    return ""

MODEL_DIR = _model_dir()
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipIf(not MODEL_DIR, "no Practical-RIFE model weights available")
@unittest.skipIf(not HAS_TORCH, "torch missing")
class ExportParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2  # noqa
        import onnxruntime as ort
        import torch

        from rife.export_ifnet import export, load_rife

        cls.onnx_path = os.path.join(ROOT, "tmp", "rife-test-parity.onnx")
        export(os.path.join(ROOT, MODEL_DIR), cls.onnx_path)
        cls.torch_model = load_rife(os.path.join(ROOT, MODEL_DIR), None)

        demo = os.path.join(ROOT, "tmp", "Practical-RIFE", "demo")
        def read(path):
            img = cv2.imread(path).astype(np.float32) / 255.0
            return np.transpose(img, (2, 0, 1))

        cls.i0 = read(os.path.join(demo, "i0.png"))
        cls.i1 = read(os.path.join(demo, "i1.png"))

    def _pair(self, ih, iw):
        ph = ((ih - 1) // 128 + 1) * 128
        pw = ((iw - 1) // 128 + 1) * 128
        a0 = np.pad(self.i0, ((0, 0), (0, ph - ih), (0, pw - iw)))
        a1 = np.pad(self.i1, ((0, 0), (0, ph - ih), (0, pw - iw)))
        return np.concatenate((a0, a1), axis=0)[np.newaxis]

    def test_onnx_matches_torch(self):
        import onnxruntime as ort
        import torch

        pair = self._pair(*self.i0.shape[1:])
        with torch.no_grad():
            expected = self.torch_model(torch.from_numpy(pair), torch.ones(1, 1, 1, 1) * 0.5).numpy()

        sess = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"x": pair.astype(np.float32), "timestep": np.full((1, 1, 1, 1), 0.5, np.float32)})[0]
        mse = float(np.mean((out - expected) ** 2))
        self.assertLess(mse, 1e-7, f"MAE-based divergence: mse={mse}")


if __name__ == "__main__":
    unittest.main()
