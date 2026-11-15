import os
import struct
import sys
import unittest
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rife.config import DEFAULTS, KEY_ALIASES, decode_user_data, load_settings


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        os.environ.pop("MPV_RIFE_MODEL", None)
        settings = load_settings(None, {})
        for key, value in DEFAULTS.items():
            if key == "model":
                continue
            self.assertEqual(settings[key], value)

    def test_overrides_win(self):
        settings = load_settings(None, {"model": "a.onnx", "backend": "ov", "sc": 0.5})
        self.assertEqual(settings["model"], "a.onnx")
        self.assertEqual(settings["backend"], "ov")
        self.assertEqual(settings["sc"], 0.5)

    def test_short_aliases(self):
        settings = load_settings(None, {"m": "x.onnx", "b": "ort", "f": 4, "g": 2})
        self.assertEqual(settings["model"], "x.onnx")
        self.assertEqual(settings["backend"], "ort")
        self.assertEqual(settings["factor_num"], 4)
        self.assertEqual(settings["factor_den"], 2)

    def test_alias_table(self):
        self.assertEqual(KEY_ALIASES["m"], "model")

    def test_decode_json(self):
        data = decode_user_data('{"model": "a.onnx", "pad": 64}')
        self.assertEqual(data["model"], "a.onnx")
        self.assertEqual(data["pad"], 64)

    def test_env(self):
        old = os.environ.get("MPV_RIFE_MODEL")
        os.environ["MPV_RIFE_MODEL"] = "env.onnx"
        try:
            settings = load_settings(None, {})
            self.assertEqual(settings["model"], "env.onnx")
        finally:
            if old is None:
                os.environ.pop("MPV_RIFE_MODEL", None)
            else:
                os.environ["MPV_RIFE_MODEL"] = old

    def test_user_data_beats_env(self):
        old = os.environ.get("MPV_RIFE_BACKEND")
        os.environ["MPV_RIFE_BACKEND"] = "ort"
        try:
            settings = load_settings('{"backend": "ov"}')
            self.assertEqual(settings["backend"], "ov")
        finally:
            if old is None:
                os.environ.pop("MPV_RIFE_BACKEND", None)
            else:
                os.environ["MPV_RIFE_BACKEND"] = old


if __name__ == "__main__":
    unittest.main()
