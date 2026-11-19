"""Contract tests for snapcraft.yaml and snap/wrappers."""

import os
import re
import unittest

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SNAPCRAFT_YAML = os.path.join(ROOT, "snapcraft.yaml")
WRAPPERS = ("bench", "rife-mpv")


def load_snapcraft():
    with open(SNAPCRAFT_YAML) as fobj:
        return yaml.safe_load(fobj)


def wrapper_source(name):
    with open(os.path.join(ROOT, "snap", "wrappers", name)) as fobj:
        return fobj.read()


def ld_library_path_line(source):
    lines = [ln for ln in source.splitlines() if ln.startswith("export LD_LIBRARY_PATH=")]
    return lines[0] if lines else ""


class VapoursynthRuntimeTests(unittest.TestCase):
    """The VS core links libzimg; classic snaps resolve the rest from the
    host, but libzimg2 is not on base installs (it bit CI), so it must be
    staged into the snap."""

    def test_libzimg_is_staged(self):
        part = load_snapcraft()["parts"]["vapoursynth"]
        self.assertIn("libzimg2", part.get("stage-packages", []))

    def test_libzimg_not_staged_via_dev_package(self):
        part = load_snapcraft()["parts"]["vapoursynth"]
        for pkg in part.get("stage-packages", []):
            self.assertNotIn("libzimg-dev", pkg)

    def test_wrapper_prepends_staged_lib_dir(self):
        for name in WRAPPERS:
            line = ld_library_path_line(wrapper_source(name))
            self.assertIn("${SNAP}/usr/local/lib", line, name)
            self.assertIn("${SNAP}/usr/lib/x86_64-linux-gnu", line, name)
            self.assertIn("${SNAP}/lib/x86_64-linux-gnu", line, name)

    def test_wrapper_ld_path_order(self):
        for name in WRAPPERS:
            line = ld_library_path_line(wrapper_source(name))
            positions = [
                line.index("${SNAP}/usr/local/lib"),
                line.index("${SNAP}/usr/lib/x86_64-linux-gnu"),
                line.index("${SNAP}/lib/x86_64-linux-gnu"),
            ]
            self.assertEqual(positions, sorted(positions), name)

    def test_wrapper_ld_path_preserves_prior_env(self):
        for name in WRAPPERS:
            self.assertRegex(wrapper_source(name), re.escape("${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"))


class RuntimePathTests(unittest.TestCase):
    def test_wrappers_use_snap_local_paths(self):
        for name in WRAPPERS:
            source = wrapper_source(name)
            self.assertIn("${SNAP}/lib/python3.12/site-packages", source, name)

    def test_bench_defaults_to_bundled_model(self):
        self.assertIn("${SNAP}/models/rife-4.25-lite-s1.0.onnx", wrapper_source("bench"))


if __name__ == "__main__":
    unittest.main()
