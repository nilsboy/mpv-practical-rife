"""Executable tests for the release job's publish script (stubbed gh CLI)."""

import os
import subprocess
import unittest

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "build-snap.yml")


def publish_script() -> str:
    with open(WORKFLOW) as fobj:
        doc = yaml.safe_load(fobj)
    return next(
        s["run"] for s in doc["jobs"]["release"]["steps"] if s.get("name") == "Publish release"
    )


def run_publish(ref_type: str, ref: str) -> list[list[str]]:
    stub_dir = os.path.join(ROOT, "tmp", "gh-stub")
    os.makedirs(stub_dir, exist_ok=True)
    gh_log = os.path.join(stub_dir, "gh-calls.log")
    if os.path.exists(gh_log):
        os.remove(gh_log)
    gh_stub = os.path.join(stub_dir, "gh")
    with open(gh_stub, "w") as fobj:
        fobj.write(
            "#!/bin/sh\n"
            'printf "%s\\n" "$*" >> "$GH_LOG"\n'
            "exit 0\n"
        )
    os.chmod(gh_stub, 0o755)
    env = {
        **os.environ,
        "PATH": f"{stub_dir}:{os.environ['PATH']}",
        "GH_LOG": gh_log,
        "GH_TOKEN": "stub-token",
        "GITHUB_REPOSITORY": "example/repo",
        "REF": ref,
        "REF_TYPE": ref_type,
        "SHA": "deadbeef123456",
    }
    proc = subprocess.run(
        ["bash", "-c", publish_script()],
        capture_output=True,
        text=True,
        env=env,
        cwd=stub_dir,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    with open(gh_log) as fobj:
        return [line.removeprefix("gh ").split() for line in fobj.read().splitlines()]


class ReleasePublishScriptTests(unittest.TestCase):
    def test_tag_push_creates_exact_release(self):
        calls = run_publish("tag", "v0.1.0")
        create = calls[0]
        self.assertEqual(create[0], "release")
        self.assertEqual(create[1], "create")
        self.assertIn("v0.1.0", create)
        self.assertIn("--generate-notes", create)
        self.assertIn("release-files/", " ".join(create))
        self.assertFalse(any("delete" in call for call in calls))

    def test_branch_push_replaces_rolling_latest(self):
        calls = run_publish("branch", "master")
        self.assertEqual(calls[0][:2], ["release", "delete"])
        self.assertIn("latest", calls[0])
        self.assertIn("--cleanup-tag", calls[0])
        create = calls[1]
        self.assertEqual(create[:2], ["release", "create"])
        self.assertIn("latest", create)
        self.assertIn("--prerelease", create)
        self.assertIn("--target", create)
        self.assertIn("deadbeef123456", create)
        self.assertIn("--notes", create)
        self.assertNotIn("--generate-notes", create)

    def test_tag_push_creates_with_clobber_fallback(self):
        script = publish_script()
        self.assertIn("|| gh release upload", script)


if __name__ == "__main__":
    unittest.main()
