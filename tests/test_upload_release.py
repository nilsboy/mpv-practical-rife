"""Tests for bin/upload-release (GitHub release upload helper)."""

import os
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPT = os.path.join(ROOT, "bin", "upload-release")


def run(args, expect=0):
    proc = subprocess.run(
        [SCRIPT, *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
    )
    if expect is not None:
        assert proc.returncode == expect, proc.stderr
    return proc


class UploadReleaseTests(unittest.TestCase):
    def test_help(self):
        out = run(["--help"]).stdout
        self.assertIn("--dry-run", out)

    def test_dry_run_default_tag_and_assets(self):
        out = run(["--dry-run", "--repo", "example/repo"]).stdout
        self.assertIn("gh release create v0.1.0", out)
        for asset in sorted(os.listdir(os.path.join(ROOT, "dist"))):
            self.assertIn(asset, out)
        self.assertIn("mpv-practical-rife_0.1.0_amd64.snap", out)
        self.assertNotIn("--clobber", out.split("#")[0])

    def test_dry_run_clobber_prints_upload(self):
        out = run(["--dry-run", "--repo", "example/repo", "--clobber"]).stdout
        self.assertIn("gh release upload", out)
        self.assertIn("--clobber", out)

    def test_explicit_tag_title_notes(self):
        out = run(
            [
                "--dry-run",
                "v9.9.9",
                "--repo",
                "example/repo",
                "--title",
                "My Release",
                "--notes",
                "hello",
            ]
        ).stdout
        self.assertIn("gh release create v9.9.9", out)
        self.assertIn("--title My\\ Release", out)
        self.assertIn("--notes hello", out)

    def test_no_repo_fails(self):
        proc = run(["--dry-run"], expect=1)
        self.assertIn("cannot determine repo", proc.stderr)

    def test_unknown_option_fails(self):
        proc = run(["--wat"], expect=1)
        self.assertIn("unknown option", proc.stderr)

    def test_notes_file(self):
        notes = os.path.join(ROOT, "tmp", "release-notes-test.txt")
        os.makedirs(os.path.dirname(notes), exist_ok=True)
        with open(notes, "w") as fobj:
            fobj.write("some notes")
        out = run(["--dry-run", "--repo", "example/repo", "--notes", f"@{notes}"]).stdout
        self.assertIn("--notes some\\ notes", out)

    def test_repo_from_origin_remote(self):
        # point a throwaway repository at a fake origin
        test_repo = os.path.join(ROOT, "tmp", "upload-release-origin")
        subprocess.run(["rm", "-rf", test_repo], check=True)
        os.makedirs(test_repo, exist_ok=True)
        subprocess.run(["git", "init", "-q", test_repo], check=True)
        subprocess.run(
            ["git", "-C", test_repo, "remote", "add", "origin",
             "https://github.com/owner/test-repo.git"],
            check=True,
        )
        proc = subprocess.run(
            [SCRIPT, "--dry-run"],
            capture_output=True,
            text=True,
            cwd=test_repo,
            env={**os.environ},
            timeout=60,
        )
        self.assertIn("--repo owner/test-repo", proc.stdout)
        self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
