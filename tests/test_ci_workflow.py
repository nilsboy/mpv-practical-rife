"""Contract tests for .github/workflows/build-snap.yml."""

import os
import unittest

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "build-snap.yml")
ARTIFACT_NAME = "mpv-practical-rife-snap"


class BuildSnapWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(WORKFLOW) as fobj:
            cls.doc = yaml.safe_load(fobj)

    def step_names(self, job):
        return [s.get("name") for s in self.doc["jobs"][job]["steps"] if isinstance(s, dict)]

    def test_triggers(self):
        on = self.doc[True] if True in self.doc else self.doc["on"]
        self.assertIn("push", on)
        self.assertIn("pull_request", on)
        self.assertIn("workflow_dispatch", on)
        self.assertEqual(on["push"]["tags"], ["v*"])
        self.assertIn("master", on["push"]["branches"])

    def test_build_job(self):
        job = self.doc["jobs"]["build"]
        self.assertEqual(job["runs-on"], "ubuntu-24.04")
        self.assertLessEqual(job["timeout-minutes"], 120)
        self.assertEqual(job["permissions"], {"contents": "read"})
        names = self.step_names("build")
        pack = names.index("Pack the snap")
        install = names.index("Install snapcraft and LXD")
        smoke = names.index("Smoke-test the snap (headless interpolation)")
        artifact = names.index("Upload snap artifact")
        self.assertLess(install, pack)
        self.assertLess(pack, smoke)
        self.assertLess(smoke, artifact)
        for name in names:
            name = name or ""
            self.assertNotIn("release", name.lower())

    def test_build_command_matches_verified_local_path(self):
        steps = self.doc["jobs"]["build"]["steps"]
        install_sh = next(s["run"] for s in steps if s.get("name") == "Install snapcraft and LXD")
        pack_sh = next(s["run"] for s in steps if s.get("name") == "Pack the snap")
        for cmd in ("snap install snapcraft --classic", "snap install lxd", "lxd init --auto"):
            self.assertIn(cmd, install_sh)
        self.assertIn("sudo snapcraft pack", pack_sh)

    def test_lxd_network_reachability_workarounds(self):
        # GitHub runners ship Docker (FORWARD policy DROP) and hand the
        # build instance a non-routable IPv6 address; both leave the LXD
        # instance without outbound network, failing `snapcraft pack` with
        # "A network related operation failed in a context of no network
        # access". Both fixes must come after `lxd init --auto`, which
        # creates lxdbr0.
        steps = self.doc["jobs"]["build"]["steps"]
        install_sh = next(s["run"] for s in steps if s.get("name") == "Install snapcraft and LXD")
        self.assertIn("iptables -P FORWARD ACCEPT", install_sh)
        self.assertIn("lxc network set lxdbr0 ipv6.address none", install_sh)
        self.assertLess(
            install_sh.index("lxd init --auto"),
            install_sh.index("lxc network set lxdbr0"),
        )

    def test_smoke_test_checks_interpolation(self):
        steps = self.doc["jobs"]["build"]["steps"]
        smoke = next(s["run"] for s in steps if s.get("name") == "Smoke-test the snap (headless interpolation)")
        self.assertIn("mpv-practical-rife.bench", smoke)
        self.assertIn("grep -q \"values\"", smoke)
        self.assertIn("snap install --dangerous --classic", smoke)

    def test_artifact_is_shared_with_release(self):
        build_steps = self.doc["jobs"]["build"]["steps"]
        release_steps = self.doc["jobs"]["release"]["steps"]
        up = next(s for s in build_steps if s.get("uses") == "actions/upload-artifact@v7")
        down = next(s for s in release_steps if s.get("uses") == "actions/download-artifact@v8")
        self.assertEqual(up["with"]["name"], ARTIFACT_NAME)
        self.assertEqual(down["with"]["name"], ARTIFACT_NAME)
        self.assertEqual(down["with"]["path"], "release-files/")

    def test_release_job(self):
        release = self.doc["jobs"]["release"]
        self.assertEqual(release["needs"], "build")
        self.assertEqual(release["permissions"], {"contents": "write"})
        # releases happen on pushes only: master pushes refresh the rolling
        # "latest" snapshot, v* tags get an exact release; PRs/dispatch don't
        self.assertEqual(release["if"], "github.event_name == 'push'")
        self.assertEqual(
            [s.get("name") for s in release["steps"] if isinstance(s, dict)],
            ["Download artifact", "Publish release"],
        )
        down = next(s for s in release["steps"] if s.get("uses") == "actions/download-artifact@v8")
        self.assertEqual(down["with"]["name"], ARTIFACT_NAME)
        self.assertEqual(down["with"]["path"], "release-files/")

    def test_release_step_env(self):
        publish = next(s for s in self.doc["jobs"]["release"]["steps"] if s.get("name") == "Publish release")
        self.assertEqual(publish["env"]["GH_TOKEN"], "${{ secrets.GITHUB_TOKEN }}")
        self.assertEqual(publish["env"]["REF_TYPE"], "${{ github.ref_type }}")

    def test_versioned_tag_release_path(self):
        script = self._publish_script()
        self.assertIn('[ "$REF_TYPE" = "tag" ]', script)
        self.assertIn('gh release create "$REF" release-files/*', script)
        self.assertIn("--generate-notes", script)
        self.assertIn("--clobber", script)

    def test_rolling_latest_snapshot_path(self):
        script = self._publish_script()
        self.assertIn("gh release delete latest", script)
        self.assertIn("--cleanup-tag", script)
        self.assertIn('gh release create latest release-files/*', script)
        self.assertIn("--target \"$SHA\"", script)
        self.assertIn("--prerelease", script)

    def _publish_script(self):
        return next(
            s["run"] for s in self.doc["jobs"]["release"]["steps"] if s.get("name") == "Publish release"
        )

    def test_no_release_step_in_build_job(self):
        for step in self.doc["jobs"]["build"]["steps"]:
            script = step.get("run", "") or ""
            self.assertNotIn("gh release", script)

    def _publish_script(self):
        return next(
            s["run"] for s in self.doc["jobs"]["release"]["steps"] if s.get("name") == "Publish release"
        )


if __name__ == "__main__":
    unittest.main()
