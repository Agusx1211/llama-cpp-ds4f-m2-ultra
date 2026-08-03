#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "release_manager.py"
SPEC = importlib.util.spec_from_file_location("release_manager", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
release_manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_manager)


class ReleaseManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "deployment"
        self.input = self.base / "input"
        self.input.mkdir()
        self.server = self.input / "llama-server"
        self.server.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.server.chmod(0o755)
        self.config = self.input / "server.json"
        self.config.write_text('{"host":"127.0.0.1"}\n', encoding="utf-8")
        self.dashboard = self.input / "dashboard"
        (self.dashboard / "assets").mkdir(parents=True)
        (self.dashboard / "index.html").write_text("<main>safe</main>\n", encoding="utf-8")
        (self.dashboard / "assets" / "app.js").write_text("export default 1;\n", encoding="utf-8")

    def tearDown(self) -> None:
        for path in sorted(self.base.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_symlink():
                continue
            if path.is_dir():
                path.chmod(0o755)
            elif path.exists():
                path.chmod(0o644)
        self.temporary.cleanup()

    def args(self, release_id: str) -> Namespace:
        return Namespace(
            root=self.root,
            release_id=release_id,
            server=self.server,
            config=self.config,
            dashboard=self.dashboard,
            created_utc="2026-08-03T00:00:00Z",
        )

    def install(self, release_id: str) -> dict[str, object]:
        return release_manager.install_release(self.args(release_id))

    def test_install_hashes_every_artifact_and_freezes_tree(self) -> None:
        manifest = self.install("release-001")
        self.assertEqual(os.readlink(self.root / "current"), "releases/release-001")
        self.assertFalse((self.root / "previous").exists())
        verified = release_manager.verify_release(self.root, "release-001")
        self.assertEqual(verified, manifest)
        roles = {item["role"] for item in manifest["artifacts"]}
        self.assertEqual(roles, {"server", "config", "dashboard"})
        release = self.root / "releases" / "release-001"
        self.assertEqual(stat.S_IMODE(release.stat().st_mode), 0o555)
        self.assertEqual(stat.S_IMODE((release / "bin" / "llama-server").stat().st_mode), 0o555)
        for item in manifest["artifacts"]:
            path = release / str(item["path"])
            self.assertEqual(item["sha256"], release_manager.sha256_file(path))

    def test_second_install_preserves_previous_and_rollback_swaps(self) -> None:
        self.install("release-001")
        self.server.write_bytes(b"#!/bin/sh\nexit 2\n")
        self.install("release-002")
        self.assertEqual(os.readlink(self.root / "current"), "releases/release-002")
        self.assertEqual(os.readlink(self.root / "previous"), "releases/release-001")
        current, previous = release_manager.rollback(self.root)
        self.assertEqual((current, previous), ("release-001", "release-002"))
        self.assertEqual(os.readlink(self.root / "current"), "releases/release-001")
        self.assertEqual(os.readlink(self.root / "previous"), "releases/release-002")

    def test_tamper_and_unmanifested_files_fail_verification(self) -> None:
        self.install("release-001")
        release = self.root / "releases" / "release-001"
        server = release / "bin" / "llama-server"
        server.chmod(0o755)
        server.write_bytes(b"tampered")
        server.chmod(0o555)
        with self.assertRaisesRegex(release_manager.ReleaseError, "hash mismatch|size mismatch"):
            release_manager.verify_release(self.root, "release-001")

        self.install("release-002")
        release = self.root / "releases" / "release-002"
        release.chmod(0o755)
        extra = release / "extra"
        extra.write_bytes(b"not in manifest")
        extra.chmod(0o444)
        release.chmod(0o555)
        with self.assertRaisesRegex(release_manager.ReleaseError, "unmanifested"):
            release_manager.verify_release(self.root, "release-002")

    def test_bad_input_does_not_change_active_pointer_or_leave_staging(self) -> None:
        self.install("release-001")
        (self.dashboard / "unsafe").symlink_to(self.dashboard / "index.html")
        with self.assertRaisesRegex(release_manager.ReleaseError, "symlink"):
            self.install("release-002")
        self.assertEqual(os.readlink(self.root / "current"), "releases/release-001")
        self.assertFalse((self.root / "previous").exists())
        self.assertEqual(
            [path.name for path in (self.root / "releases").iterdir()],
            ["release-001"],
        )

    def test_invalid_ids_duplicate_release_and_unsafe_pointer_are_rejected(self) -> None:
        for release_id in ("../escape", ".hidden", "bad/id", ""):
            with self.subTest(release_id=release_id):
                with self.assertRaises(release_manager.ReleaseError):
                    self.install(release_id)
        self.install("release-001")
        with self.assertRaisesRegex(release_manager.ReleaseError, "already exists"):
            self.install("release-001")
        (self.root / "current").unlink()
        (self.root / "current").symlink_to("../outside")
        with self.assertRaisesRegex(release_manager.ReleaseError, "unsafe target"):
            release_manager.status(self.root)

    def test_manifest_path_traversal_and_mode_changes_fail(self) -> None:
        self.install("release-001")
        release = self.root / "releases" / "release-001"
        manifest_path = release / "manifest.json"
        release.chmod(0o755)
        manifest_path.chmod(0o644)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"][0]["path"] = "../outside"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_path.chmod(0o444)
        release.chmod(0o555)
        with self.assertRaisesRegex(release_manager.ReleaseError, "unsafe"):
            release_manager.verify_release(self.root, "release-001")

        self.install("release-002")
        config = self.root / "releases" / "release-002" / "config" / "server.json"
        config.chmod(0o644)
        with self.assertRaisesRegex(release_manager.ReleaseError, "mode mismatch"):
            release_manager.verify_release(self.root, "release-002")

        self.install("release-003")
        release = self.root / "releases" / "release-003"
        dashboard = release / "dashboard"
        dashboard.chmod(0o755)
        with self.assertRaisesRegex(release_manager.ReleaseError, "directory mode"):
            release_manager.verify_release(self.root, "release-003")

    def test_status_verifies_all_releases(self) -> None:
        self.install("release-001")
        with self.assertRaisesRegex(release_manager.ReleaseError, "requires both"):
            release_manager.rollback(self.root)
        self.assertEqual(os.readlink(self.root / "current"), "releases/release-001")
        self.install("release-002")
        self.assertEqual(release_manager.status(self.root), {
            "current": "release-002",
            "previous": "release-001",
            "releases": ["release-001", "release-002"],
        })
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH), "status", "--root", str(self.root)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(completed.stdout), release_manager.status(self.root))


if __name__ == "__main__":
    unittest.main()
