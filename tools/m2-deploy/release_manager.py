#!/usr/bin/env python3
"""Build, verify, activate, and roll back immutable M2 server releases."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Iterator


FORMAT_VERSION = 1
RELEASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
POINTERS = ("current", "previous")


class ReleaseError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_release_id(release_id: str) -> None:
    if not RELEASE_ID_RE.fullmatch(release_id):
        raise ReleaseError(f"invalid release id: {release_id!r}")


def require_regular_source(path: Path, role: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ReleaseError(f"missing {role}: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ReleaseError(f"{role} must be a regular non-symlink file: {path}")


def collect_dashboard_files(root: Path) -> list[Path]:
    try:
        mode = root.lstat().st_mode
    except FileNotFoundError as exc:
        raise ReleaseError(f"missing dashboard directory: {root}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ReleaseError(f"dashboard must be a non-symlink directory: {root}")

    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in dirnames:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ReleaseError(f"dashboard symlink is not permitted: {candidate}")
        for name in filenames:
            candidate = directory_path / name
            require_regular_source(candidate, "dashboard asset")
            files.append(candidate)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ReleaseError("dashboard directory is empty")
    if not (root / "index.html").is_file() or (root / "index.html").is_symlink():
        raise ReleaseError("dashboard must contain a regular index.html")
    return files


def copy_file(source: Path, destination: Path, executable: bool) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    mode = 0o555 if executable else 0o444
    os.chmod(destination, mode)
    return {
        "path": destination.as_posix(),
        "size": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "mode": f"{mode:04o}",
    }


def make_manifest(
        release_id: str,
        created_utc: str,
        artifacts: list[dict[str, object]]) -> dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "release_id": release_id,
        "created_utc": created_utc,
        "artifacts": sorted(artifacts, key=lambda item: str(item["path"])),
    }


def parse_created_utc(value: str | None) -> str:
    if value is None:
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseError(f"invalid UTC timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ReleaseError("created timestamp must include the UTC offset")
    return parsed.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def initialize_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    releases = root / "releases"
    releases.mkdir(exist_ok=True)
    if root.is_symlink() or releases.is_symlink():
        raise ReleaseError("release root and releases directory must not be symlinks")
    return releases


@contextlib.contextmanager
def release_lock(root: Path) -> Iterator[None]:
    initialize_root(root)
    lock_path = root / ".release.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def pointer_target(root: Path, name: str) -> str | None:
    pointer = root / name
    if not pointer.exists() and not pointer.is_symlink():
        return None
    if not pointer.is_symlink():
        raise ReleaseError(f"{name} is not a symlink")
    target = os.readlink(pointer)
    parts = Path(target).parts
    if len(parts) != 2 or parts[0] != "releases":
        raise ReleaseError(f"{name} has unsafe target: {target!r}")
    validate_release_id(parts[1])
    if not (root / target).is_dir():
        raise ReleaseError(f"{name} target does not exist: {target!r}")
    return target


def replace_pointer(root: Path, name: str, target: str | None) -> None:
    pointer = root / name
    temporary = root / f".{name}.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        if target is None:
            pointer.unlink(missing_ok=True)
        else:
            os.symlink(target, temporary)
            os.replace(temporary, pointer)
        fsync_directory(root)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(release: Path) -> dict[str, object]:
    manifest_path = release / "manifest.json"
    require_regular_source(manifest_path, "release manifest")
    if stat.S_IMODE(manifest_path.stat().st_mode) != 0o444:
        raise ReleaseError(f"manifest mode mismatch: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReleaseError(f"invalid manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != FORMAT_VERSION:
        raise ReleaseError(f"unsupported manifest format: {manifest_path}")
    return manifest


def verify_release(root: Path, release_id: str) -> dict[str, object]:
    validate_release_id(release_id)
    release = root / "releases" / release_id
    if release.is_symlink() or not release.is_dir():
        raise ReleaseError(f"release does not exist: {release_id}")
    manifest = load_manifest(release)
    if manifest.get("release_id") != release_id:
        raise ReleaseError("manifest release id mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ReleaseError("manifest has no artifacts")

    expected_paths: set[str] = set()
    roles: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            raise ReleaseError("manifest artifact is not an object")
        relative = item.get("path")
        role = item.get("role")
        if not isinstance(relative, str) or not isinstance(role, str):
            raise ReleaseError("manifest artifact path/role is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative in expected_paths:
            raise ReleaseError(f"unsafe or duplicate manifest path: {relative!r}")
        expected_paths.add(relative)
        roles.add(role)
        artifact = release / relative_path
        require_regular_source(artifact, "release artifact")
        actual_mode = stat.S_IMODE(artifact.stat().st_mode)
        if item.get("size") != artifact.stat().st_size:
            raise ReleaseError(f"size mismatch: {relative}")
        if item.get("sha256") != sha256_file(artifact):
            raise ReleaseError(f"hash mismatch: {relative}")
        if item.get("mode") != f"{actual_mode:04o}" or actual_mode not in (0o444, 0o555):
            raise ReleaseError(f"mode mismatch: {relative}")

    if not {"server", "config", "dashboard"}.issubset(roles):
        raise ReleaseError("manifest is missing a required artifact role")
    actual_paths: set[str] = set()
    for directory, dirnames, filenames in os.walk(release, followlinks=False):
        directory_path = Path(directory)
        if directory_path.is_symlink() or stat.S_IMODE(directory_path.stat().st_mode) != 0o555:
            raise ReleaseError(f"release directory mode/type mismatch: {directory_path}")
        for name in dirnames:
            child = directory_path / name
            if child.is_symlink():
                raise ReleaseError(f"release directory symlink is not permitted: {child}")
        for name in filenames:
            child = directory_path / name
            require_regular_source(child, "release file")
            if child != release / "manifest.json":
                actual_paths.add(child.relative_to(release).as_posix())
    if actual_paths != expected_paths:
        raise ReleaseError("release contains unmanifested or missing files")
    return manifest


def install_release(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve()
    validate_release_id(args.release_id)
    # Preserve the final path component for lstat-based symlink rejection.
    server = args.server.absolute()
    config = args.config.absolute()
    dashboard = args.dashboard.absolute()
    require_regular_source(server, "server binary")
    require_regular_source(config, "server config")
    if server.stat().st_size == 0 or config.stat().st_size == 0:
        raise ReleaseError("server binary and config must be non-empty")
    dashboard_files = collect_dashboard_files(dashboard)
    created_utc = parse_created_utc(args.created_utc)

    with release_lock(root):
        releases = initialize_root(root)
        destination = releases / args.release_id
        if destination.exists() or destination.is_symlink():
            raise ReleaseError(f"release already exists: {args.release_id}")
        staging = releases / f".staging-{args.release_id}-{uuid.uuid4().hex}"
        staging.mkdir(mode=0o700)
        published = False
        try:
            artifacts: list[dict[str, object]] = []
            server_item = copy_file(server, staging / "bin" / "llama-server", True)
            server_item["role"] = "server"
            artifacts.append(server_item)
            config_item = copy_file(config, staging / "config" / config.name, False)
            config_item["role"] = "config"
            artifacts.append(config_item)
            for source in dashboard_files:
                relative = source.relative_to(dashboard)
                item = copy_file(source, staging / "dashboard" / relative, False)
                item["role"] = "dashboard"
                artifacts.append(item)

            for item in artifacts:
                item["path"] = Path(str(item["path"])).relative_to(staging).as_posix()
            manifest = make_manifest(args.release_id, created_utc, artifacts)
            manifest_path = staging / "manifest.json"
            with manifest_path.open("x", encoding="utf-8") as output:
                json.dump(manifest, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.chmod(manifest_path, 0o444)
            for directory in sorted(
                    (path for path in staging.rglob("*") if path.is_dir()),
                    key=lambda path: len(path.parts), reverse=True):
                os.chmod(directory, 0o555)
            os.chmod(staging, 0o555)
            fsync_directory(releases)
            os.rename(staging, destination)
            published = True
            fsync_directory(releases)
            verify_release(root, args.release_id)

            old_current = pointer_target(root, "current")
            if old_current is not None:
                replace_pointer(root, "previous", old_current)
            replace_pointer(root, "current", f"releases/{args.release_id}")
            return manifest
        finally:
            if not published and staging.exists():
                os.chmod(staging, 0o700)
                shutil.rmtree(staging)


def rollback(root: Path) -> tuple[str, str]:
    root = root.resolve()
    with release_lock(root):
        current = pointer_target(root, "current")
        previous = pointer_target(root, "previous")
        if current is None or previous is None:
            raise ReleaseError("rollback requires both current and previous releases")
        current_id = Path(current).name
        previous_id = Path(previous).name
        verify_release(root, current_id)
        verify_release(root, previous_id)
        replace_pointer(root, "previous", current)
        replace_pointer(root, "current", previous)
        return previous_id, current_id


def status(root: Path) -> dict[str, object]:
    root = root.resolve()
    releases = initialize_root(root)
    result: dict[str, object] = {
        "current": None,
        "previous": None,
        "releases": [],
    }
    for pointer in POINTERS:
        target = pointer_target(root, pointer)
        result[pointer] = Path(target).name if target else None
    release_ids = sorted(
        path.name for path in releases.iterdir()
        if path.is_dir() and not path.is_symlink() and not path.name.startswith(".staging-")
    )
    for release_id in release_ids:
        verify_release(root, release_id)
    result["releases"] = release_ids
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("install", help="publish and activate one release")
    install.add_argument("--root", type=Path, required=True)
    install.add_argument("--release-id", required=True)
    install.add_argument("--server", type=Path, required=True)
    install.add_argument("--config", type=Path, required=True)
    install.add_argument("--dashboard", type=Path, required=True)
    install.add_argument("--created-utc")

    verify = subparsers.add_parser("verify", help="verify one installed release")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--release-id", required=True)

    rollback_parser = subparsers.add_parser("rollback", help="atomically reactivate previous")
    rollback_parser.add_argument("--root", type=Path, required=True)

    status_parser = subparsers.add_parser("status", help="verify and print release state")
    status_parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "install":
            result = install_release(args)
        elif args.command == "verify":
            result = verify_release(args.root.resolve(), args.release_id)
        elif args.command == "rollback":
            current, previous = rollback(args.root)
            result = {"current": current, "previous": previous}
        else:
            result = status(args.root)
    except (OSError, ReleaseError) as exc:
        print(f"release-manager: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
