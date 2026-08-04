#!/usr/bin/env python3
"""Fail-closed validation for the M2 Ultra trusted-subdomain gateway."""

from __future__ import annotations

import argparse
import ipaddress
import os
import pathlib
import re
import shutil
import socket
import stat
import subprocess
import sys
import urllib.parse

ENV_NAMES = (
    "M2_GATEWAY_LOW_HOST",
    "M2_GATEWAY_NORMAL_HOST",
    "M2_GATEWAY_FAST_HOST",
    "M2_GATEWAY_ADMIN_HOST",
)
API_KEY_ENV_NAMES = (
    "M2_GATEWAY_LOW_API_KEY",
    "M2_GATEWAY_NORMAL_API_KEY",
    "M2_GATEWAY_FAST_API_KEY",
    "M2_GATEWAY_ADMIN_API_KEY",
)
DNS_NAME = re.compile(
    r"(?=^.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
URL_SAFE_SECRET = re.compile(r"[A-Za-z0-9_-]{32,256}")
ACME_EMAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+")
PINNED_DASHBOARD_ENV = "M2_GATEWAY_PINNED_DASHBOARD_ROOT"
LAUNCH_GUARD_ENV = "M2_GATEWAY_LAUNCH_GUARD"


class ValidationError(RuntimeError):
    pass


def mutable_by_untrusted(
    metadata: os.stat_result, *, allow_root_sticky_directory: bool = False
) -> bool:
    if not metadata.st_mode & 0o022:
        return False
    return not (
        allow_root_sticky_directory
        and metadata.st_uid == 0
        and stat.S_ISDIR(metadata.st_mode)
        and metadata.st_mode & stat.S_ISVTX
    )


def required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValidationError(f"{name} is required")
    if any(character in value for character in "\r\n\0"):
        raise ValidationError(f"{name} contains a forbidden control character")
    return value


def validate_host(name: str, value: str, test_mode: bool) -> tuple[str, int | None]:
    candidate = value if "://" in value else f"https://{value}"
    parsed = urllib.parse.urlsplit(candidate)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValidationError(f"{name} must be an HTTPS DNS site address")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValidationError(f"{name} must not contain a path, query, or fragment")
    host = parsed.hostname.lower()
    if not DNS_NAME.fullmatch(host):
        raise ValidationError(f"{name} must contain a fully-qualified DNS name")
    if not test_mode and parsed.port is not None:
        raise ValidationError(f"{name} must use the production HTTPS port")
    if not test_mode and host.endswith((".localhost", ".test")):
        raise ValidationError(f"{name} must not use a reserved test domain")
    return host, parsed.port


def validate_backend(value: str) -> None:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http" or not parsed.hostname or parsed.port is None:
        raise ValidationError(
            "M2_GATEWAY_BACKEND must be an explicit loopback HTTP URL with a port"
        )
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as error:
        raise ValidationError(
            "M2_GATEWAY_BACKEND must use a literal loopback address"
        ) from error
    if (
        not address.is_loopback
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
    ):
        raise ValidationError(
            "M2_GATEWAY_BACKEND must use only a literal loopback origin"
        )
    if parsed.query or parsed.fragment:
        raise ValidationError("M2_GATEWAY_BACKEND must not contain a query or fragment")


def validate_secrets(trusted_token: str, api_keys: list[str]) -> None:
    if not URL_SAFE_SECRET.fullmatch(trusted_token):
        raise ValidationError(
            "M2_GATEWAY_TRUSTED_TOKEN must contain 32 to 256 URL-safe token characters"
        )
    for name, api_key in zip(API_KEY_ENV_NAMES, api_keys):
        if not URL_SAFE_SECRET.fullmatch(api_key):
            raise ValidationError(
                f"{name} must contain 32 to 256 URL-safe token characters"
            )
    if len(set(api_keys)) != len(api_keys):
        raise ValidationError(
            "the low, normal, fast, and admin API keys must be distinct"
        )
    if trusted_token in api_keys:
        raise ValidationError(
            "the trusted scheduling token must be distinct from every client API key"
        )


def validate_tls(acme_email: str, tls_mode: str, test_mode: bool) -> None:
    if not ACME_EMAIL.fullmatch(acme_email) or ".." in acme_email:
        raise ValidationError(
            "M2_GATEWAY_ACME_EMAIL must be a valid single-token email address"
        )
    if test_mode:
        if tls_mode not in ("internal", acme_email):
            raise ValidationError("test TLS mode must be internal or the ACME email")
    elif tls_mode != acme_email:
        raise ValidationError(
            "production TLS mode must equal the ACME email; internal TLS is test-only"
        )


def validate_dashboard(
    root: pathlib.Path, *, trusted_owner_uids: set[int] | None = None
) -> pathlib.Path:
    if not root.is_absolute():
        raise ValidationError("M2_GATEWAY_DASHBOARD_ROOT must be an absolute directory")
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise ValidationError("M2_GATEWAY_DASHBOARD_ROOT cannot be resolved") from error
    if not resolved.is_dir():
        raise ValidationError("M2_GATEWAY_DASHBOARD_ROOT must resolve to a directory")
    root_metadata = resolved.stat()
    if mutable_by_untrusted(root_metadata):
        raise ValidationError("dashboard root is group/world writable")
    if (
        trusted_owner_uids is not None
        and root_metadata.st_uid not in trusted_owner_uids
    ):
        raise ValidationError("canonical dashboard root has an untrusted owner")

    if trusted_owner_uids is not None:
        current = resolved
        while True:
            metadata = current.stat()
            if metadata.st_uid not in trusted_owner_uids:
                raise ValidationError(
                    f"canonical dashboard ancestor has an untrusted owner: {current}"
                )
            if mutable_by_untrusted(
                metadata, allow_root_sticky_directory=current != resolved
            ):
                raise ValidationError(
                    f"canonical dashboard ancestor is group/world writable: {current}"
                )
            if current.parent == current:
                break
            current = current.parent

    files = 0
    total_bytes = 0
    for entry in resolved.rglob("*"):
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValidationError(f"dashboard asset symlink is forbidden: {entry}")
        if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise ValidationError(
                f"dashboard asset must be a regular file or directory: {entry}"
            )
        if mutable_by_untrusted(metadata):
            raise ValidationError(f"dashboard asset is group/world writable: {entry}")
        if trusted_owner_uids is not None and metadata.st_uid not in trusted_owner_uids:
            raise ValidationError(f"dashboard asset has an untrusted owner: {entry}")
        if stat.S_ISREG(metadata.st_mode):
            files += 1
            total_bytes += metadata.st_size
            if files > 2048 or total_bytes > 64 * 1024 * 1024:
                raise ValidationError("dashboard tree exceeds the static asset budget")
    index = resolved / "index.html"
    if not index.is_file() or index.is_symlink():
        raise ValidationError(
            "M2_GATEWAY_DASHBOARD_ROOT must contain a regular index.html"
        )
    return resolved


def validate_runtime_file(
    path: pathlib.Path,
    label: str,
    trusted_owner_uids: set[int],
    *,
    executable: bool = False,
) -> pathlib.Path:
    if not path.is_absolute():
        raise ValidationError(f"{label} must resolve to an absolute path")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise ValidationError(f"{label} cannot be resolved") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValidationError(f"{label} must resolve to a regular file")
    if metadata.st_uid not in trusted_owner_uids or mutable_by_untrusted(metadata):
        raise ValidationError(
            f"{label} must have a trusted owner and no group/world writes"
        )
    if executable and not metadata.st_mode & 0o111:
        raise ValidationError(f"{label} is not executable")

    current = resolved.parent
    while True:
        ancestor = current.stat()
        if ancestor.st_uid not in trusted_owner_uids or mutable_by_untrusted(
            ancestor, allow_root_sticky_directory=True
        ):
            raise ValidationError(
                f"{label} ancestor is mutable by an untrusted account: {current}"
            )
        if current.parent == current:
            break
        current = current.parent
    return resolved


def resolve_hosts(hosts: list[str], expected_addresses: set[str]) -> None:
    for host in hosts:
        try:
            answers = {
                item[4][0]
                for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            }
        except socket.gaierror as error:
            raise ValidationError(
                f"DNS resolution failed for {host}: {error}"
            ) from error
        normalized = {str(ipaddress.ip_address(answer)) for answer in answers}
        if not normalized:
            raise ValidationError(f"DNS returned no addresses for {host}")
        if expected_addresses and not normalized.issubset(expected_addresses):
            raise ValidationError(
                f"DNS for {host} returned {sorted(normalized)}, outside expected {sorted(expected_addresses)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caddy", default="caddy")
    parser.add_argument(
        "--caddyfile",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("Caddyfile"),
    )
    parser.add_argument("--expected-address", action="append", default=[])
    parser.add_argument("--skip-dns", action="store_true")
    parser.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--run",
        action="store_true",
        help="replace this validator with Caddy after pinning validated paths",
    )
    args = parser.parse_args()

    try:
        trusted_owner_uids = {0, os.geteuid()} if args.test_mode else {0}
        caddy_candidate = shutil.which(args.caddy)
        if caddy_candidate is None:
            raise ValidationError(f"Caddy executable was not found: {args.caddy}")
        caddy_executable = validate_runtime_file(
            pathlib.Path(caddy_candidate),
            "Caddy executable",
            trusted_owner_uids,
            executable=True,
        )
        caddyfile = validate_runtime_file(
            args.caddyfile, "Caddyfile", trusted_owner_uids
        )
        sites = [
            validate_host(name, required_environment(name), args.test_mode)
            for name in ENV_NAMES
        ]
        hosts = [site[0] for site in sites]
        if len(set(hosts)) != len(hosts):
            raise ValidationError(
                "the low, normal, fast, and admin DNS names must be distinct"
            )

        validate_backend(required_environment("M2_GATEWAY_BACKEND"))
        token = required_environment("M2_GATEWAY_TRUSTED_TOKEN")
        api_keys = [required_environment(name) for name in API_KEY_ENV_NAMES]
        validate_secrets(token, api_keys)
        acme_email = required_environment("M2_GATEWAY_ACME_EMAIL")
        validate_tls(
            acme_email, required_environment("M2_GATEWAY_TLS_MODE"), args.test_mode
        )
        dashboard = pathlib.Path(required_environment("M2_GATEWAY_DASHBOARD_ROOT"))
        if os.environ.get(PINNED_DASHBOARD_ENV) or os.environ.get(LAUNCH_GUARD_ENV):
            raise ValidationError(
                "launcher-owned gateway environment must not be supplied"
            )
        resolved_dashboard = validate_dashboard(
            dashboard,
            trusted_owner_uids=trusted_owner_uids,
        )

        runtime_environment = os.environ.copy()
        runtime_environment[PINNED_DASHBOARD_ENV] = str(resolved_dashboard)
        runtime_environment[LAUNCH_GUARD_ENV] = "0s"

        expected = {str(ipaddress.ip_address(value)) for value in args.expected_address}
        if not args.skip_dns:
            resolve_hosts(hosts, expected)

        completed = subprocess.run(
            [
                str(caddy_executable),
                "validate",
                "--config",
                str(caddyfile),
                "--adapter",
                "caddyfile",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=runtime_environment,
        )
        if completed.returncode != 0:
            raise ValidationError(
                f"caddy rejected the gateway configuration:\n{completed.stderr.strip()}"
            )
    except (OSError, ValueError, ValidationError) as error:
        print(f"gateway validation failed: {error}", file=sys.stderr)
        return 1

    print("gateway validation passed")
    if args.run:
        sys.stdout.flush()
        try:
            os.execve(
                str(caddy_executable),
                [
                    str(caddy_executable),
                    "run",
                    "--config",
                    str(caddyfile),
                    "--adapter",
                    "caddyfile",
                ],
                runtime_environment,
            )
        except OSError as error:
            print(f"gateway launch failed: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
