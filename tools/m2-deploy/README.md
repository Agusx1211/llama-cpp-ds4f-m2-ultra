# M2 Ultra release packaging

`release_manager.py` creates immutable, content-addressed-by-manifest release
directories for the single production `llama-server` process. It copies and
SHA-256 hashes the server binary, one complete server configuration, and every
static dashboard asset. It then atomically replaces the relative `current`
symlink and retains the prior target as `previous`.

The model files are deliberately not copied into each release. A 150+ GiB
model has one separately managed installation and its identity belongs in the
server configuration and readiness policy. This prevents a release operation
from starting or duplicating a second model process.

Example:

```sh
python3 tools/m2-deploy/release_manager.py install \
  --root /opt/llama-m2 \
  --release-id 2026-08-03-c6a7a0aaf \
  --server build-release/bin/llama-server \
  --config deployment/server.json \
  --dashboard tools/m2-dashboard/dist

python3 tools/m2-deploy/release_manager.py status --root /opt/llama-m2
python3 tools/m2-deploy/release_manager.py verify \
  --root /opt/llama-m2 --release-id 2026-08-03-c6a7a0aaf
python3 tools/m2-deploy/release_manager.py rollback --root /opt/llama-m2
```

Every source and installed artifact must be a regular file; symlinks and
unmanifested files are rejected. Installed files are read-only, the server
binary is read/execute-only, and release directories are read/execute-only.
This is application-level immutability under the deployment account, not a
macOS filesystem flag or protection against root.

The `current` replacement is one atomic rename on the target filesystem. The
`previous` pointer is updated first, so a crash cannot make `current` point to
an incomplete release. A process lock serializes install and rollback. Release
construction happens below `releases/.staging-*`; a complete verified tree is
renamed into its final name before either pointer changes.

This component does not install launchd jobs, start or stop a server, declare
readiness, or perform drain. Those remain separate Phase 9 gates.
