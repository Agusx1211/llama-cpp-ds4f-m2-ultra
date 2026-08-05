# Dormant 128-token resync journal

`server_capture::resync_journal` is a bounded prototype for rebuilding a
committed token suffix after a control-plane resynchronization.  It is not
wired into `llama-server`, `capture_store`, or any inference loop.  A future
owner may call it from a background/control thread only after the capture
store reports the observation as committed.

## Exact contract

- The replay state is an ordered suffix of at most 128 **accepted** token IDs.
  The suffix is exposed by `inspect().replay_tokens` in replay order, with
  contiguous `replay_sequence` and `token_position` values.
- A batch contains one `resync_capture_commit` identity and an accepted prefix.
  It may end with one rejected proposal.  The rejected token is retained in
  one separate `rejection_boundary` record for diagnostics; it never enters
  `replay_tokens` and does not consume one of the 128 accepted slots.  Thus a
  single call may contain 128 accepted IDs plus one final rejection.
- Physical `event_sequence` values are monotonic for accepted and rejected
  inputs.  `next_replay_sequence` and `next_token_position` are persisted
  watermarks.  The accepted ring evicts its oldest ID on wrap while those
  watermarks continue forward; integer wrap, gaps, and token-position overflow
  are rejected rather than silently rebased.
- A zero-length batch records a committed observation watermark without
  changing token state.  Retrying an identical retained commit is an explicit
  `duplicate_commit`; the same identity with different data is a
  `commit_conflict`.  An older identity is `out_of_order_commit`, including
  after its events have been evicted, because the monotonic commit watermark
  survives retention.
- The commit identity is historical provenance (`generation`, shard sequence,
  record index, and observation sequence).  The journal does not open or
  require the shard to still exist; capture retention may delete it after the
  journal commit.

## Persistence and privacy

Persistence is off by default.  In-memory mode keeps exact IDs inside the
process and has no restart recovery.  Enabling persistence requires both an
explicit sensitive-token opt-in and a private root; requesting
`require_private_root=false` is rejected.  The sidecar is written as
`resync.journal` with mode 0600 below a 0700 owner-only root.  A held
`.resync.journal.lock` flock gives one journal owner per root across threads,
processes, and restarts; the kernel releases it if the owner dies.  No request text,
request IDs, or prompt content is stored.  Exact token IDs are still sensitive
because a tokenizer can make them meaningful, so deployments should prefer
the in-memory mode unless durable replay is required.  These IDs never enter
the capture-store background shards, whose compact privacy policy remains
unchanged.

The root walk uses the same absolute, no-follow, owner, and private-mode
invariants as `server-capture-store` (that helper is currently translation-unit
private, so the invariants are mirrored until a shared authority can be
extracted without changing the capture ABI).  Symlinked or group/world-writable
roots and journal files fail closed.  The journal object must outlive all
concurrent calls; destruction requires external call quiescence.

## Crash and corruption behavior

Every persisted state is serialized with a versioned `SCRJNL01` header, fixed
event framing, and a SHA-256 footer over all preceding bytes.  Publication is
write-temp, `fsync` the file, atomic `renameat`, then `fsync` the root
directory.  A crash before rename leaves the previous complete image; an
orphan `.resync.journal.tmp` is ignored on restart.  If the directory fence is
uncertain, the new image is still accepted only after checksum validation and
the caller receives `commit_uncertain`.

The footer uses the standard SHA-256 constants and padding.  Earlier dormant
prototype builds emitted non-standard digests for `SCRJNL01` and the related
`SCAPMF01`/`SCAPSH01` capture-store sidecars; those images fail checksum
validation and are intentionally unsupported rather than silently migrated.
Neither sidecar has live `llama-server` wiring, so replacing a pre-fix root is
safe for the current product surface and a future format migration can be
designed before activation.

Malformed headers, unsupported future versions, impossible counts or sequences,
trailing bytes, truncation, and checksum failures return an explicit status.
Recovery never exposes a
partial suffix or silently repairs a corrupt image; append remains disabled
until the owner replaces the file.  A valid image is sufficient for restart
even when the referenced capture shard has subsequently been retained away.
