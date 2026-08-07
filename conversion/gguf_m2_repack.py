#!/usr/bin/env python3
"""gguf-m2 repack: convert the served DSV4-Flash GGUF into the fork-owned
gguf-m2 v1 artifact.

What it does (see notes/2026-08-07-gguf-m2-artifact-format-design.md):

- E4M3-encodes exactly the dense-plane tensors that the recoverability
  verifier proved bit-exact: the served BF16 values are e4m3 * 2^k with one
  E8M0 scale per 128x128 tile of the (out, in) matrix, so the conversion is
  a *recovery* of the checkpoint's own FP8 encoding, not a quantization.
  Every converted tensor is re-decoded and compared bit-exact against the
  source BF16 before being written; any mismatch aborts the run.
- Passes every other tensor through byte-identical (the MXFP4 expert plane
  goes through `encode_expert_plane`, a pluggable stub for the coming
  MXFP4_M2 workstream).
- Writes the required `m2.layout.version = 1` (u32) gate metadata plus a
  layout hash, and 16 KiB-aligns every tensor's data (Metal sparse-pool /
  MTLIO page size).

The output is a single file (input split.* metadata is dropped) and must be
named *.m2.gguf.

Usage:
    python3 conversion/gguf_m2_repack.py SHARD1 [SHARD2 ...] -o OUT.m2.gguf

The full-model conversion is expected to run on the M2 in a coordinated
window (I/O-heavy; do not co-run with latency-sensitive benchmarks).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np

# import from the repo checkout, not an installed gguf
sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType  # noqa: E402

logger = logging.getLogger("gguf-m2-repack")

# ---------------------------------------------------------------------------
# format constants (must match block_e4m3_m2 in ggml/src/ggml-common.h)

M2_LAYOUT_VERSION = 1
QK = 1024            # elements per block
GSZ = 128            # elements per E8M0 scale (and source tile edge)
NSC = QK // GSZ      # scale bytes per block (8)
NPAD = 8             # zero pad bytes per block
BLOCK_BYTES = QK + NSC + NPAD  # 1040
TILE = 128           # source recovery tile is TILE x TILE (out, in)

M2_LAYOUT_DESC = (
    f"e4m3_m2:v{M2_LAYOUT_VERSION}:block{QK}:codes{QK}+sc{NSC}+pad{NPAD}"
    f":scale=e8m0-per-{GSZ}:tile{TILE}x{TILE}:align16384"
)

# The exact pass list proved bit-exactly recoverable by the verifier
# (/tmp/e4m3-verify.log on the m2, 2026-08-07: 365 tensors, 10.91 GiB, zero
# failing tiles). A pass-list tensor that fails recovery is a fatal error —
# it means the served artifact changed and the design must be revisited.
E4M3_PASS_SUFFIXES = (
    ".attn_output_a.weight",
    ".attn_output_b.weight",
    ".attn_q_a.weight",
    ".attn_q_b.weight",       # also matches blk.N.indexer.attn_q_b.weight
    ".attn_kv.weight",
    ".ffn_gate_shexp.weight",
    ".ffn_up_shexp.weight",
    ".ffn_down_shexp.weight",
)


def is_e4m3_target(name: str) -> bool:
    return name.startswith("blk.") and name.endswith(E4M3_PASS_SUFFIXES)


# ---------------------------------------------------------------------------
# E4M3 tables

def _e4m3_pos_values() -> np.ndarray:
    """The 127 finite non-negative OCP E4M3(fn) values + [127]=inf sentinel,
    indexed by code (codes 0..126 are monotonically increasing in value;
    code 127 is NaN and is never produced)."""
    vals = np.empty(128, dtype=np.float64)
    for c in range(127):
        e = c >> 3
        m = c & 0x7
        vals[c] = m * 2.0**-9 if e == 0 else (1.0 + m/8.0) * 2.0**(e - 7)
    vals[127] = np.inf  # NaN slot; sentinel keeps searchsorted monotonic
    return vals


E4M3_POS = _e4m3_pos_values()
E4M3_MAX = 448.0


def _e4m3_decode_lut_f32() -> np.ndarray:
    lut = np.empty(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        c = b & 0x7F
        v = np.inf if c == 127 else E4M3_POS[c]
        lut[b] = np.float32(s * v)
    # keep -0.0 for code 0x80 (np keeps the sign through the multiply)
    lut[0x80] = np.float32(-0.0)
    return lut


E4M3_LUT_F32 = _e4m3_decode_lut_f32()


# ---------------------------------------------------------------------------
# encoder

class RecoveryError(RuntimeError):
    pass


def encode_e4m3_m2(name: str, u16: np.ndarray) -> tuple[np.ndarray, dict]:
    """Encode a BF16 tensor (u16: uint16 array of shape (ne1, ne0), bf16 bit
    patterns, row-major with ne0 contiguous) to E4M3_M2 block bytes.

    Recovers the per-128x128-tile E8M0 scales exactly (set-membership over the
    e4m3 value set; small k search upward from the minimal admissible k) and
    self-checks the result bit-exactly against the source before returning.

    Returns (bytes of shape (ne1, ne0//QK * BLOCK_BYTES), stats).
    """
    ne1, ne0 = u16.shape
    if ne0 % QK != 0:
        raise RecoveryError(f"{name}: ne0={ne0} is not a multiple of {QK}")
    if ne1 % TILE != 0:
        raise RecoveryError(f"{name}: ne1={ne1} is not a multiple of {TILE} (tile grid)")

    t0 = ne0 // TILE  # tiles along the input dim

    out = np.empty((ne1, (ne0 // QK) * BLOCK_BYTES), dtype=np.uint8)

    k_min_seen = 999
    k_max_seen = -999

    # process in bands of TILE rows (one tile-row per iteration): bounded
    # memory, exact tile alignment
    for r0 in range(0, ne1, TILE):
        band_u16 = u16[r0:r0 + TILE]
        f32 = (band_u16.astype(np.uint32) << 16).view(np.float32)
        a = f32.astype(np.float64)                       # exact
        sabs = np.abs(a).reshape(TILE, t0, TILE)         # (row, tile, col-in-tile)

        # minimal admissible k per tile: smallest k with amax <= 448 * 2^k
        amax = sabs.max(axis=(0, 2))                     # (t0,)
        m, e = np.frexp(amax)                            # amax = m * 2^e, m in [0.5, 1)
        k = np.where(m <= 0.875, e - 9, e - 8).astype(np.int64)
        k[amax == 0.0] = 0

        # membership test with upward k search (the served artifact needs at
        # most a few steps; the verifier used the same window)
        codes = None
        for _attempt in range(4):
            scaled = np.ldexp(sabs, -k[np.newaxis, :, np.newaxis])  # exact
            idx = np.searchsorted(E4M3_POS, scaled.reshape(-1))
            idx = np.minimum(idx, 126)
            ok_elem = E4M3_POS[idx] == scaled.reshape(-1)
            ok_tile = ok_elem.reshape(TILE, t0, TILE).all(axis=(0, 2))
            if ok_tile.all():
                codes = idx.astype(np.uint8).reshape(TILE, t0, TILE)
                break
            k[~ok_tile] += 1
        if codes is None:
            bad = int((~ok_tile).sum())
            raise RecoveryError(
                f"{name}: {bad}/{t0} tiles in rows [{r0},{r0+TILE}) are NOT e4m3*2^k recoverable "
                f"— the served artifact does not match the verifier's pass list; refusing to continue")

        if k.min() < k_min_seen:
            k_min_seen = int(k.min())
        if k.max() > k_max_seen:
            k_max_seen = int(k.max())

        sc = (k + 127).astype(np.int64)
        if sc.min() < 1 or sc.max() > 254:
            raise RecoveryError(f"{name}: E8M0 scale byte out of [1,254]: {sc.min()}..{sc.max()}")
        sc_u8 = sc.astype(np.uint8)

        sign = (band_u16 >> 15).astype(np.uint8) << 7    # bf16 sign bit -> e4m3 sign bit
        code_bytes = codes.reshape(TILE, ne0) | sign     # (TILE, ne0)

        # self-check: decode -> f32 -> must exactly equal the source bf16
        dec = E4M3_LUT_F32[code_bytes].reshape(TILE, t0, TILE) * \
            np.ldexp(np.float32(1.0), sc.astype(np.int32) - 127)[np.newaxis, :, np.newaxis].astype(np.float32)
        dec_u32 = dec.reshape(TILE, ne0).view(np.uint32)
        if (dec_u32 & 0xFFFF).any() or not np.array_equal((dec_u32 >> 16).astype(np.uint16), band_u16):
            n_bad = int(np.count_nonzero((dec_u32 >> 16).astype(np.uint16) != band_u16) +
                        np.count_nonzero(dec_u32 & 0xFFFF))
            raise RecoveryError(
                f"{name}: SELF-CHECK FAILED in rows [{r0},{r0+TILE}): {n_bad} decoded values "
                f"are not bit-identical to the source BF16 — aborting before writing")

        # assemble the block layout: per row, per 1024-block:
        # [1024 codes][8 scale bytes][8 zero pad]
        nb = ne0 // QK
        band_out = out[r0:r0 + TILE].reshape(TILE, nb, BLOCK_BYTES)
        band_out[:, :, :QK] = code_bytes.reshape(TILE, nb, QK)
        # scale bytes: tile j covers columns [j*128, (j+1)*128); block b holds
        # groups b*8..b*8+7 == tiles b*8..b*8+7; identical for every row of the band
        band_out[:, :, QK:QK + NSC] = np.broadcast_to(sc_u8.reshape(nb, NSC), (TILE, nb, NSC))
        band_out[:, :, QK + NSC:] = 0

    stats = {"k_min": k_min_seen, "k_max": k_max_seen}
    return out.reshape(ne1, -1), stats


# ---------------------------------------------------------------------------
# expert-plane hook (MXFP4_M2 comes from a separate workstream)

def encode_expert_plane(tensor) -> tuple[GGMLQuantizationType, np.ndarray] | None:
    """Pluggable encoder for the routed-expert plane (GGML_TYPE_MXFP4 input).

    Return None to pass the tensor through byte-identical, or
    (raw_dtype, byte_data) to repack it. gguf-m2 v1 passes the expert plane
    through unchanged; the MXFP4_M2 split-plane encoding (16-B-aligned code
    plane + 4-bit absolute E8M0 scales) plugs in here when its workstream
    lands, gated by its own layout version bump.
    """
    del tensor
    return None


# ---------------------------------------------------------------------------
# driver

SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="input GGUF shard(s), in order")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output artifact (*.m2.gguf)")
    parser.add_argument("--dry-run", action="store_true", help="plan + verify recoverability of the pass list, write nothing")
    parser.add_argument("--max-e4m3", type=int, default=0, help="convert at most N tensors (testing), 0 = all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not str(args.output).endswith(".m2.gguf"):
        parser.error("output must be named *.m2.gguf (the artifact is not a stock GGUF)")

    t_start = time.time()

    readers = [GGUFReader(p) for p in args.inputs]

    arch_field = readers[0].fields.get("general.architecture")
    arch = arch_field.contents() if arch_field is not None else "unknown"

    writer = GGUFWriter(args.output if not args.dry_run else Path("/dev/null"), arch, use_temp_file=False)
    writer.add_custom_alignment(16384)  # Metal sparse-pool / MTLIO page size

    # copy metadata from the first shard; drop split bookkeeping (single file)
    for field in readers[0].fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        if field.name in SPLIT_KEYS or field.name == "general.alignment":
            continue
        if field.name.startswith("m2."):
            raise RuntimeError(f"input already carries fork metadata ({field.name}) — refusing to re-convert")
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    writer.add_uint32("m2.layout.version", M2_LAYOUT_VERSION)
    writer.add_string("m2.layout.hash", hashlib.sha256(M2_LAYOUT_DESC.encode()).hexdigest())
    writer.add_string("m2.layout.description", M2_LAYOUT_DESC)

    # plan
    all_tensors = [t for r in readers for t in r.tensors]
    n_e4m3 = 0
    plan: list[tuple] = []  # (tensor, mode) with mode in {"e4m3", "expert", "copy"}
    for t in all_tensors:
        if is_e4m3_target(t.name):
            if t.tensor_type != GGMLQuantizationType.BF16 or len(t.shape) != 2:
                raise RuntimeError(f"pass-list tensor {t.name} is {t.tensor_type.name}/{len(t.shape)}D, expected BF16/2D")
            if args.max_e4m3 and n_e4m3 >= args.max_e4m3:
                plan.append((t, "copy"))
                continue
            n_e4m3 += 1
            plan.append((t, "e4m3"))
        elif t.tensor_type == GGMLQuantizationType.MXFP4:
            plan.append((t, "expert"))
        else:
            plan.append((t, "copy"))

    # tensor infos (must be complete before the header is written)
    for t, mode in plan:
        if mode == "e4m3":
            ne0, ne1 = int(t.shape[0]), int(t.shape[1])
            nbytes = ne1 * (ne0 // QK) * BLOCK_BYTES
            writer.add_tensor_info(t.name, (ne1, ne0 // QK * BLOCK_BYTES), np.uint8, nbytes,
                                   raw_dtype=GGMLQuantizationType.E4M3_M2)
        elif mode == "expert":
            enc = encode_expert_plane(t)
            if enc is not None:
                raise NotImplementedError("expert-plane repack is a stub in gguf-m2 v1")
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    logger.info(f"plan: {len(plan)} tensors, {n_e4m3} -> E4M3_M2, "
                f"{sum(1 for _, m in plan if m == 'expert')} expert passthrough, "
                f"{sum(1 for _, m in plan if m == 'copy')} copied")

    if args.dry_run:
        n_bytes_saved = 0
        for t, mode in plan:
            if mode != "e4m3":
                continue
            ne0, ne1 = int(t.shape[0]), int(t.shape[1])
            u16 = t.data.view(np.uint16).reshape(ne1, ne0)
            tt = time.time()
            _, stats = encode_e4m3_m2(t.name, u16)
            n_bytes_saved += t.n_bytes - ne1 * (ne0 // QK) * BLOCK_BYTES
            logger.info(f"[dry] {t.name}: RECOVERED bit-exact, k in [{stats['k_min']},{stats['k_max']}], "
                        f"{time.time()-tt:.1f}s")
        logger.info(f"[dry] all pass-list tensors recovered; would save {n_bytes_saved/2**30:.3f} GiB")
        return 0

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    n_done = 0
    bytes_in = 0
    bytes_out = 0
    for t, mode in plan:
        if mode == "e4m3":
            ne0, ne1 = int(t.shape[0]), int(t.shape[1])
            u16 = t.data.view(np.uint16).reshape(ne1, ne0)
            tt = time.time()
            enc, stats = encode_e4m3_m2(t.name, u16)  # self-checks internally, raises on any mismatch
            writer.write_tensor_data(enc)
            logger.info(f"[{n_done+1}/{len(plan)}] E4M3_M2 {t.name} ({ne0}x{ne1}) "
                        f"k in [{stats['k_min']},{stats['k_max']}] {time.time()-tt:.1f}s")
            bytes_out += enc.nbytes
        else:
            writer.write_tensor_data(t.data, tensor_endianess=readers[0].endianess)
            bytes_out += t.n_bytes
        bytes_in += t.n_bytes
        n_done += 1

    writer.close()

    logger.info(f"done in {time.time()-t_start:.0f}s: {bytes_in/2**30:.3f} GiB in -> "
                f"{bytes_out/2**30:.3f} GiB out (tensor data, before alignment padding), "
                f"saved {(bytes_in-bytes_out)/2**30:.3f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
