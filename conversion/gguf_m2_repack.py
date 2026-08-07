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
- Repacks the routed-expert MXFP4 plane to GGML_TYPE_MXFP4_M2: split
  code/scale planes (aligned 16-byte code groups) with 4-bit absolute
  biased E8M0 scale nibbles. This is a byte shuffle — identical codes,
  identical scale values — and every converted slab is reconstructed and
  compared byte-identical against the source before writing. If any scale
  byte falls outside the nibble window the conversion aborts (the window is
  a layout constant validated against the measured census; never clamp).
- Passes every other tensor through byte-identical.
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

# expert plane (must match block_mxfp4_m2 in ggml/src/ggml-common.h)
MXFP4_QK = 32                 # source MXFP4 sub-block elements
MXFP4_BLOCK_BYTES = 17        # source: 1 E8M0 byte + 16 nibble-packed code bytes
M2X_QK = 2048                 # elements per MXFP4_M2 block
M2X_SB = M2X_QK // MXFP4_QK   # 64 sub-blocks per block
M2X_BLOCK_BYTES = M2X_SB*16 + M2X_SB//2  # 1024 code bytes + 32 scale-nibble bytes = 1056
M2X_SCALE_BIAS = 116          # E8M0 byte = bias + nibble; census range [118,126] on both planes

M2_LAYOUT_DESC = (
    f"e4m3_m2:v{M2_LAYOUT_VERSION}:block{QK}:codes{QK}+sc{NSC}+pad{NPAD}"
    f":scale=e8m0-per-{GSZ}:tile{TILE}x{TILE};"
    f"mxfp4_m2:v{M2_LAYOUT_VERSION}:block{M2X_QK}:codes{M2X_SB}x16+scnib{M2X_SB}"
    f":scale=e8m0-nibble-bias{M2X_SCALE_BIAS};align16384"
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
# expert-plane encoder (GGML_TYPE_MXFP4 -> GGML_TYPE_MXFP4_M2)

def encode_mxfp4_m2_slab(name: str, raw: np.ndarray) -> np.ndarray:
    """Repack a slab of MXFP4 rows to MXFP4_M2 block bytes.

    raw: uint8 (nrows, ne0//32*17) — interleaved [E8M0][16 code bytes] blocks.
    Splits planes, packs the scales to biased nibbles, and self-checks by
    reconstructing the original 17-byte stream byte-identically.
    Raises RecoveryError if any scale byte falls outside the nibble window
    [M2X_SCALE_BIAS, M2X_SCALE_BIAS+15] — that means the artifact does not
    match the measured census and the layout must be revisited, never clamped.
    """
    nrows, row_bytes = raw.shape
    if row_bytes % MXFP4_BLOCK_BYTES != 0:
        raise RecoveryError(f"{name}: row byte count {row_bytes} not a multiple of 17")
    nb17 = row_bytes // MXFP4_BLOCK_BYTES
    if nb17 % M2X_SB != 0:
        raise RecoveryError(f"{name}: {nb17} MXFP4 blocks/row not a multiple of {M2X_SB} (ne0 % {M2X_QK} != 0)")
    nbm2 = nb17 // M2X_SB

    blocks = raw.reshape(nrows, nb17, MXFP4_BLOCK_BYTES)
    e  = blocks[:, :, 0]
    qs = blocks[:, :, 1:]

    e_min = int(e.min())
    e_max = int(e.max())
    if e_min < M2X_SCALE_BIAS or e_max > M2X_SCALE_BIAS + 15:
        raise RecoveryError(
            f"{name}: E8M0 scale bytes span [{e_min}, {e_max}], outside the 4-bit window "
            f"[{M2X_SCALE_BIAS}, {M2X_SCALE_BIAS+15}] — the artifact does not match the measured "
            f"census; refusing to convert (bump the layout, do not clamp)")

    nib = (e - M2X_SCALE_BIAS).astype(np.uint8)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).reshape(nrows, nbm2, M2X_SB//2)

    out = np.empty((nrows, nbm2, M2X_BLOCK_BYTES), dtype=np.uint8)
    out[:, :, :M2X_SB*16] = qs.reshape(nrows, nbm2, M2X_SB*16)
    out[:, :, M2X_SB*16:] = packed

    # self-check: reconstruct the source 17-byte stream from the encoded bytes
    chk = np.empty_like(blocks)
    enc_sc = out[:, :, M2X_SB*16:].reshape(nrows, nb17//2)
    chk[:, 0::2, 0] = M2X_SCALE_BIAS + (enc_sc & 0x0F)
    chk[:, 1::2, 0] = M2X_SCALE_BIAS + (enc_sc >> 4)
    chk[:, :, 1:] = out[:, :, :M2X_SB*16].reshape(nrows, nb17, 16)
    if not np.array_equal(chk, blocks):
        n_bad = int(np.count_nonzero(chk != blocks))
        raise RecoveryError(f"{name}: SELF-CHECK FAILED: {n_bad} reconstructed bytes differ from the source MXFP4")

    return out.reshape(nrows, nbm2*M2X_BLOCK_BYTES)


def encode_expert_plane(tensor, stats: dict):
    """Encoder for the routed-expert plane (GGML_TYPE_MXFP4 input, 3D).

    Yields one encoded uint8 array per expert slice (bounded memory; the
    caller streams them into the output file). Scale-window scan and the
    byte-identical self-check happen per slice inside encode_mxfp4_m2_slab.
    """
    data = tensor.data  # (ne2, ne1, ne0//32*17) uint8, mmap-backed
    if data.ndim != 3:
        raise RecoveryError(f"{tensor.name}: expected a 3D expert tensor, got {data.ndim}D")
    for e2 in range(data.shape[0]):
        enc = encode_mxfp4_m2_slab(f"{tensor.name}[expert {e2}]", np.ascontiguousarray(data[e2]))
        stats["experts"] = stats.get("experts", 0) + 1
        yield enc


def expert_plane_m2_nbytes(tensor) -> int:
    ne0 = int(tensor.shape[0])
    return (ne0 // M2X_QK) * M2X_BLOCK_BYTES * int(tensor.shape[1]) * int(tensor.shape[2])


def write_tensor_data_streamed(writer: GGUFWriter, chunks, total_nbytes: int) -> None:
    """Streamed twin of GGUFWriter.write_tensor_data: pops the next tensor
    info and writes its payload from an iterator of byte chunks, with the
    same alignment padding semantics."""
    from gguf.gguf_writer import WriterState  # local import: private-ish API

    if writer.state is not WriterState.TI_DATA and writer.state is not WriterState.WEIGHTS:
        raise ValueError(f"expected TI_DATA/WEIGHTS writer state, got {writer.state}")

    file_id = next(i for i, tensors in enumerate(writer.tensors) if len(tensors) > 0)
    fout = writer.fout[file_id]

    first_tensor_name = next(iter(writer.tensors[file_id]))
    ti = writer.tensors[file_id].pop(first_tensor_name)
    if ti.nbytes != total_nbytes:
        raise ValueError(f"{first_tensor_name}: tensor info nbytes {ti.nbytes} != streamed nbytes {total_nbytes}")

    writer.write_padding(fout, fout.tell())
    written = 0
    for chunk in chunks:
        b = chunk.tobytes() if hasattr(chunk, "tobytes") else bytes(chunk)
        fout.write(b)
        written += len(b)
    if written != total_nbytes:
        raise ValueError(f"{first_tensor_name}: wrote {written} bytes, expected {total_nbytes}")
    writer.write_padding(fout, written)

    writer.state = WriterState.WEIGHTS


# ---------------------------------------------------------------------------
# driver

SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="input GGUF shard(s), in order")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output artifact (*.m2.gguf)")
    parser.add_argument("--dry-run", action="store_true", help="plan + verify recoverability of the pass list, write nothing")
    parser.add_argument("--max-e4m3", type=int, default=0, help="convert at most N dense tensors (testing), 0 = all")
    parser.add_argument("--max-expert", type=int, default=0, help="convert at most N expert tensors (testing), 0 = all")
    parser.add_argument("--keep-expert-plane", action="store_true",
                        help="pass MXFP4 expert tensors through unchanged instead of repacking to MXFP4_M2")
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
    n_expert = 0
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
            if args.keep_expert_plane or (args.max_expert and n_expert >= args.max_expert):
                plan.append((t, "copy"))
                continue
            if len(t.shape) != 3 or int(t.shape[0]) % M2X_QK != 0:
                # this converter targets the DSV4 expert geometry; anything else
                # is unexpected — refuse rather than silently pass through
                raise RuntimeError(
                    f"MXFP4 tensor {t.name} shape {tuple(int(v) for v in t.shape)} does not fit the "
                    f"MXFP4_M2 block (ne0 % {M2X_QK} != 0 or not 3D); use --keep-expert-plane to bypass")
            n_expert += 1
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
            ne0, ne1, ne2 = (int(v) for v in t.shape[:3])
            nbytes = expert_plane_m2_nbytes(t)
            writer.add_tensor_info(t.name, (ne2, ne1, ne0 // M2X_QK * M2X_BLOCK_BYTES), np.uint8, nbytes,
                                   raw_dtype=GGMLQuantizationType.MXFP4_M2)
            # per-tensor record of the bias the scale nibbles were packed
            # against; the loader refuses the artifact if this differs from
            # the MXFP4_M2_SCALE_BIAS constant its kernels decode with
            writer.add_uint32(f"m2.mxfp4_m2.scale_bias.{t.name}", M2X_SCALE_BIAS)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    logger.info(f"plan: {len(plan)} tensors, {n_e4m3} -> E4M3_M2, "
                f"{n_expert} -> MXFP4_M2, "
                f"{sum(1 for _, m in plan if m == 'copy')} copied")

    if args.dry_run:
        n_bytes_saved = 0
        for t, mode in plan:
            if mode == "e4m3":
                ne0, ne1 = int(t.shape[0]), int(t.shape[1])
                u16 = t.data.view(np.uint16).reshape(ne1, ne0)
                tt = time.time()
                _, stats = encode_e4m3_m2(t.name, u16)
                n_bytes_saved += t.n_bytes - ne1 * (ne0 // QK) * BLOCK_BYTES
                logger.info(f"[dry] {t.name}: RECOVERED bit-exact, k in [{stats['k_min']},{stats['k_max']}], "
                            f"{time.time()-tt:.1f}s")
            elif mode == "expert":
                tt = time.time()
                st: dict = {}
                for _ in encode_expert_plane(t, st):
                    pass
                n_bytes_saved += t.n_bytes - expert_plane_m2_nbytes(t)
                logger.info(f"[dry] {t.name}: repacked + self-checked byte-identical "
                            f"({st.get('experts', 0)} experts, {time.time()-tt:.1f}s)")
        logger.info(f"[dry] all convertible tensors verified; would save {n_bytes_saved/2**30:.3f} GiB")
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
        elif mode == "expert":
            tt = time.time()
            st: dict = {}
            nbytes = expert_plane_m2_nbytes(t)
            # streamed per expert slice; scan + self-check happen inside
            write_tensor_data_streamed(writer, encode_expert_plane(t, st), nbytes)
            logger.info(f"[{n_done+1}/{len(plan)}] MXFP4_M2 {t.name} "
                        f"({int(t.shape[0])}x{int(t.shape[1])}x{int(t.shape[2])}) "
                        f"self-checked byte-identical, {time.time()-tt:.1f}s")
            bytes_out += nbytes
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
