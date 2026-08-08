#pragma once

// gguf-m2 artifact layout gate.
//
// conversion/gguf_m2_repack.py writes three layout keys into every *.m2.gguf:
//
//   m2.layout.version     u32     the layout generation
//   m2.layout.description string  every layout constant the artifact was
//                                 packed against, in one canonical string
//   m2.layout.hash        string  sha256(m2.layout.description)
//
// The loader used to validate only the version. That made the version the sole
// barrier between an artifact and a build that decodes it differently, and a
// version is a number a human has to remember to bump: an incompatible layout
// change that kept version 1 would have been accepted and would have silently
// misread packed code bytes as plausible weights. That is exactly the failure
// mode this format exists to make impossible.
//
// So the *description* — the hash's preimage, which is built here out of the
// same constants the kernels decode with — is what is actually enforced. It
// carries no discipline requirement: change QK_E4M3_M2, the scale bias, the
// NF8 group size or the alignment and the string this build expects changes
// with it, so every previously converted artifact stops loading, loudly,
// naming both strings. The hash is validated too, against the paired literal
// below, so an artifact whose two keys disagree is refused as well.
//
// Regenerating the hashes after a deliberate layout change (and bump VERSION,
// and the converter's M2_LAYOUT_VERSION, in the same change):
//
//   ./build/bin/test-m2-layout-gate --print
//
// prints the description and the sha256 this build now needs; tests/test-m2-
// layout-gate.cpp pins both, so the change cannot land without updating them.
//
// See notes/2026-08-07-gguf-m2-artifact-format-design.md.

#include <cstdint>
#include <string>

namespace llama_m2_layout {

// must equal M2_LAYOUT_VERSION in conversion/gguf_m2_repack.py
constexpr uint32_t VERSION = 1;

// dense plane, E4M3_M2 — must match block_e4m3_m2 in ggml/src/ggml-common.h
constexpr int E4M3_QK   = 1024;                // QK_E4M3_M2
constexpr int E4M3_GSZ  = 128;                 // elements per E8M0 scale
constexpr int E4M3_NSC  = E4M3_QK/E4M3_GSZ;    // scale bytes per block
constexpr int E4M3_NPAD = 8;                   // zero pad bytes per block
constexpr int E4M3_TILE = 128;                 // converter source-recovery tile

// dense plane, NF8_M2 — must match block_nf8_m2 in ggml/src/ggml-common.h
constexpr int NF8_QK   = 1024;                 // QK_NF8_M2
constexpr int NF8_GSZ  = 32;                   // NF8_M2_GSZ
constexpr int NF8_BIAS = 63;                   // NF8_M2_EXP_BIAS

// expert plane, MXFP4_M2 — must match block_mxfp4_m2 in ggml/src/ggml-common.h
constexpr int MXFP4_QK         = 2048;                    // QK_MXFP4_M2
constexpr int MXFP4_SUB        = 32;                      // source MXFP4 sub-block
constexpr int MXFP4_NSB        = MXFP4_QK/MXFP4_SUB;      // sub-blocks per block
constexpr int MXFP4_SCALE_BIAS = 116;                     // MXFP4_M2_SCALE_BIAS

// tensor data alignment the converter applies
constexpr int ALIGN = 16384;

// the dense encoding an artifact was converted with. The converter records one
// description per run, so an artifact declares exactly one of these — including
// an experts-only artifact (--keep-dense-plane), which still names the dense
// encoding its run was configured for.
enum dense_plane {
    DENSE_E4M3 = 0,
    DENSE_NF8  = 1,
    DENSE_COUNT,
};

inline std::string description(dense_plane dense) {
    const std::string v = std::to_string(VERSION);

    const std::string d = dense == DENSE_NF8
        ? "nf8_m2:v"  + v + ":block" + std::to_string(NF8_QK)
              + ":codes" + std::to_string(NF8_QK) + "+sb" + std::to_string(NF8_QK/NF8_GSZ)
              + ":base=per-" + std::to_string(NF8_GSZ) + "-bias" + std::to_string(NF8_BIAS)
              + ":escape=e4m3"
        : "e4m3_m2:v" + v + ":block" + std::to_string(E4M3_QK)
              + ":codes" + std::to_string(E4M3_QK) + "+sc" + std::to_string(E4M3_NSC)
              + "+pad" + std::to_string(E4M3_NPAD)
              + ":scale=e8m0-per-" + std::to_string(E4M3_GSZ)
              + ":tile" + std::to_string(E4M3_TILE) + "x" + std::to_string(E4M3_TILE);

    return d + ";mxfp4_m2:v" + v + ":block" + std::to_string(MXFP4_QK)
             + ":codes" + std::to_string(MXFP4_NSB) + "x16+scnib" + std::to_string(MXFP4_NSB)
             + ":scale=e8m0-nibble-bias" + std::to_string(MXFP4_SCALE_BIAS)
             + ";align" + std::to_string(ALIGN);
}

// sha256(description(dense)). Regenerate with `test-m2-layout-gate --print`
// after any change above; a stale value here fails closed, refusing artifacts
// rather than accepting them.
inline const char * hash(dense_plane dense) {
    return dense == DENSE_NF8
        ? "e2445da406f6a1478d5a41193841a3b08c402d4c27f829f68bd57c59c89eaeb9"
        : "9c3da0a661f99161c6296131bbbdefff543fe0af0dde31553b4c850f38a86728";
}

// Validate an artifact's declared layout keys against this build.
//
// Returns an empty string when the artifact may be decoded, otherwise the
// reason it may not, ready to be put in the exception the loader throws. Never
// returns "accept anyway": there is no fallback path, because reinterpreting
// packed code bytes under the wrong layout produces plausible garbage rather
// than a visible failure.
//
// `description` and `hash` are the artifact's declared values; either may be
// null, meaning the key was absent.
inline std::string check(uint32_t version, const char * desc, const char * hash_hex) {
    if (version != VERSION) {
        return "m2.layout.version = " + std::to_string(version) + " but this build implements version "
             + std::to_string(VERSION) + " exactly — rebuild or reconvert";
    }

    if (desc == nullptr) {
        return "the required 'm2.layout.description' metadata key is missing — refusing to interpret "
               "fork-owned tensor encodings from an artifact that does not state its layout";
    }
    if (hash_hex == nullptr) {
        return "the required 'm2.layout.hash' metadata key is missing — refusing to interpret "
               "fork-owned tensor encodings from an artifact that does not state its layout";
    }

    for (int i = 0; i < DENSE_COUNT; ++i) {
        const dense_plane d = (dense_plane) i;

        if (description(d) != desc) {
            continue;
        }
        if (hash(d) != std::string(hash_hex)) {
            return std::string("m2.layout.hash does not match m2.layout.description.\n"
                   "  description: ") + desc + "\n"
                   "  declared hash: " + hash_hex + "\n"
                   "  sha256 of that description: " + hash(d) + "\n"
                   "the artifact's own layout keys disagree — reconvert it";
        }
        return "";
    }

    std::string err = std::string("m2.layout.description does not describe a layout this build decodes.\n"
              "  artifact: ") + desc + "\n";
    for (int i = 0; i < DENSE_COUNT; ++i) {
        err += "  this build: " + description((dense_plane) i) + "\n";
    }
    err += "every constant in that string is one the kernels decode with, so a difference means the packed "
           "bytes would be read as something they are not — reconvert the artifact with this build's "
           "conversion/gguf_m2_repack.py";

    return err;
}

} // namespace llama_m2_layout
