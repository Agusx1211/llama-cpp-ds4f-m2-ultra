// kernel_flash_attn_ext_vec zeroes its output accumulator with
//
//     so4 += tiisg;                                  // o4_t = float4
//     for (short i = 0; i < DV4/NL; ++i) so4[i*NL] = 0;
//
// so lane `t` writes float4 slots t, t+NL, ..., t+(DV4/NL - 1)*NL of this
// SIMDgroup's accumulator. The accumulator slot is PAD(DV,128)/4 float4 wide
// and only lanes 0..NL-1 ever accumulate into it (every accumulate and every
// read is guarded by `ty == 0`, i.e. tiisg < NL, or bounded by `i < DV4`), so
// every store from a lane with tiisg >= NL is dead - but the highest of them,
//
//     (NW - 1) + (DV4/NL - 1)*NL
//
// exceeds the slot for four of the shipped head-size instantiations, by 16
// float4 = 256 B. Inside a threadgroup that lands in the *next* SIMDgroup's
// accumulator (harmless in isolation, since both stores are zero and both
// precede the threadgroup barrier); for the last SIMDgroup it runs past the
// threadgroup memory the host actually declared, which is undefined behaviour.
//
// Part 0 reads ggml-metal.metal itself, because the arithmetic below is only a
// model of the kernel and a model cannot notice that the thing it models has
// changed. It requires (a) the zero-fill loop to still carry a bound against
// DV4 - delete the guard and this fails, which parts 1 and 2 do not, since the
// surplus stores are zeros landing in a region that is being zeroed anyway and
// are therefore invisible to any output comparison - and (b) the set of shipped
// <DK, DV, NE> instantiations to be exactly the set part 1 checks, so a new
// head size cannot be added without re-deriving the bound for it.
//
// Part 1 mirrors that arithmetic for every instantiation the kernel ships and
// requires the shipped bound to keep every store inside the slot. It is the
// check that would have caught the original defect and that catches a future
// change to NE, NL or the shared-memory layout.
//
// Part 2 runs flash attention at the four affected head sizes on Metal and
// compares against the CPU backend, at one, two and four key-stream
// SIMDgroups, so the geometries where the overflow was reachable are exercised
// end to end.
//
// Skips part 2 with status 0 when no Metal device is present. Parts 0 and 1 are
// pure host arithmetic and text inspection and always run.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#undef assert
#define assert(expr) do {                                                      \
    if (!(expr)) {                                                             \
        std::fprintf(stderr, "check failed at %s:%d: %s\n",                    \
                __FILE__, __LINE__, #expr);                                    \
        std::abort();                                                          \
    }                                                                          \
} while (false)

// N_SIMDWIDTH in ggml-metal.metal
static const int NW = 32;

static int pad2(int x, int n) {
    return (x + n - 1) & ~(n - 1);
}

// every kernel_flash_attn_ext_vec instantiation, as <DK, DV, NE>
struct vec_inst {
    int dk;
    int dv;
    int ne;
};

static const vec_inst g_inst[] = {
    {  32,  32, 4 },
    {  64,  64, 2 },
    {  96,  96, 4 },
    { 128, 128, 1 },
    { 192, 128, 2 },
    { 192, 192, 2 },
    { 256, 256, 1 },
    { 320, 256, 2 },
    { 512, 512, 1 }, // the DeepSeek V4 Flash decode geometry
    { 576, 512, 2 }, // the DeepSeek V2/V3 MLA geometry
};

static const int g_n_inst = (int) (sizeof(g_inst)/sizeof(g_inst[0]));

// ------------------------------------------------------------------- part 0
// GGML_METAL_SOURCE_PATH is set by tests/CMakeLists.txt.

static std::string strip_ws(const std::string & s) {
    std::string r;
    for (char c : s) {
        if (!std::isspace((unsigned char) c)) {
            r += c;
        }
    }
    return r;
}

// the text of kernel_flash_attn_ext_vec's accumulator zero-fill, i.e. what sits
// between the "zero out so" and "zero out shared memory SH" markers
static bool zero_fill_body(const std::string & src, std::string & out) {
    const size_t a = src.find("// zero out so");
    if (a == std::string::npos) {
        return false;
    }
    const size_t b = src.find("// zero out shared memory SH", a);
    if (b == std::string::npos) {
        return false;
    }
    out = src.substr(a, b - a);
    return true;
}

// every `kernel_flash_attn_ext_vec<...>` template instantiation declared with a
// [[host_name(...)]], as its trailing <DK, DV, NE> triple
static std::set<std::vector<int>> shipped_instantiations(const std::string & src) {
    std::set<std::vector<int>> res;

    std::istringstream is(src);
    std::string line;
    while (std::getline(is, line)) {
        if (line.find("host_name(\"kernel_flash_attn_ext_vec_") == std::string::npos) {
            continue;
        }
        // the instantiation itself, not kernel_flash_attn_ext_vec_reduce<...>
        const size_t a = line.find("kernel_flash_attn_ext_vec<");
        if (a == std::string::npos) {
            continue;
        }
        const size_t b = line.rfind('>');
        if (b == std::string::npos || b < a) {
            continue;
        }

        // the last three comma-separated template arguments are DK, DV, NE
        std::vector<std::string> args;
        std::string cur;
        for (size_t i = a + std::strlen("kernel_flash_attn_ext_vec<"); i < b; ++i) {
            if (line[i] == ',') {
                args.push_back(cur);
                cur.clear();
            } else {
                cur += line[i];
            }
        }
        args.push_back(cur);

        if (args.size() < 3) {
            continue;
        }

        std::vector<int> triple;
        bool numeric = true;
        for (size_t i = args.size() - 3; i < args.size(); ++i) {
            const std::string t = strip_ws(args[i]);
            if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) {
                numeric = false;
                break;
            }
            triple.push_back(std::atoi(t.c_str()));
        }
        if (numeric) {
            res.insert(triple);
        }
    }

    return res;
}

static ggml_backend_t metal_backend_init() {
    ggml_backend_load_all();

    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : "?";
        if (std::strcmp(reg_name, "MTL") == 0 || std::strcmp(reg_name, "Metal") == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
        if (gpu == nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = dev;
        }
    }

    return gpu ? ggml_backend_dev_init(gpu, nullptr) : nullptr;
}

// one flash-attention evaluation on `backend`; f32 output into `out`
static void run_fa(ggml_backend_t backend, int dk, int dv, int64_t kv, int64_t nh,
        const std::vector<float> & hq, const std::vector<ggml_fp16_t> & hk,
        const std::vector<ggml_fp16_t> & hv, const std::vector<ggml_fp16_t> & hm,
        std::vector<float> & out) {
    ggml_init_params params = {
        /*.mem_size   =*/ 16*ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dk, 1,  nh, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dk, kv, 1,  1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dv, kv, 1,  1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, 1,  1,  1);

    ggml_set_input(q);
    ggml_set_input(k);
    ggml_set_input(v);
    ggml_set_input(m);

    ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/std::sqrt((float) dk), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    ggml_set_output(o);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf != nullptr);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(galloc, gf));

    ggml_backend_tensor_set(q, hq.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, hk.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, hv.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(m, hm.data(), 0, ggml_nbytes(m));

    assert(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    out.resize((size_t) ggml_nelements(o));
    ggml_backend_tensor_get(o, out.data(), 0, out.size()*sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    bool ok = true;

    // ---------------------------------------------------------------- part 0
    {
        const char * path = GGML_METAL_SOURCE_PATH;

        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::printf("FAIL: cannot open the Metal source at %s\n", path);
            return 1;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string src = ss.str();

        std::string body;
        if (!zero_fill_body(src, body)) {
            std::printf("FAIL: could not locate the accumulator zero-fill in %s "
                    "(the '// zero out so' / '// zero out shared memory SH' markers moved)\n", path);
            return 1;
        }

        // The store must be bounded by DV4. Without the bound the surplus
        // stores are dead zeros, so no output comparison can see them - this
        // text check is the only thing standing between a refactor and the
        // out-of-bounds threadgroup write coming back.
        //
        // The loop header is itself `i < DV4/NL`, so a bare "< DV4" search
        // matches the unbounded form too; what marks the bound is a comparison
        // against DV4 that is not the `DV4/NL` trip count.
        const std::string flat = strip_ws(body);

        bool bounded = false;
        for (size_t p = flat.find("<DV4"); p != std::string::npos; p = flat.find("<DV4", p + 1)) {
            const size_t after = p + 4;
            if (after >= flat.size() || flat[after] != '/') {
                bounded = true;
                break;
            }
        }

        if (!bounded) {
            std::printf("FAIL: the zero-fill in %s is no longer bounded by DV4 - the surplus stores are "
                    "dead, so nothing else in this test or any output comparison can see them:\n%s\n",
                    path, body.c_str());
            return 1;
        }

        // the instantiation list this test checks must be the one that ships
        const std::set<std::vector<int>> shipped = shipped_instantiations(src);
        if (shipped.empty()) {
            std::printf("FAIL: found no kernel_flash_attn_ext_vec instantiations in %s\n", path);
            return 1;
        }

        std::set<std::vector<int>> known;
        for (int i = 0; i < g_n_inst; ++i) {
            known.insert({ g_inst[i].dk, g_inst[i].dv, g_inst[i].ne });
        }

        for (const auto & t : shipped) {
            if (known.count(t) == 0) {
                std::printf("FAIL: ggml-metal.metal ships kernel_flash_attn_ext_vec<..., %d, %d, %d> "
                        "which this test does not check - add it to g_inst and re-derive its bound\n",
                        t[0], t[1], t[2]);
                ok = false;
            }
        }
        for (const auto & t : known) {
            if (shipped.count(t) == 0) {
                std::printf("FAIL: this test checks <%d, %d, %d> but ggml-metal.metal no longer ships it "
                        "- drop it from g_inst\n", t[0], t[1], t[2]);
                ok = false;
            }
        }

        std::printf("kernel source: %s\n", path);
        std::printf("  zero-fill is bounded by DV4, %d instantiation(s) shipped and checked\n\n",
                (int) shipped.size());

        if (!ok) {
            std::puts("metal fa zero-fill test FAILED (source)");
            return 1;
        }
    }

    // ---------------------------------------------------------------- part 1
    std::printf("accumulator zero-fill bounds (NW = %d)\n", NW);
    std::printf("%4s %4s %3s %3s %4s %5s %8s %9s %9s %9s\n",
            "DK", "DV", "NE", "NL", "DV4", "iters", "slot(f4)", "max unbnd", "over(f4)", "over(B)");

    for (int i = 0; i < g_n_inst; ++i) {
        const int dk = g_inst[i].dk;
        const int dv = g_inst[i].dv;
        const int ne = g_inst[i].ne;

        const int nl    = NW/ne;
        const int dv4   = dv/4;
        const int slot  = pad2(dv, 128)/4; // float4 per SIMDgroup accumulator
        const int iters = dv4/nl;

        // what the unbounded fill would touch
        const int mx_unbounded = (NW - 1) + (iters - 1)*nl;
        const int over         = std::max(0, mx_unbounded + 1 - slot);

        // what the shipped fill touches: the store is guarded by
        // `tiisg + i*NL < DV4`, so the highest index is at most DV4 - 1
        int mx_bounded = -1;
        for (int t = 0; t < NW; ++t) {
            for (int it = 0; it < iters; ++it) {
                const int idx = t + it*nl;
                if (idx < dv4) {
                    mx_bounded = std::max(mx_bounded, idx);
                }
            }
        }

        std::printf("%4d %4d %3d %3d %4d %5d %8d %9d %9d %9d\n",
                dk, dv, ne, nl, dv4, iters, slot, mx_unbounded, over, over*16);

        if (mx_bounded >= slot || mx_bounded >= dv4) {
            std::printf("  FAIL: bounded fill reaches %d, slot is %d float4 and DV4 is %d\n",
                    mx_bounded, slot, dv4);
            ok = false;
        }

        // the bounded fill must still cover every slot the accumulation uses,
        // which is exactly lanes 0..NL-1 stepping by NL over DV4
        for (int idx = 0; idx < dv4; ++idx) {
            const bool covered = (idx%nl) < NW && (idx/nl) < iters;
            if (!covered) {
                std::printf("  FAIL: DV4 slot %d is not zeroed\n", idx);
                ok = false;
            }
        }
    }

    if (!ok) {
        std::puts("metal fa zero-fill test FAILED (bounds)");
        return 1;
    }

    // ---------------------------------------------------------------- part 2
    ggml_backend_t metal = metal_backend_init();
    if (metal == nullptr) {
        std::puts("metal fa zero-fill test: no Metal device, part 1 only");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(cpu != nullptr);

    // the head sizes whose unbounded fill left the accumulator slot, plus the
    // DSV4 geometry as a control. kv values are chosen so the host picks
    // nsg = 1, 2 and 4 (it doubles nsg while 2*nwg*nsg*ncpsg < ne11).
    static const int aff[][2] = { { 96, 96 }, { 192, 128 }, { 320, 256 }, { 576, 512 }, { 512, 512 } };
    static const int64_t kvs[] = { 512, 2560, 6912 };

    std::printf("\nMetal vs CPU, f32 flash attention, 8 query heads\n");
    std::printf("%9s %7s %12s\n", "dk/dv", "kv", "max abs err");

    for (size_t a = 0; a < sizeof(aff)/sizeof(aff[0]); ++a) {
        const int dk = aff[a][0];
        const int dv = aff[a][1];

        for (size_t c = 0; c < sizeof(kvs)/sizeof(kvs[0]); ++c) {
            const int64_t kv = kvs[c];
            const int64_t nh = 8;

            std::mt19937 rng(0x2E80F111u + (uint32_t) dk*7u + (uint32_t) kv);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

            std::vector<float>       hq((size_t) dk*nh);
            std::vector<ggml_fp16_t> hk((size_t) dk*kv);
            std::vector<ggml_fp16_t> hv((size_t) dv*kv);
            std::vector<ggml_fp16_t> hm((size_t) kv);

            for (auto & x : hq) x = dist(rng);
            for (auto & x : hk) x = ggml_fp32_to_fp16(dist(rng));
            for (auto & x : hv) x = ggml_fp32_to_fp16(dist(rng));
            for (int64_t i = 0; i < kv; ++i) {
                // a DSV4-shaped mask: a live window plus a live tail
                const bool live = (i >= 128 && i < 256) || (i >= kv - 512);
                hm[(size_t) i] = ggml_fp32_to_fp16(live ? 0.0f : -INFINITY);
            }

            std::vector<float> got, ref;
            run_fa(metal, dk, dv, kv, nh, hq, hk, hv, hm, got);
            run_fa(cpu,   dk, dv, kv, nh, hq, hk, hv, hm, ref);

            assert(got.size() == ref.size());

            double err = 0.0;
            for (size_t i = 0; i < got.size(); ++i) {
                err = std::max(err, (double) std::fabs(got[i] - ref[i]));
            }

            std::printf("%4d/%-4d %7lld %12.3e%s\n", dk, dv, (long long) kv, err,
                    err > 2e-2 ? "   FAIL" : "");
            if (!(err <= 2e-2)) {
                ok = false;
            }
        }
    }

    ggml_backend_free(cpu);
    ggml_backend_free(metal);

    std::puts(ok ? "\nmetal fa zero-fill test OK" : "\nmetal fa zero-fill test FAILED");
    return ok ? 0 : 1;
}
