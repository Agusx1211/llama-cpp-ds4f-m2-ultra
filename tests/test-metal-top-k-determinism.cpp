// Regression test for the DeepSeek V4 Lightning Indexer top-k selection.
//
// ggml_metal_op_top_k routes k == 512 rows longer than 1024 elements to
// kernel_top_k_radix_f32_i32_impl. That kernel finds the k-th largest key with
// an MSD radix walk and then collects the partition. Because the indexer score
// row is ReLU'd, the pivot is routinely exactly 0.0f with a tie group thousands
// of elements wide, so the kernel has to *choose* which tied elements to keep.
//
// The original implementation appended both the strictly-greater and the tied
// elements to a threadgroup array using an atomic counter, i.e. in GPU arrival
// order. That made both the retained set and its order a function of thread
// scheduling: the same DeepSeek V4 Flash prompt produced different attention
// inputs, different logits and a different transcript on every run once the
// compressed cache passed 1024 rows (~4k prompt tokens). See
// notes/2026-08-08-dsv4-depth-determinism.md.
//
// This test pins the contract that replaced it:
//   1. repeated evaluations of the same row return byte-identical indices;
//   2. the result equals a host reference that reproduces the selected set
//      exactly and orders it by a strict total order, so no ties are left.
//
// *Which* members of the tie group fill the quota is a free parameter selected
// by LLAMA_DSV4_TOPK_TIE:
//   asc  - the oldest tied keys (smallest source indices); the default;
//   desc - the newest tied keys (largest source indices);
//   hash - a deterministic unbiased sample of the tie group.
// All three must satisfy 1 and 2 against their own reference, so this test
// forks one child per policy and runs the whole case list in each. A child
// whose knob is wired wrong fails the reference check, because the test also
// asserts that the three references genuinely differ on every tie case.
//
// The test skips itself when no Metal device is present, so it is safe to run
// in CTest on any host.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#undef assert
#define assert(expr) do {                                                      \
    if (!(expr)) {                                                             \
        std::fprintf(stderr, "check failed at %s:%d: %s\n",                    \
                __FILE__, __LINE__, #expr);                                    \
        std::abort();                                                          \
    }                                                                          \
} while (false)

static ggml_backend_t metal_backend_init() {
    ggml_backend_load_all();

    // the Metal device reports the GPU's marketing name ("Apple M2 Ultra"),
    // so select on the backend registration (or, failing that, on any GPU)
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : "?";
        // the Metal registration is named "MTL" in this tree
        if (std::strcmp(reg_name, "MTL") == 0 || std::strcmp(reg_name, "Metal") == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
        if (gpu == nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = dev;
        }
    }

    return gpu ? ggml_backend_dev_init(gpu, nullptr) : nullptr;
}

enum tie_policy {
    TIE_ASC  = 0,
    TIE_DESC = 1,
    TIE_HASH = 2,
};

static const char * const tie_policy_name[3] = { "asc", "desc", "hash" };

// the kernel's IEEE-754 -> monotone unsigned key transform
static uint32_t order_key(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b ^ ((uint32_t) ((int32_t) b >> 31) | 0x80000000u);
}

// must match top_k_tie_hash() in ggml-metal.metal exactly
static uint32_t tie_hash(uint32_t x) {
    x += 0x9e3779b9u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Host reference. Reproduces the kernel's selection rule (every strictly
// greater element, then the policy's share of the pivot tie group) and then
// puts the retained set into the kernel's output order: value descending, ties
// by source index (descending under TIE_DESC, ascending otherwise). Both are
// strict total orders over distinct indices, so the reference is unique.
//
// *tie_slots receives how many of the k output slots had to be filled from the
// tie group, and *tie_width the size of that group - the two numbers that
// decide whether the policy can matter at all for a given row.
static std::vector<int32_t> reference_top_k(const float * row, int64_t n, int k, int policy,
                                            int * tie_slots = nullptr, int * tie_width = nullptr) {
    std::vector<uint32_t> keys((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        keys[(size_t) i] = order_key(row[i]);
    }

    std::vector<uint32_t> tmp = keys;
    std::nth_element(tmp.begin(), tmp.begin() + (k - 1), tmp.end(), std::greater<uint32_t>());
    const uint32_t pivot = tmp[(size_t) (k - 1)];

    std::vector<int32_t> gt;
    std::vector<int32_t> eq;
    for (int64_t i = 0; i < n; ++i) {
        if (keys[(size_t) i] > pivot) {
            gt.push_back((int32_t) i);
        } else if (keys[(size_t) i] == pivot) {
            eq.push_back((int32_t) i);
        }
    }

    const size_t quota = gt.size() < (size_t) k ? (size_t) k - gt.size() : 0;
    if (tie_slots) *tie_slots = (int) quota;
    if (tie_width) *tie_width = (int) eq.size();

    std::vector<int32_t> sel = gt;
    sel.resize(std::min(sel.size(), (size_t) k));

    if (policy == TIE_DESC) {
        for (size_t j = 0; j < quota && j < eq.size(); ++j) {
            sel.push_back(eq[eq.size() - 1 - j]);
        }
    } else if (policy == TIE_HASH) {
        const uint32_t thr = eq.empty() || quota >= eq.size()
            ? 0xffffffffu
            : (uint32_t) (((uint64_t) quota << 32)/(uint64_t) eq.size());
        std::vector<int32_t> hit;
        std::vector<int32_t> miss;
        for (int32_t i : eq) {
            (tie_hash((uint32_t) i) < thr ? hit : miss).push_back(i);
        }
        for (size_t j = 0; j < quota && j < hit.size(); ++j) {
            sel.push_back(hit[j]);
        }
        for (size_t j = 0; sel.size() < (size_t) k && j < miss.size(); ++j) {
            sel.push_back(miss[j]);
        }
    } else {
        for (size_t j = 0; j < quota && j < eq.size(); ++j) {
            sel.push_back(eq[j]);
        }
    }

    // the kernel leaves any unfilled slot at 0
    sel.resize((size_t) k, 0);

    const bool desc_order = policy == TIE_DESC;
    std::sort(sel.begin(), sel.end(), [row, desc_order](int32_t a, int32_t b) {
        if (row[a] != row[b]) {
            return row[a] > row[b];
        }
        return desc_order ? a > b : a < b;
    });
    return sel;
}

// A Lightning-Indexer-shaped score row: a few distinct positive scores, a very
// wide exact-zero tie group (the ReLU floor), and a masked tail at -inf.
static void fill_indexer_row(float * row, int64_t n, int n_positive, int n_masked, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.25f, 8.0f);

    for (int64_t i = 0; i < n; ++i) {
        row[i] = 0.0f;
    }
    for (int64_t i = n - n_masked; i < n; ++i) {
        row[i] = -INFINITY;
    }

    // spread the positives over the unmasked prefix at deterministic positions
    const int64_t span = n - n_masked;
    for (int i = 0; i < n_positive; ++i) {
        row[(int64_t) i*span/n_positive] = dist(rng);
    }
}

struct case_desc {
    int64_t     ne00;
    int64_t     ne01;
    int         k;
    int         n_positive;
    int         n_masked;
    // only the radix kernel promises the (value desc, index asc) tie-break;
    // the generic bitonic path breaks ties by network position, so it is
    // checked for reproducibility only
    bool        check_reference;
    const char * name;
};

static bool run_case(ggml_backend_t backend, const case_desc & c, int policy) {
    const size_t ctx_size = 4*ggml_tensor_overhead() + ggml_graph_overhead();

    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne00, c.ne01);
    ggml_set_input(src);
    ggml_tensor * dst = ggml_top_k(ctx, src, c.k);
    ggml_set_output(dst);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf != nullptr);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(galloc, gf));

    std::vector<float> host_src((size_t) c.ne00*c.ne01);
    for (int64_t r = 0; r < c.ne01; ++r) {
        fill_indexer_row(host_src.data() + r*c.ne00, c.ne00, c.n_positive, c.n_masked, 1234u + (uint32_t) r);
    }
    ggml_backend_tensor_set(src, host_src.data(), 0, ggml_nbytes(src));

    const size_t n_out = (size_t) c.k*c.ne01;
    std::vector<int32_t> first(n_out);
    std::vector<int32_t> cur(n_out);

    bool ok = true;
    const int n_runs = 8;
    for (int run = 0; run < n_runs; ++run) {
        assert(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(dst, cur.data(), 0, n_out*sizeof(int32_t));

        if (run == 0) {
            first = cur;
            continue;
        }
        if (cur != first) {
            size_t d = 0;
            while (d < n_out && cur[d] == first[d]) {
                ++d;
            }
            std::fprintf(stderr,
                    "%s: run %d differs from run 0 at element %zu (%d vs %d)\n",
                    c.name, run, d, (int) first[d], (int) cur[d]);
            ok = false;
            break;
        }
    }

    int tie_slots = 0;
    int tie_width = 0;
    if (ok && c.check_reference) {
        for (int64_t r = 0; r < c.ne01 && ok; ++r) {
            const float * hrow = host_src.data() + r*c.ne00;
            const auto ref = reference_top_k(hrow, c.ne00, c.k, policy, &tie_slots, &tie_width);
            for (int i = 0; i < c.k; ++i) {
                if (first[(size_t) r*c.k + i] != ref[i]) {
                    std::fprintf(stderr,
                            "%s [%s]: row %lld element %d = %d, reference = %d\n",
                            c.name, tie_policy_name[policy], (long long) r, i,
                            (int) first[(size_t) r*c.k + i], (int) ref[i]);
                    ok = false;
                    break;
                }
            }

            // A tie case is only a meaningful gate if the three policies pick
            // genuinely different sets; otherwise a mis-wired knob would go
            // unnoticed. Assert that here rather than trusting the fixture.
            if (ok && tie_slots > 1) {
                const auto ref_asc  = reference_top_k(hrow, c.ne00, c.k, TIE_ASC);
                const auto ref_desc = reference_top_k(hrow, c.ne00, c.k, TIE_DESC);
                const auto ref_hash = reference_top_k(hrow, c.ne00, c.k, TIE_HASH);
                assert(ref_asc != ref_desc && ref_asc != ref_hash && ref_desc != ref_hash);
            }
        }
    }

    std::printf("[%-4s] %-28s ne00=%5lld ne01=%lld k=%d positives=%d masked=%d "
            "tie_slots=%d/%d tie_width=%d ref=%s : %s\n",
            tie_policy_name[policy], c.name, (long long) c.ne00, (long long) c.ne01, c.k,
            c.n_positive, c.n_masked, tie_slots, c.k, tie_width,
            c.check_reference ? "yes" : "no", ok ? "OK" : "FAILED");

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return ok;
}

// ne00 > 1024 with k == 512 is exactly the routing condition for the radix
// kernel in ggml_metal_op_top_k. 1024 stays on the bitonic path and is kept
// as a control: it must be reproducible too.
static const case_desc g_cases[] = {
    { 1024, 2, 512,  64,   0, false, "bitonic-control"          },
    { 2048, 2, 512, 100,   0, true,  "radix/wide-zero-tie"      },
    { 4096, 3, 512,  50, 512, true,  "radix/zero-tie+masked"    },
    { 9472, 2, 512, 700, 256, true,  "radix/deep-context-like"  },
    { 4096, 2, 512, 900,   0, true,  "radix/mostly-positive"    },
};

// runs the whole case list under one policy; the policy is already in the
// environment when this is called
static int run_policy(int policy) {
    ggml_backend_t backend = metal_backend_init();
    if (backend == nullptr) {
        std::puts("metal top-k determinism test skipped: no Metal device");
        return 0;
    }

    bool ok = true;
    for (const auto & c : g_cases) {
        ok = run_case(backend, c, policy) && ok;
    }

    ggml_backend_free(backend);

    std::printf("policy %s: %s\n", tie_policy_name[policy], ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}

static std::string self_path(const char * argv0) {
#ifdef __APPLE__
    char     buf[4096];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0) {
        return std::string(buf);
    }
#endif
    return argv0 ? std::string(argv0) : std::string();
}

int main(int argc, char ** argv) {
    // A child already told which policy to run.
    if (const char * child = std::getenv("LLAMA_DSV4_TOPK_TIE_CHILD")) {
        return run_policy(std::atoi(child));
    }

    // One child process per policy: LLAMA_DSV4_TOPK_TIE is read once per
    // process by ggml_metal_op_top_k, so the policies cannot be interleaved in
    // a single Metal context. The child must exec, not just fork: this binary
    // links Foundation/Metal, and macOS aborts CoreFoundation use in a forked
    // child that has not exec'd.
    const std::string exe = self_path(argc > 0 ? argv[0] : nullptr);
    if (exe.empty()) {
        std::puts("metal top-k determinism test: cannot locate own executable");
        return 1;
    }

    bool ok = true;
    for (int policy = 0; policy < 3; ++policy) {
        char child_val[2] = { (char) ('0' + policy), '\0' };
        setenv("LLAMA_DSV4_TOPK_TIE",       tie_policy_name[policy], 1);
        setenv("LLAMA_DSV4_TOPK_TIE_CHILD", child_val,               1);

        std::fflush(stdout);
        const pid_t pid = fork();
        if (pid < 0) {
            std::perror("fork");
            return 1;
        }
        if (pid == 0) {
            char * const args[] = { (char *) exe.c_str(), nullptr };
            execv(exe.c_str(), args);
            _exit(127);
        }
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            std::perror("waitpid");
            return 1;
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            std::printf("policy %s: child failed (status %d)\n", tie_policy_name[policy], status);
            ok = false;
        }
    }
    unsetenv("LLAMA_DSV4_TOPK_TIE_CHILD");

    if (!ok) {
        std::puts("metal top-k determinism test FAILED");
        return 1;
    }

    std::puts("metal top-k determinism test OK (asc, desc, hash)");
    return 0;
}
