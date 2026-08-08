#include "ggml-metal-device.h"

#include "ggml-metal-impl.h"

#include "ggml-impl.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

struct ggml_metal_device_deleter {
    void operator()(ggml_metal_device_t ctx) {
        ggml_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_metal_device, ggml_metal_device_deleter> ggml_metal_device_ptr;

ggml_metal_device_t ggml_metal_device_get(int device) {
    static std::vector<ggml_metal_device_ptr> devs;

    devs.emplace_back(ggml_metal_device_init(device));

    return devs.back().get();
}

struct ggml_metal_pipelines {
    std::unordered_map<std::string, ggml_metal_pipeline_t> data;
};

ggml_metal_pipelines_t ggml_metal_pipelines_init(void) {
    ggml_metal_pipelines_t res = new ggml_metal_pipelines();

    return res;
}

void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls) {
    if (!ppls) {
        return;
    }

    for (auto it = ppls->data.begin(); it != ppls->data.end(); ++it) {
        ggml_metal_pipeline_free(it->second);
    }

    delete ppls;
}

void ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline) {
    ppls->data[name] = pipeline;
}

ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name) {
    if (ppls->data.find(name) == ppls->data.end()) {
        return nullptr;
    }

    return ppls->data[name];
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base(ggml_metal_library_t lib, ggml_op op) {
    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op) {
        case GGML_OP_ADD_ID: op_str = "add_id"; break;
        case GGML_OP_DSV4_COMPRESS: op_str = "dsv4_compress"; break;
        case GGML_OP_DSV4_TOP_K_MASK: op_str = "dsv4_top_k_mask"; break;
        case GGML_OP_DSV4_INDEXED_CONCAT: op_str = "dsv4_indexed_concat"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s", op_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cpy_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d(ggml_metal_library_t lib, const ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    const char * pool_str = "undefined";
    switch (op_pool) {
        case GGML_OP_POOL_AVG: pool_str = "avg"; break;
        case GGML_OP_POOL_MAX: pool_str = "max"; break;
        default: GGML_ASSERT(false && "not implemented");
    };

    char base[256];
    char name[256];

    snprintf(base, sizeof(base), "kernel_pool_1d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, sizeof(name), "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d(ggml_metal_library_t lib, const ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    const char * pool_str = "undefined";
    switch (op_pool) {
        case GGML_OP_POOL_AVG: pool_str = "avg"; break;
        case GGML_OP_POOL_MAX: pool_str = "max"; break;
        default: GGML_ASSERT(false && "not implemented");
    };

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pool_2d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const auto tsrc = op->src[0]->type;
    const auto tidx = op->src[1]->type;
    const auto tdst = op->type;

    snprintf(base, 256, "kernel_set_rows_%s_%s_%s", ggml_type_name(tsrc), ggml_type_name(tidx), ggml_type_name(tdst));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int n = op->src[0]->ne[0];

    snprintf(base, 256, "kernel_diag_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_n=%d", base, n);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.nsg  = 1;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_repeat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_concat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_SCALE:      op_num = OP_UNARY_NUM_SCALE;      break;
        case GGML_OP_FILL:       op_num = OP_UNARY_NUM_FILL;       break;
        case GGML_OP_CLAMP:      op_num = OP_UNARY_NUM_CLAMP;      break;
        case GGML_OP_SQR:        op_num = OP_UNARY_NUM_SQR;        break;
        case GGML_OP_SQRT:       op_num = OP_UNARY_NUM_SQRT;       break;
        case GGML_OP_SIN:        op_num = OP_UNARY_NUM_SIN;        break;
        case GGML_OP_COS:        op_num = OP_UNARY_NUM_COS;        break;
        case GGML_OP_LOG:        op_num = OP_UNARY_NUM_LOG;        break;
        case GGML_OP_LEAKY_RELU: op_num = OP_UNARY_NUM_LEAKY_RELU; break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:        op_num = OP_UNARY_NUM_TANH;        break;
                case GGML_UNARY_OP_RELU:        op_num = OP_UNARY_NUM_RELU;        break;
                case GGML_UNARY_OP_SIGMOID:     op_num = OP_UNARY_NUM_SIGMOID;     break;
                case GGML_UNARY_OP_GELU:        op_num = OP_UNARY_NUM_GELU;        break;
                case GGML_UNARY_OP_GELU_ERF:    op_num = OP_UNARY_NUM_GELU_ERF;    break;
                case GGML_UNARY_OP_GELU_QUICK:  op_num = OP_UNARY_NUM_GELU_QUICK;  break;
                case GGML_UNARY_OP_SILU:        op_num = OP_UNARY_NUM_SILU;        break;
                case GGML_UNARY_OP_ELU:         op_num = OP_UNARY_NUM_ELU;         break;
                case GGML_UNARY_OP_NEG:         op_num = OP_UNARY_NUM_NEG;         break;
                case GGML_UNARY_OP_ABS:         op_num = OP_UNARY_NUM_ABS;         break;
                case GGML_UNARY_OP_SGN:         op_num = OP_UNARY_NUM_SGN;         break;
                case GGML_UNARY_OP_STEP:        op_num = OP_UNARY_NUM_STEP;        break;
                case GGML_UNARY_OP_HARDSWISH:   op_num = OP_UNARY_NUM_HARDSWISH;   break;
                case GGML_UNARY_OP_HARDSIGMOID: op_num = OP_UNARY_NUM_HARDSIGMOID; break;
                case GGML_UNARY_OP_EXP:         op_num = OP_UNARY_NUM_EXP;         break;
                case GGML_UNARY_OP_SOFTPLUS:    op_num = OP_UNARY_NUM_SOFTPLUS;    break;
                case GGML_UNARY_OP_EXPM1:       op_num = OP_UNARY_NUM_EXPM1;       break;
                case GGML_UNARY_OP_FLOOR:       op_num = OP_UNARY_NUM_FLOOR;       break;
                case GGML_UNARY_OP_CEIL:        op_num = OP_UNARY_NUM_CEIL;        break;
                case GGML_UNARY_OP_ROUND:       op_num = OP_UNARY_NUM_ROUND;       break;
                case GGML_UNARY_OP_TRUNC:       op_num = OP_UNARY_NUM_TRUNC;       break;
                case GGML_UNARY_OP_XIELU:       op_num = OP_UNARY_NUM_XIELU;       break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;
    const bool is_cnt = ggml_is_contiguous(op->src[0]) && ggml_nelements(op) < 32768;

    snprintf(base, 256, "kernel_unary_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_cnt=%d", base, op_num, is_cnt);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_UNARY + 0);
        ggml_metal_cv_set_bool (cv, is_cnt, FC_UNARY + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_cnt;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op->op) {
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:        op_str = "reglu";        break;
                case GGML_GLU_OP_GEGLU:        op_str = "geglu";        break;
                case GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       break;
                case GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   break;
                case GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    break;
                case GGML_GLU_OP_GEGLU_QUICK:  op_str = "geglu_quick";  break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_swiglu_clamp(ggml_metal_library_t lib) {
    const char * name = "kernel_dsv4_swiglu_clamp_f32";

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_op_sum_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_SUM_ROWS: op_num = OP_SUM_ROWS_NUM_SUM_ROWS; break;
        case GGML_OP_MEAN:     op_num = OP_SUM_ROWS_NUM_MEAN;     break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;

    snprintf(base, 256, "kernel_sum_rows_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d", base, op_num);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_SUM_ROWS + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*sizeof(float);

    if (is_c4) {
        res.smem *= 4;
    }

    res.c4  = is_c4;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cumsum_blk_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cumsum_add_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_TRI);
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    const char * op_str = "tri";
    const int ttype = op->op_params[0];

    snprintf(base, 256, "kernel_%s_%s_%d", op_str, ggml_type_name(op->src[0]->type), ttype);

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(!op->src[1] || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);

    char base[256];
    char name[256];

    const char * suffix = "";

    if (op->src[0]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    const ggml_type tsrc1 = op->src[1] ? op->src[1]->type : GGML_TYPE_F32;

    snprintf(base, 256, "kernel_soft_max_%s%s", ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc(ggml_metal_library_t lib, ggml_op op) {
    const char * name = nullptr;

    switch (op) {
        case GGML_OP_DSV4_HC_COMB: name = "kernel_dsv4_hc_comb_f32"; break;
        case GGML_OP_DSV4_HC_PRE:  name = "kernel_dsv4_hc_pre_f32";  break;
        case GGML_OP_DSV4_HC_POST: name = "kernel_dsv4_hc_post_f32"; break;
        default: GGML_ABORT("fatal error");
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc_pre_norm(ggml_metal_library_t lib) {
    const char * name = "kernel_dsv4_hc_pre_norm_f32";

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_router(ggml_metal_library_t lib) {
    const char * name = "kernel_dsv4_router_f32";

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_sparse_pack(
        ggml_metal_library_t lib, ggml_type type, bool indexed) {
    char name[256];
    snprintf(name, 256, "kernel_dsv4_%ssparse_pack_%s", indexed ? "indexed_" : "", ggml_type_name(type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer(ggml_metal_library_t lib, ggml_type type, bool tail) {
    const char * name = nullptr;

    switch (type) {
        case GGML_TYPE_F16:  name = tail ? "kernel_lightning_indexer_f16_tail" : "kernel_lightning_indexer_f16"; break;
        case GGML_TYPE_Q8_0: name = tail ? "kernel_lightning_indexer_q8_0_tail" : "kernel_lightning_indexer_q8_0"; break;
        default: GGML_ABORT("fatal error");
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    const char * suffix = "";

    if (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched(ggml_metal_library_t lib, const ggml_tensor * op, int ssm_conv_bs) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s_batched%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_ssm_conv_bs=%d", base, ssm_conv_bs);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, ssm_conv_bs, FC_SSM_CONV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan(ggml_metal_library_t lib, const ggml_tensor * op)  {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);

    char base[256];
    char name[256];

    const int nsg = (ne00 + 31)/32;

    snprintf(base, 256, "kernel_ssm_scan_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    // Shared memory layout:
    // - sgptg * NW floats for partial sums (nsg * 32)
    // - sgptg floats for shared_x_dt (nsg)
    // - sgptg floats for shared_dA (nsg)
    // Total: nsg * (32 + 2) floats
    res.smem = (32 + 2)*sizeof(float)*nsg;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    switch (op->op) {
        case GGML_OP_RWKV_WKV6:
            {
                GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv6_%s", ggml_type_name(op->src[0]->type));
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv7_%s", ggml_type_name(op->src[0]->type));
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    // v is src[2], dimensions: S_v = ne[0], H = ne[1]
    const int ne20 = op->src[2]->ne[0]; // S_v
    const int ne21 = op->src[2]->ne[1]; // H
    const int ne30 = op->src[3]->ne[0]; // G
    // state is src[5], 4D [S_v, S_v, H_v, n_seqs] (s0 only); K is op param 0.
    const int K = ggml_get_op_params_i32(op, 0);

    const int nsg = op->src[2]->ne[0]/32;

    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->ne[0] == ne20 * ne21);
    GGML_ASSERT(ne20 % 32 == 0);

    snprintf(base, 256, "kernel_gated_delta_net_%s_%d", ggml_type_name(op->src[0]->type), nsg);
    snprintf(name, 256, "%s_ne20=%d_ne30=%d_K=%d", base, ne20, ne30, K);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, ne20, FC_GATED_DELTA_NET + 0);
        ggml_metal_cv_set_int16(cv, ne30, FC_GATED_DELTA_NET + 1);
        ggml_metal_cv_set_int16(cv, K,    FC_GATED_DELTA_NET + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg = nsg;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int nsg = 8;
    const int n   = op->src[1]->ne[1];
    const int k   = op->src[1]->ne[0];

    snprintf(base, 256, "kernel_solve_tri_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d_n=%d_k=%d", base, nsg, n, k);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_SOLVE_TRI + 0);
        ggml_metal_cv_set_int16(cv, n,   FC_SOLVE_TRI + 1);
        ggml_metal_cv_set_int16(cv, k,   FC_SOLVE_TRI + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = GGML_PAD(GGML_PAD(n, 32)*nsg*sizeof(float), 16);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, const ggml_tensor * op, int nsg, int nxpsg, int r1ptg) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const int       ne12  = op->src[1]->ne[2];
    const int       r2    = ne12 / op->src[0]->ne[2];
    const int       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d_ne12=%d_r2=%d_r3=%d", base, nsg, nxpsg, ne12, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, nxpsg,          FC_MUL_MV + 1);
        ggml_metal_cv_set_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, (int16_t) r2,   FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, (int16_t) r3,   FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    // fork: the dedicated E4M3_M2 ext kernel stages its 256-entry decode LUT
    // in threadgroup memory (the generic template kernels take no threadgroup
    // buffer; smem stays 0 for them)
    if (tsrc0 == GGML_TYPE_E4M3_M2) {
        res.smem = 256*sizeof(float);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const bool bc_inp = op->src[0]->ne[0] % 32 != 0;

    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;

    const bool has_tensor = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_tensor;

    // fork: the dedicated E4M3_M2 GEMM processes NT1 32-column B tiles per
    // threadgroup against one staged+decoded A tile (amortizes the e4m3
    // decode and the A refetch across 32*NT1 columns instead of 32).
    // ROUND 2 (notes/2026-08-07-gguf-m2-artifact-format-design.md): NT1=2
    // measured 1.03-1.09x BF16 in ISOLATION but collapsed IN-GRAPH (-28%
    // end-to-end prefill, 2.4x 30-token prompt evals). The bench-graph
    // concurrency harness (test-m2-e4m3 bench-graph) reproduced the
    // collapse: whenever ne1 % 64 != 0 the nt2 pipeline compiles its
    // bounds-checked specialization at th_max 576 (vs 1024 for the BF16
    // template) and the whole mixed-op graph runs 2.6x (n=664) to 3.5x
    // (n=30) BF16; even aligned it sits at th_max 640 with the second
    // B tile fully wasted at small ne1 (branch-free staging + MMA run
    // regardless). NT1=1 with the same u16c decode keeps th_max 896-1024
    // and measured 1.05x total-graph at every ne1 probed (30/512/664/2048).
    // DEFAULT: NT1=1 everywhere. NT1=2 stays selectable for A/B:
    // GGML_E4M3_MM_NT1=2 forces it, GGML_E4M3_MM_NT2_MIN_N=<n> re-enables
    // it for tile-aligned ops with ne1 >= n (in-bench it is 1-4pp faster
    // than NT1=1 at aligned n >= 512, but that is exactly the regime where
    // the bench could NOT reproduce round 1's aligned-prefill deficit, so
    // enabling it needs an end-to-end A/B first).
    static const int e4m3_nt2_min_n = [] {
        const char * s = getenv("GGML_E4M3_MM_NT2_MIN_N");
        return s != NULL ? atoi(s) : 0; // 0 = NT1=2 never (round-2 default)
    }();
    // the NT2_MIN_N lever only ever selects NT1=2 for fully tile-aligned
    // outputs: the nt2 bco=1 specialization is the measured 2.6-3.5x
    // in-graph catastrophe (bench-graph, n=664/30) and must be unreachable
    int e4m3_nt1 = (e4m3_nt2_min_n > 0 && op->ne[1] >= e4m3_nt2_min_n &&
                    op->ne[0] % 64 == 0 && op->ne[1] % 64 == 0) ? 2 : 1;
    if (tsrc0 == GGML_TYPE_E4M3_M2 && getenv("GGML_E4M3_MM_NT1") != NULL) {
        // NT1=4 was measured DEAD (mc[32] register collapse, 6.4-8.4x) and
        // its instantiations were removed — only 1 and 2 exist
        e4m3_nt1 = atoi(getenv("GGML_E4M3_MM_NT1")) == 2 ? 2 : 1;
    }
    if (tsrc0 == GGML_TYPE_E4M3_M2 && getenv("GGML_E4M3_MM_TMPL") != NULL) {
        e4m3_nt1 = 1; // the shared-template probe is a 64x32 tile
    }

    const bool bc_out = tsrc0 == GGML_TYPE_E4M3_M2
        ? (op->ne[0] % 64  != 0 || op->ne[1] % (e4m3_nt1*32) != 0)
        : tsrc0 == GGML_TYPE_NF8_M2 // fork: dedicated single-tile 64x32 kernel
        ? (op->ne[0] % 64  != 0 || op->ne[1] % 32  != 0)
        : has_tensor
        ? (op->ne[0] % NRA != 0 || op->ne[1] % NRB != 0)
        : (op->ne[0] % 64  != 0 || op->ne[1] % 32  != 0);

    GGML_ASSERT(op->src[1]->ne[2] <= INT16_MAX && op->src[1]->ne[3] <= INT16_MAX);
    const int16_t ne12 = (int16_t) op->src[1]->ne[2];
    const int16_t ne13 = (int16_t) op->src[1]->ne[3];
    const int16_t r2   = (int16_t) (ne12 / op->src[0]->ne[2]);
    const int16_t r3   = (int16_t) (ne13 / op->src[0]->ne[3]);

    // decode-style A/B (attribution knobs; the DEFAULT for E4M3_M2 is
    // "_u16c" — constant-memory u16 bf16-bits LUT + integer exponent add,
    // the measured winner): GGML_E4M3_MM_STYLE in {tg, cl, u16, u16c}
    const char * e4m3_style = "_u16c";
    size_t       e4m3_lut_bytes = 0;
    if (tsrc0 == GGML_TYPE_E4M3_M2 && getenv("GGML_E4M3_MM_STYLE") != NULL) {
        const char * s = getenv("GGML_E4M3_MM_STYLE");
        if (strcmp(s, "tg") == 0) {
            e4m3_style = "";
            e4m3_lut_bytes = 1024;
        } else if (strcmp(s, "cl") == 0) {
            e4m3_style = "_cl";
        } else if (strcmp(s, "u16") == 0) {
            e4m3_style = "_u16";
            e4m3_lut_bytes = 512;
        } else {
            e4m3_style = "_u16c";
        }
    }

    // GGML_E4M3_MM_SHARED=1: shared-sb two-tile variant (u16/u16c styles only)
    const bool e4m3_shared = tsrc0 == GGML_TYPE_E4M3_M2 && getenv("GGML_E4M3_MM_SHARED") != NULL &&
                             (strcmp(e4m3_style, "_u16") == 0 || strcmp(e4m3_style, "_u16c") == 0);
    if (e4m3_shared) {
        e4m3_nt1 = 2;
    }

    if (tsrc0 == GGML_TYPE_E4M3_M2 && getenv("GGML_E4M3_MM_TMPL") != NULL) {
        // A/B probe: the shared-template instantiation (part-2 ship state)
        e4m3_nt1 = 1;
        snprintf(base, 256, "kernel_mul_mm_%s_%s_tmpl", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    } else if (tsrc0 == GGML_TYPE_E4M3_M2) {
        snprintf(base, 256, "kernel_mul_mm_%s_%s_nt%d%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), e4m3_nt1,
                 e4m3_shared ? "s" : "", e4m3_style);
    } else if (tsrc0 == GGML_TYPE_NF8_M2) {
        // fork: NF8_M2 GEMM is single-tile only (NT1=2 is deliberately not
        // implemented for this type: it collapsed in-graph prefill by -28%
        // for E4M3_M2 round 1). GGML_NF8_MM_STYLE=f32 selects the f32-decode
        // attribution variant; the default is the integer bf16-bits decode.
        const char * s = getenv("GGML_NF8_MM_STYLE");
        snprintf(base, 256, "kernel_mul_mm_%s_%s_nt1%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1),
                 (s != NULL && strcmp(s, "f32") == 0) ? "_f32" : "");
    } else {
        snprintf(base, 256, "kernel_mul_mm_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    }
    snprintf(name, 256, "%s_bci=%d_bco=%d_ne12=%d_ne13=%d_r2=%d_r3=%d",
             base, bc_inp, bc_out, ne12, ne13, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_set_bool(cv, bc_out, FC_MUL_MM + 1);
        ggml_metal_cv_set_int16(cv, ne12,  FC_MUL_MM + 2);
        ggml_metal_cv_set_int16(cv, ne13,  FC_MUL_MM + 3);
        ggml_metal_cv_set_int16(cv, r2,    FC_MUL_MM + 4);
        ggml_metal_cv_set_int16(cv, r3,    FC_MUL_MM + 5);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    if (tsrc0 == GGML_TYPE_E4M3_M2) {
        // fork: dedicated kernel (always the classic simdgroup pipeline —
        // the target M2 Ultra has no tensor API): 64 x (NT1*32) tile,
        // sa 4096 + NT1 B tiles of 2048 + 1024-byte decode LUT; the
        // bounds-checked output path stages 64x32 f32 = 8192 bytes over the
        // then-dead sa/sb region.
        res.nr0 = 64;
        res.nr1 = e4m3_nt1*32;

        if (getenv("GGML_E4M3_MM_TMPL") != NULL) {
            res.smem = bc_out ? 8192 : (4096 + 2048); // template geometry
        } else if (e4m3_shared) {
            res.smem = std::max<size_t>(4096 + 2048 + e4m3_lut_bytes, bc_out ? 8192 : 0);
        } else {
            res.smem = std::max<size_t>(4096 + e4m3_nt1*2048 + e4m3_lut_bytes, bc_out ? 8192 : 0);
        }

        // A/B attribution knob: pad the threadgroup allocation to probe pure
        // occupancy effects (concurrent tgs/core = 32 KiB / smem)
        if (getenv("GGML_E4M3_MM_SMEM_PAD") != NULL) {
            res.smem += (size_t) atoi(getenv("GGML_E4M3_MM_SMEM_PAD"));
        }
    } else if (tsrc0 == GGML_TYPE_NF8_M2) {
        // fork: dedicated single-tile kernel with the BF16 template's exact
        // geometry and threadgroup allocation (sa 4096 + sb 2048; no decode
        // LUT bytes — the decode is pure ALU, escape uses constant memory);
        // the bounds-checked output path stages 64x32 f32 = 8192 bytes over
        // the then-dead sa/sb region.
        res.nr0 = 64;
        res.nr1 = 32;

        res.smem = bc_out ? 8192 : (4096 + 2048);
    } else if (has_tensor) {
        res.nr0 = NRA;
        res.nr1 = NRB;

        const size_t smem_a = NRA * N_MM_NK_TOTAL * sizeof(ggml_fp16_t);
        res.smem = smem_a;
    } else {
        res.nr0 = 64;
        res.nr1 = 32;

        res.smem = bc_out ? 8192 : (4096 + 2048);
    }

    res.nsg = N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    return res;
}

// fork: number of simdgroups that actually have work in the shared GEMV
// kernel family (kernel_mul_mv_t_t{,_4}, kernel_mul_mv_e4m3_m2_f32_4,
// kernel_mul_mv_nf8_m2_f32_4).
//
// Those kernels partition a row into NB=32-element blocks and hand each
// simdgroup NF consecutive blocks per round:
//
//     ib0 = sgitg*NF + ix;  for (ib = ib0; ib < ne00/NB; ib += NSG*NF)
//
// so simdgroup `s` executes the loop body zero times whenever
// s*NF >= ne00/NB, i.e. whenever s >= ne00/(NB*NF). Upstream sizes the
// threadgroup as min(4, ceil(ne00/128)), which is only correct for
// NB*NF = 128; the _4 kernels use NF = 16 (span 512 elements) and the
// scalar kernels NF = 8 (span 256), so every row shorter than 4 spans gets
// idle simdgroups.
//
// On DSV4 Flash this is not a corner case: attn_q_b is [1024 x 32768]
// e4m3_m2 and is the single largest GEMV in a decode step (43 dispatches,
// 34.1 MiB of weights each). With ne00 = 1024 the _4 kernel needs 2
// simdgroups and is handed 4, so half of every threadgroup's threads never
// issue a load and the kernel sustains only ~354 GB/s where the same kernel
// on longer rows reaches 540 GB/s.
//
// Shrinking the threadgroup is bit-identical by construction: the mapping
// from simdgroup index to blocks is unchanged (the loop stride NSG*NF only
// matters when more than one round runs, and a row that needs fewer than
// NSG spans runs exactly one round), and the dropped simdgroups contributed
// an exact 0.0f to helper_mv_reduce_and_write's fixed 32-slot simd_sum,
// which zeroes every slot before the partials are written.
// fork: NR0 rows are handled by one threadgroup, so the grid is
// ceil(ne01/NR0) threadgroups. On shapes with very few rows that leaves most
// of the 60-core GPU idle: DSV4's hyper-connection mixers are
// f32 [16384 x 24], i.e. 12 threadgroups at NR0=2, and the decode census
// measures them at 119 GB/s (1.19 ms/step for the 86 dispatches). Dropping to
// one row per threadgroup doubles the grid on exactly those shapes.
//
// Bit-identical: each row's partial sums live in their own sumf[] slot and
// their own reduction slab, so the accumulation order of a row does not
// depend on how many other rows share its threadgroup.
static int ggml_metal_mv_nr0_starved(int nr0, int ne01) {
    static int mode = -1;
    if (mode < 0) {
        const char * v = getenv("GGML_MV_NR0_LEGACY");
        mode = (v && atoi(v) != 0) ? 1 : 0;
    }

    if (mode == 1 || nr0 <= 1) {
        return nr0;
    }

    // 60 GPU cores; below ~1 threadgroup per core the grid, not the memory
    // system, is the limit
    if ((ne01 + nr0 - 1)/nr0 < 64) {
        return 1;
    }

    return nr0;
}

static int ggml_metal_mv_nsg(int ne00, int span, int nsg_max) {
    static int mode = -1;
    if (mode < 0) {
        const char * v = getenv("GGML_MV_NSG_LEGACY");
        mode = (v && atoi(v) != 0) ? 1 : 0;
    }

    if (mode == 1) {
        return std::min(nsg_max, (ne00 + 127) / 128);
    }

    const int nsg = (ne00 + span - 1) / span;

    return std::max(1, std::min(nsg_max, nsg));
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const char * suffix = "";

    // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                if (ne00 < 32) {
                    nsg = 1;
                    nr0 = 32;
                    nr1 = 1;
                    suffix = "_short";
                } else {
                    // fork: NF is 16 in the _4 kernel and 8 in the scalar one,
                    // so a simdgroup spans 512 resp. 256 elements of the row
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                    nsg = ggml_metal_mv_nsg(ne00, ne00 % 4 == 0 ? 512 : 256, 4);
                    nr0 = ggml_metal_mv_nr0_starved(2, ne01);
                    nr1 = 1;
                    smem = 32*sizeof(float)*nr0;
                }
            } break;
        case GGML_TYPE_E4M3_M2:
            {
                // fork: mirrors the F32/F16/BF16 branch above exactly — the
                // E4M3_M2 GEMV is a bit-exact clone of kernel_mul_mv_bf16_f32_4
                // and must be dispatched with the same geometry. smem carries
                // an extra 256-float region: the kernel stages its decode LUT
                // in threadgroup memory after the reduction area.
                GGML_ASSERT(ne00 % 4 == 0); // ne00 % QK_E4M3_M2 == 0 by block invariant
                nsg = ggml_metal_mv_nsg(ne00, 512, 4); // NF = 16 -> 512 elems/simdgroup
                // fork: A/B knob for the NR0 sweep (bit-identical in every
                // value - per-row accumulation order is NR0-independent).
                // Default stays 4; see the note below.
                nr0 = 4; // 2 in the BF16 original; 4 halves the per-threadgroup
                         // fixed costs (LUT staging, reduce) and shares each
                         // activation load across 4 rows. Per-row accumulation
                         // order is NR0-independent, so bit-identity vs the
                         // BF16 kernels is preserved (verified by test-m2-e4m3).
                // NOTE: nr0 = 8 for the short-row shapes (attn_q_b class,
                // ne00 = 1024) was tried to amortize the per-threadgroup
                // fixed costs behind that shape's 1.06x cell — it PASSED
                // bit-identity (NR0-independent row order) but measured
                // 1.31x: the longer 8-row serial loop per thread loses more
                // than the halved fixed costs buy. The kernel keeps its
                // case-8 dispatch for future experiments; the host stays at 4.
                {
                    const char * v = getenv("GGML_MV_NR0_E4M3");
                    if (v) {
                        const int r = atoi(v);
                        if (r == 2 || r == 4 || r == 8) {
                            nr0 = r;
                        }
                    }
                }
                nr1 = 1;
                smem = 32*sizeof(float)*nr0 + 256*sizeof(float);
                suffix = "_4";
            } break;
        case GGML_TYPE_NF8_M2:
            {
                // fork: same dispatch geometry as the E4M3_M2 GEMV above
                // (both are bit-exact clones of kernel_mul_mv_bf16_f32_4;
                // NR0=4 for the same measured reasons). Unlike E4M3_M2 the
                // NF8_M2 kernel stages NO decode LUT — the decode is pure
                // bit-insertion — so smem carries only the reduction area.
                GGML_ASSERT(ne00 % 4 == 0); // ne00 % QK_NF8_M2 == 0 by block invariant
                nsg = ggml_metal_mv_nsg(ne00, 512, 4); // NF = 16 -> 512 elems/simdgroup
                nr0 = 4;
                nr1 = 1;
                smem = 32*sizeof(float)*nr0;
                suffix = "_4";
            } break;
        case GGML_TYPE_MXFP4_M2:
            {
                // fork: mirrors the MXFP4 geometry (bit-exact split-plane clone)
                nsg = N_SG_MXFP4_M2;
                nr0 = N_R0_MXFP4_M2;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } break;
        case GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*sizeof(float)*N_R0_Q8_0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } break;
        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    GGML_ASSERT(ne12 <= INT16_MAX && ne13 <= INT16_MAX);
    const int16_t r2 = (int16_t) (ne12 / ne02);
    const int16_t r3 = (int16_t) (ne13 / ne03);

    snprintf(base, 256, "kernel_mul_mv_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d", base, nsg, ne12, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, r2,             FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, r3,             FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(
        ggml_metal_library_t lib,
        int ne02,
        int ne20,
        bool compact) {
    char base[256];
    char name[256];

    if (compact) {
        GGML_ASSERT(ne20 == 6);
        snprintf(base, 256, "kernel_mul_mm_id_map0_ne20_6_dsv4_n16");
    } else {
        snprintf(base, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    }
    snprintf(name, 256, "%s_ne02=%d", base, ne02);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = (size_t) ne02*ne20*sizeof(uint16_t);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool compact,
        bool paired,
        bool weighted) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const bool bc_inp = op->src[0]->ne[0] % 32 != 0;

    static const bool dsv4_tile_enabled =
        getenv("GGML_METAL_DSV4_MM_TILE_DISABLE") == nullptr;
    const bool use_dsv4_tile =
        dsv4_tile_enabled &&
        ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->device_id == GGML_METAL_DEVICE_M2_ULTRA &&
        (tsrc0 == GGML_TYPE_MXFP4 || tsrc0 == GGML_TYPE_MXFP4_M2) && // fork: M2 twin kernels exist for every dsv4 variant
        tsrc1 == GGML_TYPE_F32 &&
        op->src[0]->ne[2] == 256 && op->src[2]->ne[0] == 6 &&
        ((op->src[0]->ne[0] == 4096 && op->src[0]->ne[1] == 2048) ||
         (op->src[0]->ne[0] == 2048 && op->src[0]->ne[1] == 4096));

    if (use_dsv4_tile) {
        const char * tname = ggml_type_name(tsrc0); // "mxfp4" or "mxfp4_m2"
        if (weighted) {
            GGML_ASSERT(compact && !paired &&
                op->src[0]->ne[0] == 2048 && op->src[0]->ne[1] == 4096);
            snprintf(base, 256, "kernel_mul_mm_id_%s_f32_dsv4_n16_compact_weighted", tname);
        } else if (paired) {
            GGML_ASSERT(compact && op->src[0]->ne[0] == 4096 && op->src[0]->ne[1] == 2048);
            snprintf(base, 256, "kernel_mul_mm_id_%s_f32_dsv4_n16_compact_pair", tname);
        } else {
            snprintf(base, 256, compact ?
                    "kernel_mul_mm_id_%s_f32_dsv4_n16_compact" :
                    "kernel_mul_mm_id_%s_f32_dsv4_n16", tname);
        }
    } else {
        GGML_ASSERT(!compact && !paired && !weighted);
        snprintf(base, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    }
    snprintf(name, 256, "%s_bci=%d", base, bc_inp);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr1 = use_dsv4_tile ? 16 : 32;
    res.smem = 8192;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(
        ggml_metal_library_t lib, const ggml_tensor * op, bool weighted) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const ggml_metal_device_props * props_dev =
        ggml_metal_device_get_props(ggml_metal_library_get_device(lib));

    const char * suffix = "";

        // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                nsg = std::min(4, (ne00 + 127) / 128);
                nr0 = 2;
                nr1 = 1;
                smem = 32*sizeof(float)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
            } break;
        case GGML_TYPE_MXFP4_M2:
            {
                // fork: mirrors the MXFP4 geometry (bit-exact split-plane clone)
                nsg = N_SG_MXFP4_M2;
                nr0 = N_R0_MXFP4_M2;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } break;
        case GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*sizeof(float)*N_R0_Q8_0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } break;
        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int)op->src[2]->type);
                GGML_ABORT("not implemented");
            }
    };

    static const bool dsv4_expert_geometry_enabled =
        getenv("GGML_METAL_DSV4_EXPERT_GEOMETRY_DISABLE") == nullptr;

    const bool use_dsv4_expert_geometry =
        dsv4_expert_geometry_enabled &&
        props_dev->device_id == GGML_METAL_DEVICE_M2_ULTRA &&
        (tsrc0 == GGML_TYPE_MXFP4 || tsrc0 == GGML_TYPE_MXFP4_M2) && // fork: M2 twin kernels exist for every dsv4 variant
        tsrc1 == GGML_TYPE_F32 &&
        ne02 == 256 && op->src[2]->ne[0] == 6 &&
        ((ne00 == 4096 && ne01 == 2048) || (ne00 == 2048 && ne01 == 4096));

    if (use_dsv4_expert_geometry) {
        // One SIMDgroup computes four output rows while sharing each activation
        // load. The generic kernel computes two rows in each of two SIMDgroups.
        nsg = 1;
        nr0 = 4;
        suffix = weighted ? "_dsv4_weighted" : "_dsv4";
    }

    GGML_ASSERT(!weighted || (use_dsv4_expert_geometry &&
                ne00 == 2048 && ne01 == 4096 && op->src[2]->ne[1] == 1));

    snprintf(base, 256, "kernel_mul_mv_id_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_argmax_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*(sizeof(float) + sizeof(int32_t));

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGSORT);

    char base[256];
    char name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGSORT);

    char base[256];
    char name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht(ggml_metal_library_t lib, int n) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_fwht_f32_%d", n);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

// note: reuse the argsort kernel for top_k
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TOP_K);

    char base[256];
    char name[256];

    // note: the top_k kernel is always descending order
    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_radix(ggml_metal_library_t lib, bool radix8, int tie_policy) {
    // six variants: {radix-4, radix-8} x {oldest-first, newest-first,
    // hash-sampled pivot tie-break}. See kernel_top_k_radix_f32_i32_impl and
    // LLAMA_DSV4_TOPK_TIE.
    static const char * const names[2][3] = {
        { "kernel_top_k_radix_f32_i32",  "kernel_top_k_radix_f32_i32_tdesc",  "kernel_top_k_radix_f32_i32_thash"  },
        { "kernel_top_k_radix8_f32_i32", "kernel_top_k_radix8_f32_i32_tdesc", "kernel_top_k_radix8_f32_i32_thash" },
    };
    GGML_ASSERT(tie_policy >= 0 && tie_policy < 3);
    const char * name = names[radix8 ? 1 : 0][tie_policy];
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }
    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_merge(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TOP_K);

    char base[256];
    char name[256];

    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_pad");

    snprintf(name, 256, "%s_mask=%d_ncpsg=%d",
            base,
            has_mask,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_PAD + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_PAD + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_PAD + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_PAD + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_PAD + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_PAD + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_PAD + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_PAD + 23);
        //ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_PAD + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_blk");

    snprintf(name, 256, "%s_nqptg=%d_ncpsg=%d",
            base,
            nqptg,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        //ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_BLK + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_BLK + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_BLK + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_BLK + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_BLK + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_BLK + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_BLK + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_BLK + 23);
        ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_BLK + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_BLK + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];

    // do bounds checks for the mask?
    const bool bc_mask = op->src[3] && (op->src[3]->ne[1] % 8 != 0);
    const bool scan_mask = has_mask && op->src[3]->ne[1] != 1;

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext",
            ggml_type_name(op->src[1]->type),
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_scanm=%d_ns10=%d_ns20=%d_nsg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            bc_mask,
            scan_mask,
            ns10,
            ns20,
            nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);

        ggml_metal_cv_set_bool(cv, bc_mask, FC_FLASH_ATTN_EXT + 10);
        ggml_metal_cv_set_bool(cv, scan_mask, FC_FLASH_ATTN_EXT + 11);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT + 22);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        int32_t nwg,
        bool    blocked) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];

    // DSV4 decode has a single 512-wide KV head, 64 query heads and a
    // one-row top-k mask. Specialize per-row mask skipping to this signature
    // so the extra checks cannot affect ordinary causal-attention kernels.
    static const bool sparse_mask_enabled =
        std::getenv("GGML_METAL_DSV4_FA_MASK_SKIP_DISABLE") == nullptr;
    const bool sparse_mask = sparse_mask_enabled &&
        has_mask && has_sinks && dk == 512 && dv == 512 &&
        op->src[0]->ne[1] == 1 && op->src[0]->ne[2] == 64 &&
        op->src[1]->ne[2] == 1 && op->src[3]->ne[1] == 1 && op->src[4]->ne[0] == 64;

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext_vec",
            ggml_type_name(op->src[1]->type),
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sink=%d_bias=%d_scap=%d_kvpad=%d_mskip=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d_blk=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            sparse_mask,
            ns10,
            ns20,
            nsg, nwg, blocked);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_VEC + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT_VEC + 4);
        ggml_metal_cv_set_bool(cv, sparse_mask, FC_FLASH_ATTN_EXT_VEC + 5);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
        ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

        ggml_metal_cv_set_bool(cv, blocked, FC_FLASH_ATTN_EXT_VEC + 24);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        int32_t dv,
        int32_t nwg,
        bool    blocked,
        int32_t assoc) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_flash_attn_ext_vec_reduce");
    snprintf(name, 256, "%s_dv=%d_nwg=%d_blk=%d_as=%d", base, dv, nwg, blocked, assoc);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int32(cv, dv,  FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
        ggml_metal_cv_set_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

        ggml_metal_cv_set_bool (cv, blocked, FC_FLASH_ATTN_EXT_VEC_REDUCE + 2);
        ggml_metal_cv_set_int32(cv, assoc,   FC_FLASH_ATTN_EXT_VEC_REDUCE + 3);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;

    GGML_UNUSED(op);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(ggml_metal_library_t lib, const ggml_tensor * op, int32_t n_fuse) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t1_str = ggml_type_name(op->src[1]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = (op->src[0]->ne[0] % 4 == 0) && (op->src[1]->ne[0] % 4 == 0);

    const bool is_cb = op->src[0]->ne[0] != op->src[1]->ne[0];
    const bool is_rb = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && (ggml_nrows(op->src[1]) == 1) && ggml_nelements(op) < 65536;

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s%s", t0_str, t1_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_nf=%d_rb=%d_cb=%d", base, op_num, n_fuse, is_rb, is_cb);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, n_fuse, FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, is_rb,  FC_BIN + 2);
        ggml_metal_cv_set_bool (cv, is_cb,  FC_BIN + 3);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_rb;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one(ggml_metal_library_t lib, ggml_op op) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s", "f32", "f32", "f32");
    snprintf(name, 256, "%s_op=%d_nf=%d", base, op_num, 1);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, 1,      FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, false,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_L2_NORM);

    char base[256];
    char name[256];

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    snprintf(base, 256, "kernel_l2_norm_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.c4   = is_c4;
    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_GROUP_NORM);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_group_norm_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm(ggml_metal_library_t lib, const ggml_tensor * op, int n_fuse) {
    assert(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    switch (op->op) {
        case GGML_OP_NORM:
            switch (n_fuse) {
                case 1: snprintf(base, 256, "kernel_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        case GGML_OP_RMS_NORM:
            switch (n_fuse) {
                case 1: snprintf(base, 256, "kernel_rms_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_rms_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_rms_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ROPE || op->op == GGML_OP_ROPE_BACK);

    const bool is_back = op->op == GGML_OP_ROPE_BACK;

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[2];

    const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_neox) {
        snprintf(base, 256, "kernel_rope_neox_%s", ggml_type_name(op->src[0]->type));
    } else if ((is_mrope || is_imrope) && !is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_multi_%s", ggml_type_name(op->src[0]->type));
    } else if (is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_vision_%s", ggml_type_name(op->src[0]->type));
    } else {
        snprintf(base, 256, "kernel_rope_norm_%s", ggml_type_name(op->src[0]->type));
    }

    snprintf(name, 256, "%s_imrope=%d_is_back=%d", base, is_imrope ? 1 : 0, is_back ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, is_imrope, FC_ROPE + 0);
        ggml_metal_cv_set_bool(cv, is_back,   FC_ROPE + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_IM2COL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;
    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    char base[256];
    char name[256];

    if (KH*KW <= 1024) {
        snprintf(base, 256, "kernel_im2col_%s", ggml_type_name(op->type));
    } else {
        snprintf(base, 256, "kernel_im2col_ext_%s", ggml_type_name(op->type));
    }
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_1d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_COL2IM_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_col2im_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake(ggml_metal_library_t lib, enum ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_snake_%s", ggml_type_name(type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw(ggml_metal_library_t lib, const ggml_tensor * op, bool tiled) {
    assert(op->op == GGML_OP_CONV_2D_DW);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_2d_dw%s_%s_%s",
             tiled ? "_tiled" : "",
             ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_3D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_3d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_UPSCALE);

    char base[256];
    char name[256];

    const int32_t mode_flags = ggml_get_op_params_i32(op, 0);
    const ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    const bool antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);

    if (mode == GGML_SCALE_MODE_BILINEAR) {
        snprintf(base, 256, "kernel_upscale_bilinear_%s", ggml_type_name(op->src[0]->type));
    } else if (mode == GGML_SCALE_MODE_BICUBIC) {
        snprintf(base, 256, "kernel_upscale_bicubic_%s", ggml_type_name(op->src[0]->type));
    } else {
        snprintf(base, 256, "kernel_upscale_nearest_%s", ggml_type_name(op->src[0]->type));
    }
    snprintf(name, 256, "%s_aa=%d", base, antialias);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, antialias, FC_UPSCALE + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ROLL);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_roll_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD);

    char base[256];
    char name[256];

    // note: this is slower
    //const bool is_c4 = op->src[0]->ne[0] % 4 == 0 && op->ne[0] % 4 == 0;
    const bool is_c4 = false;

    snprintf(base, 256, "kernel_pad_%s%s", ggml_type_name(op->src[0]->type), is_c4 ? "_4" : "");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (res.pipeline) {
        return res;
    }

    res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);

    res.c4 = is_c4;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD_REFLECT_1D);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pad_reflect_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARANGE);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_arange_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TIMESTEP_EMBEDDING);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_timestep_embedding_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_OPT_STEP_ADAMW);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_adamw_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_OPT_STEP_SGD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_sgd_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SILU_BACK);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_silu_back_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset(ggml_metal_library_t lib, const ggml_tensor *  op) {
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_memset_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_count_equal(ggml_metal_library_t lib, const ggml_tensor *  op) {
    assert(op->op == GGML_OP_COUNT_EQUAL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    GGML_ASSERT(ggml_nelements(op->src[0]) < (1LL << 31));

    char base[256];
    char name[256];

    int nsg = 1;
    while (32*nsg < ne00 && nsg < 32) {
        nsg *= 2;
    }

    snprintf(base, 256, "kernel_count_equal_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_COUNT_EQUAL + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32 * sizeof(int32_t);
    res.nsg  = nsg;

    return res;
}
