#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

struct test_buffer_context {
    std::vector<uint8_t> data;
};

struct test_source_backend {
    ggml_backend_buffer_type buffer_type = {};
    ggml_backend_device      device      = {};
    ggml_backend             backend     = {};

    test_source_backend();

    ggml_backend_t get() { return &backend; }

    ggml_backend_buffer_type_t buft() { return &buffer_type; }
};

static const char * test_buffer_type_name(ggml_backend_buffer_type_t) {
    return "scheduler-test-source";
}

static void test_buffer_free(ggml_backend_buffer_t buffer) {
    delete static_cast<test_buffer_context *>(buffer->context);
}

static void * test_buffer_base(ggml_backend_buffer_t buffer) {
    return static_cast<test_buffer_context *>(buffer->context)->data.data();
}

static enum ggml_status test_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor *) {
    return GGML_STATUS_SUCCESS;
}

static void test_buffer_memset(ggml_backend_buffer_t buffer,
                               ggml_tensor *         tensor,
                               uint8_t               value,
                               size_t                offset,
                               size_t                size) {
    GGML_UNUSED(buffer);
    memset(static_cast<uint8_t *>(tensor->data) + offset, value, size);
}

static void test_buffer_set(ggml_backend_buffer_t buffer,
                            ggml_tensor *         tensor,
                            const void *          data,
                            size_t                offset,
                            size_t                size) {
    GGML_UNUSED(buffer);
    memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
}

static void test_buffer_get(ggml_backend_buffer_t buffer,
                            const ggml_tensor *   tensor,
                            void *                data,
                            size_t                offset,
                            size_t                size) {
    GGML_UNUSED(buffer);
    memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
}

static void test_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = static_cast<test_buffer_context *>(buffer->context);
    memset(context->data.data(), value, context->data.size());
}

static ggml_backend_buffer_t test_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * context = new test_buffer_context{ std::vector<uint8_t>(size) };

    ggml_backend_buffer_i interface = {};
    interface.free_buffer           = test_buffer_free;
    interface.get_base              = test_buffer_base;
    interface.init_tensor           = test_buffer_init_tensor;
    interface.memset_tensor         = test_buffer_memset;
    interface.set_tensor            = test_buffer_set;
    interface.get_tensor            = test_buffer_get;
    interface.clear                 = test_buffer_clear;

    return ggml_backend_buffer_init(buft, interface, context, size);
}

static size_t test_buffer_alignment(ggml_backend_buffer_type_t) {
    return 16;
}

static bool test_buffer_is_host(ggml_backend_buffer_type_t) {
    // Deliberately false: the CPU scheduler backend must import these tensors,
    // reproducing a device-only input without requiring target hardware.
    return false;
}

static const char * test_backend_name(ggml_backend_t) {
    return "scheduler-test-source";
}

static enum ggml_status test_backend_graph_compute(ggml_backend_t, ggml_cgraph *) {
    GGML_ABORT("the source-only test backend must not receive graph nodes");
}

static const char * test_device_name(ggml_backend_dev_t) {
    return "scheduler-test-source";
}

static const char * test_device_description(ggml_backend_dev_t) {
    return "source-only scheduler test backend";
}

static void test_device_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type test_device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void test_device_props(ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    *props             = {};
    props->name        = test_device_name(device);
    props->description = test_device_description(device);
    props->type        = test_device_type(device);
}

static ggml_backend_t test_device_init(ggml_backend_dev_t device, const char *) {
    return &static_cast<test_source_backend *>(device->context)->backend;
}

static ggml_backend_buffer_type_t test_device_buffer_type(ggml_backend_dev_t device) {
    return &static_cast<test_source_backend *>(device->context)->buffer_type;
}

static bool test_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return false;
}

static bool test_device_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return buft == test_device_buffer_type(device);
}

test_source_backend::test_source_backend() {
    buffer_type.iface.get_name      = test_buffer_type_name;
    buffer_type.iface.alloc_buffer  = test_buffer_alloc;
    buffer_type.iface.get_alignment = test_buffer_alignment;
    buffer_type.iface.is_host       = test_buffer_is_host;
    buffer_type.device              = &device;
    buffer_type.context             = this;

    device.iface.get_name        = test_device_name;
    device.iface.get_description = test_device_description;
    device.iface.get_memory      = test_device_memory;
    device.iface.get_type        = test_device_type;
    device.iface.get_props       = test_device_props;
    device.iface.init_backend    = test_device_init;
    device.iface.get_buffer_type = test_device_buffer_type;
    device.iface.supports_op     = test_device_supports_op;
    device.iface.supports_buft   = test_device_supports_buft;
    device.context               = this;

    backend.iface.get_name      = test_backend_name;
    backend.iface.graph_compute = test_backend_graph_compute;
    backend.device              = &device;
    backend.context             = this;
}

static void sum_sources(ggml_tensor * dst, int ith, int nth, void *) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < GGML_MAX_SRC && dst->src[i] != nullptr; ++i) {
        sum += *static_cast<const float *>(dst->src[i]->data);
    }
    *static_cast<float *>(dst->data) = sum;
}

struct boundary_group {
    std::vector<int> source_ids;
};

struct boundary_case {
    const char *                name;
    int                         prefix_inputs;
    std::vector<boundary_group> groups;
    int                         expected_splits;
};

static bool close_enough(float actual, float expected) {
    return std::fabs(actual - expected) <= 1e-6f;
}

static bool run_case(const boundary_case & test) {
    constexpr int graph_size = 256;

    ggml_init_params params = {};
    params.mem_size         = 512 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false);
    params.no_alloc         = true;
    ggml_context_ptr context{ ggml_init(params) };
    GGML_ASSERT(context != nullptr);

    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), graph_size, false);

    int max_source_id = test.prefix_inputs - 1;
    for (const auto & group : test.groups) {
        for (int source_id : group.source_ids) {
            max_source_id = std::max(max_source_id, source_id);
        }
    }

    ggml_tensor * seed = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 1);
    ggml_set_input(seed);
    ggml_set_name(seed, "cpu_seed");

    std::vector<ggml_tensor *> sources;
    for (int i = 0; i <= max_source_id; ++i) {
        ggml_tensor * source = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 1);
        ggml_set_input(source);
        ggml_format_name(source, "source_%d", i);
        sources.push_back(source);
    }

    ggml_tensor * prefix = seed;
    for (int i = 0; i < test.prefix_inputs; ++i) {
        prefix = ggml_add(context.get(), prefix, sources[i]);
        ggml_format_name(prefix, "prefix_%d", i + 1);
    }
    ggml_set_output(prefix);
    ggml_build_forward_expand(graph, prefix);

    std::vector<std::pair<ggml_tensor *, float>> outputs;
    for (size_t group_id = 0; group_id < test.groups.size(); ++group_id) {
        const auto &               group = test.groups[group_id];
        std::vector<ggml_tensor *> args;
        float                      expected = 0.0f;
        for (int source_id : group.source_ids) {
            args.push_back(sources[source_id]);
            expected += static_cast<float>(source_id + 1);
        }

        ggml_tensor * output =
            ggml_custom_4d(context.get(), GGML_TYPE_F32, 1, 1, 1, 1, args.data(), args.size(), sum_sources, 1, nullptr);
        ggml_format_name(output, "boundary_%zu", group_id);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        outputs.emplace_back(output, expected);
    }

    test_source_backend source_backend;
    ggml_backend_ptr    cpu_backend{ ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr) };
    GGML_ASSERT(cpu_backend != nullptr);

    ggml_backend_t             backends[] = { source_backend.get(), cpu_backend.get() };
    ggml_backend_buffer_type_t bufts[]    = {
        source_backend.buft(),
        ggml_backend_get_default_buffer_type(cpu_backend.get()),
    };
    ggml_backend_sched_ptr scheduler{ ggml_backend_sched_new(backends, bufts, 2, graph_size, false, true) };
    GGML_ASSERT(scheduler != nullptr);

    ggml_backend_sched_set_tensor_backend(scheduler.get(), seed, cpu_backend.get());
    for (ggml_tensor * source : sources) {
        ggml_backend_sched_set_tensor_backend(scheduler.get(), source, source_backend.get());
    }

    if (!ggml_backend_sched_alloc_graph(scheduler.get(), graph)) {
        fprintf(stderr, "%s: scheduler allocation failed\n", test.name);
        return false;
    }

    const int split_count_before_compute = ggml_backend_sched_get_n_splits(scheduler.get());
    if (split_count_before_compute != test.expected_splits) {
        fprintf(stderr, "%s: expected %d splits after allocation, got %d\n", test.name, test.expected_splits,
                split_count_before_compute);
        return false;
    }

    const float zero = 0.0f;
    ggml_backend_tensor_set(seed, &zero, 0, sizeof(zero));
    for (size_t i = 0; i < sources.size(); ++i) {
        const float value = static_cast<float>(i + 1);
        ggml_backend_tensor_set(sources[i], &value, 0, sizeof(value));
    }

    const ggml_status status = ggml_backend_sched_graph_compute(scheduler.get(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed with status %d\n", test.name, status);
        return false;
    }

    const int split_count_after_compute = ggml_backend_sched_get_n_splits(scheduler.get());
    if (split_count_after_compute != test.expected_splits) {
        fprintf(stderr, "%s: expected %d splits after compute, got %d\n", test.name, test.expected_splits,
                split_count_after_compute);
        return false;
    }

    float prefix_actual = 0.0f;
    ggml_backend_tensor_get(prefix, &prefix_actual, 0, sizeof(prefix_actual));
    const float prefix_expected = test.prefix_inputs * (test.prefix_inputs + 1) / 2.0f;
    if (!close_enough(prefix_actual, prefix_expected)) {
        fprintf(stderr, "%s: prefix result mismatch: expected %.1f, got %.1f\n", test.name, prefix_expected,
                prefix_actual);
        return false;
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        float actual = 0.0f;
        ggml_backend_tensor_get(outputs[i].first, &actual, 0, sizeof(actual));
        if (!close_enough(actual, outputs[i].second)) {
            fprintf(stderr, "%s: boundary %zu mismatch: expected %.1f, got %.1f\n", test.name, i, outputs[i].second,
                    actual);
            return false;
        }
    }

    printf("PASS %-24s splits=%d prefix=%.1f boundary_groups=%zu\n", test.name, split_count_after_compute,
           prefix_actual, outputs.size());
    return true;
}

}  // namespace

int main() {
    // Public custom ops expose at most nine sources. The 20+10 and 21+10
    // capacity cases therefore use two five-input nodes while preserving the
    // exact aggregate boundary. The 29+2 case is the shape that triggered the
    // original overflow: one node introduces both inputs when one slot remains.
    const std::vector<boundary_case> cases = {
        { "prior-crash-29-plus-2", 29, { { { 29, 30 } } },                                         2 },
        { "capacity-20-plus-10",   20, { { { 20, 21, 22, 23, 24 } }, { { 25, 26, 27, 28, 29 } } }, 1 },
        { "capacity-21-plus-10",   21, { { { 21, 22, 23, 24, 25 } }, { { 26, 27, 28, 29, 30 } } }, 2 },
        { "full-30-plus-1",        30, { { { 30 } } },                                             2 },
        { "duplicate-29-plus-1",   29, { { { 29, 29 } } },                                         1 },
    };

    bool success = true;
    for (const auto & test : cases) {
        success &= run_case(test);
    }
    return success ? 0 : 1;
}
