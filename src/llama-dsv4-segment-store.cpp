#include "llama-dsv4-segment-store.h"

#include "llama-dsv4-comp-pool.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace {

constexpr mode_t   PRIVATE_DIRECTORY_MODE = 0700;
constexpr mode_t   PRIVATE_FILE_MODE      = 0600;
constexpr uint64_t CHUNK_HEADER_BYTES     = 8 + 4 + 4 + 8 + 32 + 32 + 8 + 8;
constexpr uint64_t REF_BYTES              = 8 + 4 + 4 + 32 + 8 + 32 + 32;
constexpr uint64_t MAX_DESCRIPTOR_COUNT   = 1ULL << 20;
constexpr uint32_t C4_RATIO               = 4;
constexpr uint32_t HCA_RATIO              = 128;

const std::array<uint8_t, 8> MANIFEST_MAGIC = { 'D', '4', 'S', 'G', 'M', 'A', 'N', 0 };
const std::array<uint8_t, 8> CHUNK_MAGIC    = { 'D', '4', 'S', 'G', 'C', 'H', 'K', 0 };
const std::array<uint8_t, 8> REF_MAGIC      = { 'D', '4', 'S', 'G', 'R', 'E', 'F', 0 };
const std::array<uint8_t, 8> KEY_MAGIC      = { 'D', '4', 'S', 'G', 'K', 'E', 'Y', 0 };

std::atomic<uint64_t> temporary_sequence{ 1 };

bool digest_zero(const llama_snapshot_digest & digest) {
    return std::all_of(digest.begin(), digest.end(), [](uint8_t value) { return value == 0; });
}

bool checked_add(uint64_t & total, uint64_t value) {
    if (value > UINT64_MAX - total) {
        return false;
    }
    total += value;
    return true;
}

bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs != 0 && rhs > UINT64_MAX / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

void append_u32(std::vector<uint8_t> & bytes, uint32_t value) {
    for (uint32_t index = 0; index < 4; ++index) {
        bytes.push_back(static_cast<uint8_t>(value >> (8 * index)));
    }
}

void append_u64(std::vector<uint8_t> & bytes, uint64_t value) {
    for (uint32_t index = 0; index < 8; ++index) {
        bytes.push_back(static_cast<uint8_t>(value >> (8 * index)));
    }
}

void append_i64(std::vector<uint8_t> & bytes, int64_t value) {
    append_u64(bytes, static_cast<uint64_t>(value));
}

void append_digest(std::vector<uint8_t> & bytes, const llama_snapshot_digest & digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
}

struct cursor {
    const std::vector<uint8_t> & bytes;
    size_t                       offset = 0;

    bool take(void * destination, size_t size) {
        if (offset > bytes.size() || bytes.size() - offset < size) {
            return false;
        }
        if (size != 0) {
            std::memcpy(destination, bytes.data() + offset, size);
        }
        offset += size;
        return true;
    }

    bool u32(uint32_t & value) {
        uint8_t data[4];
        if (!take(data, sizeof(data))) {
            return false;
        }
        value = 0;
        for (uint32_t index = 0; index < 4; ++index) {
            value |= static_cast<uint32_t>(data[index]) << (8 * index);
        }
        return true;
    }

    bool u64(uint64_t & value) {
        uint8_t data[8];
        if (!take(data, sizeof(data))) {
            return false;
        }
        value = 0;
        for (uint32_t index = 0; index < 8; ++index) {
            value |= static_cast<uint64_t>(data[index]) << (8 * index);
        }
        return true;
    }

    bool i64(int64_t & value) {
        uint64_t encoded = 0;
        if (!u64(encoded)) {
            return false;
        }
        value = static_cast<int64_t>(encoded);
        return true;
    }

    bool digest(llama_snapshot_digest & value) { return take(value.data(), value.size()); }
};

uint64_t fnv_u64(uint64_t hash, uint64_t value) {
    for (uint32_t index = 0; index < 8; ++index) {
        hash ^= (value >> (8 * index)) & 0xffU;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t fnv_bytes(uint64_t hash, const uint8_t * data, size_t size) {
    hash = fnv_u64(hash, size);
    for (size_t index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t logical_row_checksum(uint64_t row_begin, uint64_t row_count, const std::vector<uint8_t> & bytes) {
    uint64_t              hash     = UINT64_C(14695981039346656037);
    static constexpr char domain[] = "llama.cpp-dsv4-logical-row-chunk-v1";
    hash                           = fnv_u64(hash, sizeof(domain) - 1);
    for (size_t index = 0; index < sizeof(domain) - 1; ++index) {
        hash ^= static_cast<uint8_t>(domain[index]);
        hash *= UINT64_C(1099511628211);
    }
    hash = fnv_u64(hash, row_begin);
    hash = fnv_u64(hash, row_count);
    return fnv_bytes(hash, bytes.data(), bytes.size());
}

uint64_t logical_plane_checksum(const llama_kv_iswa_logical_plane_state & plane) {
    uint64_t hash = UINT64_C(14695981039346656037);
    hash          = fnv_u64(hash, plane.schema_version);
    hash          = fnv_u64(hash, plane.positions.size());
    for (llama_pos position : plane.positions) {
        hash = fnv_u64(hash, static_cast<uint64_t>(position));
    }
    hash = fnv_u64(hash, plane.extensions.size());
    for (const auto & extension : plane.extensions) {
        hash = fnv_u64(hash, static_cast<uint64_t>(extension.x));
        hash = fnv_u64(hash, static_cast<uint64_t>(extension.y));
    }
    return fnv_bytes(hash, plane.tensor_payload.data(), plane.tensor_payload.size());
}

enum class field_encoding : uint32_t {
    bytes      = 0,
    positions  = 1,
    extensions = 2,
};

enum class field_kind : uint32_t {
    raw_base_positions = 1,
    raw_base_extensions,
    raw_base_tensor,
    raw_swa_positions,
    raw_swa_extensions,
    raw_swa_tensor,
    compressed_rows = 16,
    recurrent_rows  = 32,
};

struct field_path {
    field_kind kind           = field_kind::raw_base_positions;
    uint32_t   group          = 0;
    uint32_t   tensor_class   = 0;
    uint32_t   tensor_ordinal = 0;
    uint32_t   layer_id       = 0;
    int32_t    type           = -1;
    uint64_t   ne0            = 0;
    uint64_t   row_size       = 0;
    uint64_t   row_begin      = 0;
    uint64_t   row_count      = 0;
};

llama_snapshot_digest semantic_key(const llama_dsv4_segment_identity & identity, const field_path & path) {
    std::vector<uint8_t> bytes;
    bytes.insert(bytes.end(), KEY_MAGIC.begin(), KEY_MAGIC.end());
    append_u32(bytes, LLAMA_DSV4_SEGMENT_FORMAT_VERSION);
    append_u32(bytes, LLAMA_DSV4_LOGICAL_STATE_SCHEMA);
    append_u64(bytes, identity.geometry_identity);
    append_digest(bytes, identity.model_artifact_digest);
    append_u32(bytes, static_cast<uint32_t>(path.kind));
    append_u32(bytes, path.group);
    append_u32(bytes, path.tensor_class);
    append_u32(bytes, path.tensor_ordinal);
    append_u32(bytes, path.layer_id);
    append_u32(bytes, static_cast<uint32_t>(path.type));
    append_u64(bytes, path.ne0);
    append_u64(bytes, path.row_size);
    append_u64(bytes, path.row_begin);
    append_u64(bytes, path.row_count);
    return llama_snapshot_sha256(bytes.data(), bytes.size());
}

struct field_source {
    field_encoding                         encoding = field_encoding::bytes;
    field_path                             path;
    llama_snapshot_digest                  key          = {};
    uint64_t                               logical_size = 0;
    const std::vector<uint8_t> *           bytes        = nullptr;
    const std::vector<llama_pos> *         positions    = nullptr;
    const std::vector<llama_kv_cell_ext> * extensions   = nullptr;
};

struct row_meta {
    uint64_t row_begin  = 0;
    uint64_t row_count  = 0;
    uint64_t byte_count = 0;
    uint64_t checksum   = 0;
};

struct tensor_meta {
    uint32_t              layer_id = 0;
    int32_t               type     = -1;
    uint64_t              ne0      = 0;
    uint64_t              row_size = 0;
    std::vector<row_meta> chunks;
};

struct component_meta {
    uint32_t                 schema_version = 0;
    uint64_t                 row_begin      = 0;
    uint64_t                 row_end        = 0;
    std::vector<tensor_meta> tensors;
};

struct recurrent_meta {
    uint32_t                 schema_version = 0;
    uint32_t                 ratio          = 0;
    uint32_t                 state_size     = 0;
    uint32_t                 n_embd_state   = 0;
    uint32_t                 n_rs_seq       = 0;
    uint64_t                 state_identity = 0;
    std::vector<tensor_meta> kv;
    std::vector<tensor_meta> score;
};

struct plane_meta {
    uint32_t schema_version  = 0;
    uint64_t position_count  = 0;
    uint64_t extension_count = 0;
    uint64_t tensor_bytes    = 0;
    uint64_t checksum        = 0;
};

struct sequence_meta {
    uint32_t       schema_version        = 0;
    uint64_t       identity              = 0;
    int64_t        accepted_frontier     = -1;
    uint32_t       rollback_index        = 0;
    uint32_t       active_rollback_depth = 0;
    uint32_t       n_rs_seq              = 0;
    uint64_t       fingerprint           = 0;
    uint32_t       raw_schema_version    = 0;
    plane_meta     base;
    plane_meta     swa;
    component_meta csa;
    component_meta hca;
    component_meta lid;
    recurrent_meta csa_recurrent;
    recurrent_meta hca_recurrent;
    recurrent_meta lid_recurrent;
};

struct fragment_record {
    uint64_t              offset = 0;
    uint64_t              size   = 0;
    llama_snapshot_digest digest = {};
    bool                  reused = false;
};

struct field_record {
    field_encoding               encoding = field_encoding::bytes;
    field_path                   path;
    llama_snapshot_digest        key          = {};
    uint64_t                     logical_size = 0;
    std::vector<fragment_record> fragments;
};

struct parsed_manifest {
    llama_dsv4_segment_identity        identity;
    llama_dsv4_segment_prefix_metadata prefix;
    bool                               has_parent    = false;
    llama_snapshot_digest              parent        = {};
    uint64_t                           fingerprint   = 0;
    uint64_t                           total_payload = 0;
    std::vector<uint8_t>               envelope;
    sequence_meta                      meta;
    std::vector<field_record>          fields;
};

void append_row_meta(std::vector<uint8_t> & bytes, const llama_dsv4_logical_row_chunk & chunk) {
    append_u64(bytes, chunk.row_begin);
    append_u64(bytes, chunk.row_count);
    append_u64(bytes, chunk.bytes.size());
    append_u64(bytes, chunk.checksum);
}

void append_tensor_meta(std::vector<uint8_t> & bytes, const llama_dsv4_logical_tensor_state & tensor) {
    append_u32(bytes, tensor.layer_id);
    append_u32(bytes, static_cast<uint32_t>(tensor.type));
    append_u64(bytes, tensor.ne0);
    append_u64(bytes, tensor.row_size);
    append_u32(bytes, static_cast<uint32_t>(tensor.chunks.size()));
    append_u32(bytes, 0);
    for (const auto & chunk : tensor.chunks) {
        append_row_meta(bytes, chunk);
    }
}

void append_tensor_list(std::vector<uint8_t> & bytes, const std::vector<llama_dsv4_logical_tensor_state> & tensors) {
    append_u32(bytes, static_cast<uint32_t>(tensors.size()));
    append_u32(bytes, 0);
    for (const auto & tensor : tensors) {
        append_tensor_meta(bytes, tensor);
    }
}

void append_component_meta(std::vector<uint8_t> & bytes, const llama_dsv4_logical_component_state & component) {
    append_u32(bytes, component.schema_version);
    append_u32(bytes, 0);
    append_u64(bytes, component.row_begin);
    append_u64(bytes, component.row_end);
    append_tensor_list(bytes, component.tensors);
}

void append_recurrent_meta(std::vector<uint8_t> & bytes, const llama_dsv4_recurrent_sequence_state & recurrent) {
    append_u32(bytes, recurrent.schema_version);
    append_u32(bytes, recurrent.ratio);
    append_u32(bytes, recurrent.state_size);
    append_u32(bytes, recurrent.n_embd_state);
    append_u32(bytes, recurrent.n_rs_seq);
    append_u32(bytes, 0);
    append_u64(bytes, recurrent.state_identity);
    append_tensor_list(bytes, recurrent.kv);
    append_tensor_list(bytes, recurrent.score);
}

std::vector<uint8_t> make_envelope(const llama_dsv4_logical_sequence_state & state) {
    std::vector<uint8_t> bytes;
    append_u32(bytes, state.schema_version);
    append_u32(bytes, 0);
    append_u64(bytes, state.identity);
    append_i64(bytes, state.accepted_frontier);
    append_u32(bytes, state.rollback_index);
    append_u32(bytes, state.active_rollback_depth);
    append_u32(bytes, state.n_rs_seq);
    append_u32(bytes, 0);
    append_u64(bytes, state.fingerprint);
    append_u32(bytes, state.raw_swa.schema_version);
    append_u32(bytes, 0);
    const auto append_plane = [&](const llama_kv_iswa_logical_plane_state & plane) {
        append_u32(bytes, plane.schema_version);
        append_u32(bytes, 0);
        append_u64(bytes, plane.positions.size());
        append_u64(bytes, plane.extensions.size());
        append_u64(bytes, plane.tensor_payload.size());
        append_u64(bytes, plane.checksum);
    };
    append_plane(state.raw_swa.base);
    append_plane(state.raw_swa.swa);
    append_component_meta(bytes, state.csa);
    append_component_meta(bytes, state.hca);
    append_component_meta(bytes, state.lid);
    append_recurrent_meta(bytes, state.csa_recurrent);
    append_recurrent_meta(bytes, state.hca_recurrent);
    append_recurrent_meta(bytes, state.lid_recurrent);
    return bytes;
}

bool read_reserved(cursor & input) {
    uint32_t reserved = 0;
    return input.u32(reserved) && reserved == 0;
}

bool read_tensor_meta(cursor & input, tensor_meta & tensor, uint64_t & descriptor_count) {
    uint32_t type        = 0;
    uint32_t chunk_count = 0;
    if (!input.u32(tensor.layer_id) || !input.u32(type) || !input.u64(tensor.ne0) || !input.u64(tensor.row_size) ||
        !input.u32(chunk_count) || !read_reserved(input) || chunk_count > MAX_DESCRIPTOR_COUNT ||
        descriptor_count > MAX_DESCRIPTOR_COUNT - chunk_count) {
        return false;
    }
    tensor.type = static_cast<int32_t>(type);
    descriptor_count += chunk_count;
    tensor.chunks.resize(chunk_count);
    for (auto & chunk : tensor.chunks) {
        if (!input.u64(chunk.row_begin) || !input.u64(chunk.row_count) || !input.u64(chunk.byte_count) ||
            !input.u64(chunk.checksum)) {
            return false;
        }
    }
    return true;
}

bool read_tensor_list(cursor & input, std::vector<tensor_meta> & tensors, uint64_t & descriptor_count) {
    uint32_t count = 0;
    if (!input.u32(count) || !read_reserved(input) || count > MAX_DESCRIPTOR_COUNT ||
        descriptor_count > MAX_DESCRIPTOR_COUNT - count) {
        return false;
    }
    descriptor_count += count;
    tensors.resize(count);
    for (auto & tensor : tensors) {
        if (!read_tensor_meta(input, tensor, descriptor_count)) {
            return false;
        }
    }
    return true;
}

bool read_component_meta(cursor & input, component_meta & component, uint64_t & descriptor_count) {
    return input.u32(component.schema_version) && read_reserved(input) && input.u64(component.row_begin) &&
           input.u64(component.row_end) && read_tensor_list(input, component.tensors, descriptor_count);
}

bool read_recurrent_meta(cursor & input, recurrent_meta & recurrent, uint64_t & descriptor_count) {
    return input.u32(recurrent.schema_version) && input.u32(recurrent.ratio) && input.u32(recurrent.state_size) &&
           input.u32(recurrent.n_embd_state) && input.u32(recurrent.n_rs_seq) && read_reserved(input) &&
           input.u64(recurrent.state_identity) && read_tensor_list(input, recurrent.kv, descriptor_count) &&
           read_tensor_list(input, recurrent.score, descriptor_count);
}

llama_dsv4_segment_status parse_envelope(const std::vector<uint8_t> & bytes, sequence_meta & meta) {
    cursor   input{ bytes };
    uint64_t descriptor_count = 0;
    if (!input.u32(meta.schema_version) || !read_reserved(input) || !input.u64(meta.identity) ||
        !input.i64(meta.accepted_frontier) || !input.u32(meta.rollback_index) ||
        !input.u32(meta.active_rollback_depth) || !input.u32(meta.n_rs_seq) || !read_reserved(input) ||
        !input.u64(meta.fingerprint) || !input.u32(meta.raw_schema_version) || !read_reserved(input)) {
        return llama_dsv4_segment_status::truncated;
    }
    const auto read_plane = [&](plane_meta & plane) {
        return input.u32(plane.schema_version) && read_reserved(input) && input.u64(plane.position_count) &&
               input.u64(plane.extension_count) && input.u64(plane.tensor_bytes) && input.u64(plane.checksum);
    };
    if (!read_plane(meta.base) || !read_plane(meta.swa) || !read_component_meta(input, meta.csa, descriptor_count) ||
        !read_component_meta(input, meta.hca, descriptor_count) ||
        !read_component_meta(input, meta.lid, descriptor_count) ||
        !read_recurrent_meta(input, meta.csa_recurrent, descriptor_count) ||
        !read_recurrent_meta(input, meta.hca_recurrent, descriptor_count) ||
        !read_recurrent_meta(input, meta.lid_recurrent, descriptor_count)) {
        return llama_dsv4_segment_status::truncated;
    }
    return input.offset == bytes.size() ? llama_dsv4_segment_status::ok : llama_dsv4_segment_status::trailing_data;
}

bool add_field_size(uint64_t & total, uint64_t value, uint64_t limit) {
    return checked_add(total, value) && total <= limit;
}

llama_dsv4_segment_status validate_tensor_meta(const tensor_meta & tensor,
                                               uint64_t            expected_rows,
                                               bool                recurrent,
                                               uint32_t            n_rs_seq,
                                               uint32_t            state_size,
                                               uint64_t            max_state_bytes,
                                               uint64_t &          payload_bytes,
                                               uint64_t &          field_count) {
    if (tensor.type < 0 || tensor.ne0 == 0 || tensor.row_size == 0) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    const uint64_t expected_chunks =
        recurrent ? static_cast<uint64_t>(n_rs_seq) + 1 : llama_dsv4_comp_segments_for_rows(expected_rows);
    if (tensor.chunks.size() != expected_chunks) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    uint64_t previous_end = 0;
    for (uint64_t index = 0; index < tensor.chunks.size(); ++index) {
        const auto &   chunk     = tensor.chunks[index];
        const uint64_t row_begin = recurrent ? index * state_size : index * LLAMA_DSV4_COMP_SEGMENT_ROWS;
        const uint64_t row_count =
            recurrent ? state_size : std::min<uint64_t>(LLAMA_DSV4_COMP_SEGMENT_ROWS, expected_rows - row_begin);
        uint64_t byte_count = 0;
        if (!checked_multiply(row_count, tensor.row_size, byte_count) || chunk.row_begin != row_begin ||
            chunk.row_count != row_count || chunk.byte_count != byte_count || chunk.row_begin != previous_end ||
            !add_field_size(payload_bytes, byte_count, max_state_bytes)) {
            return llama_dsv4_segment_status::coverage_mismatch;
        }
        previous_end = row_begin + row_count;
        ++field_count;
    }
    return llama_dsv4_segment_status::ok;
}

llama_dsv4_segment_status validate_tensor_list_meta(const std::vector<tensor_meta> & tensors,
                                                    uint64_t                         expected_rows,
                                                    bool                             recurrent,
                                                    uint32_t                         n_rs_seq,
                                                    uint32_t                         state_size,
                                                    uint64_t                         max_state_bytes,
                                                    uint64_t &                       payload_bytes,
                                                    uint64_t &                       field_count) {
    uint32_t previous_layer = 0;
    bool     first          = true;
    for (const auto & tensor : tensors) {
        if (!first && tensor.layer_id <= previous_layer) {
            return llama_dsv4_segment_status::coverage_mismatch;
        }
        first             = false;
        previous_layer    = tensor.layer_id;
        const auto status = validate_tensor_meta(tensor, expected_rows, recurrent, n_rs_seq, state_size,
                                                 max_state_bytes, payload_bytes, field_count);
        if (status != llama_dsv4_segment_status::ok) {
            return status;
        }
    }
    return llama_dsv4_segment_status::ok;
}

llama_dsv4_segment_status validate_meta(const sequence_meta &               meta,
                                        const llama_dsv4_segment_identity & identity,
                                        uint64_t                            max_state_bytes,
                                        uint64_t &                          payload_bytes,
                                        uint64_t &                          field_count) {
    payload_bytes = 0;
    field_count   = 0;
    if (meta.schema_version != LLAMA_DSV4_LOGICAL_STATE_SCHEMA ||
        meta.raw_schema_version != LLAMA_KV_ISWA_LOGICAL_STATE_SCHEMA ||
        meta.base.schema_version != LLAMA_KV_ISWA_LOGICAL_STATE_SCHEMA ||
        meta.swa.schema_version != LLAMA_KV_ISWA_LOGICAL_STATE_SCHEMA) {
        return llama_dsv4_segment_status::invalid_schema;
    }
    if (meta.identity != identity.geometry_identity) {
        return llama_dsv4_segment_status::identity_mismatch;
    }
    if (meta.accepted_frontier < -1 || meta.active_rollback_depth > meta.n_rs_seq ||
        meta.rollback_index > meta.active_rollback_depth || meta.fingerprint == 0) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    const uint64_t accepted_tokens = meta.accepted_frontier < 0 ? 0 : static_cast<uint64_t>(meta.accepted_frontier) + 1;
    if (meta.base.position_count > accepted_tokens || meta.swa.position_count > accepted_tokens ||
        (meta.base.extension_count != 0 && meta.base.extension_count != meta.base.position_count) ||
        (meta.swa.extension_count != 0 && meta.swa.extension_count != meta.swa.position_count)) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    uint64_t base_positions  = 0;
    uint64_t base_extensions = 0;
    uint64_t swa_positions   = 0;
    uint64_t swa_extensions  = 0;
    if (!checked_multiply(meta.base.position_count, 8, base_positions) ||
        !checked_multiply(meta.base.extension_count, 16, base_extensions) ||
        !checked_multiply(meta.swa.position_count, 8, swa_positions) ||
        !checked_multiply(meta.swa.extension_count, 16, swa_extensions)) {
        return llama_dsv4_segment_status::state_too_large;
    }
    for (uint64_t size : { base_positions, base_extensions, meta.base.tensor_bytes, swa_positions, swa_extensions,
                           meta.swa.tensor_bytes }) {
        if (!add_field_size(payload_bytes, size, max_state_bytes)) {
            return llama_dsv4_segment_status::state_too_large;
        }
        ++field_count;
    }
    const uint64_t c4_rows            = accepted_tokens / C4_RATIO;
    const uint64_t hca_rows           = accepted_tokens / HCA_RATIO;
    const auto     validate_component = [&](const component_meta & component, uint64_t rows) {
        if (component.schema_version != LLAMA_DSV4_LOGICAL_STATE_SCHEMA || component.row_begin != 0 ||
            component.row_end != rows) {
            return llama_dsv4_segment_status::coverage_mismatch;
        }
        return validate_tensor_list_meta(component.tensors, rows, false, 0, 0, max_state_bytes, payload_bytes,
                                             field_count);
    };
    auto status = validate_component(meta.csa, c4_rows);
    if (status != llama_dsv4_segment_status::ok) {
        return status;
    }
    status = validate_component(meta.hca, hca_rows);
    if (status != llama_dsv4_segment_status::ok) {
        return status;
    }
    status = validate_component(meta.lid, c4_rows);
    if (status != llama_dsv4_segment_status::ok) {
        return status;
    }
    if (meta.csa.tensors.size() != meta.lid.tensors.size()) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    const auto validate_recurrent = [&](const recurrent_meta & recurrent, uint32_t ratio) {
        if (recurrent.schema_version != LLAMA_DSV4_LOGICAL_STATE_SCHEMA || recurrent.ratio != ratio ||
            recurrent.state_size == 0 || recurrent.n_embd_state == 0 || recurrent.n_rs_seq != meta.n_rs_seq ||
            recurrent.state_identity == 0 || recurrent.kv.size() != recurrent.score.size()) {
            return llama_dsv4_segment_status::coverage_mismatch;
        }
        auto recurrent_status =
            validate_tensor_list_meta(recurrent.kv, 0, true, recurrent.n_rs_seq, recurrent.state_size, max_state_bytes,
                                      payload_bytes, field_count);
        if (recurrent_status != llama_dsv4_segment_status::ok) {
            return recurrent_status;
        }
        recurrent_status = validate_tensor_list_meta(recurrent.score, 0, true, recurrent.n_rs_seq, recurrent.state_size,
                                                     max_state_bytes, payload_bytes, field_count);
        if (recurrent_status != llama_dsv4_segment_status::ok) {
            return recurrent_status;
        }
        for (size_t index = 0; index < recurrent.kv.size(); ++index) {
            if (recurrent.kv[index].layer_id != recurrent.score[index].layer_id) {
                return llama_dsv4_segment_status::coverage_mismatch;
            }
        }
        return llama_dsv4_segment_status::ok;
    };
    status = validate_recurrent(meta.csa_recurrent, C4_RATIO);
    if (status != llama_dsv4_segment_status::ok) {
        return status;
    }
    status = validate_recurrent(meta.hca_recurrent, HCA_RATIO);
    if (status != llama_dsv4_segment_status::ok) {
        return status;
    }
    status = validate_recurrent(meta.lid_recurrent, C4_RATIO);
    return status;
}

void add_plane_sources(std::vector<field_source> &               fields,
                       const llama_kv_iswa_logical_plane_state & plane,
                       bool                                      base,
                       const llama_dsv4_segment_identity &       identity) {
    const field_kind position_kind  = base ? field_kind::raw_base_positions : field_kind::raw_swa_positions;
    const field_kind extension_kind = base ? field_kind::raw_base_extensions : field_kind::raw_swa_extensions;
    const field_kind tensor_kind    = base ? field_kind::raw_base_tensor : field_kind::raw_swa_tensor;
    field_source     positions;
    positions.encoding     = field_encoding::positions;
    positions.path.kind    = position_kind;
    positions.key          = semantic_key(identity, positions.path);
    positions.logical_size = static_cast<uint64_t>(plane.positions.size()) * 8;
    positions.positions    = &plane.positions;
    fields.push_back(positions);
    field_source extensions;
    extensions.encoding     = field_encoding::extensions;
    extensions.path.kind    = extension_kind;
    extensions.key          = semantic_key(identity, extensions.path);
    extensions.logical_size = static_cast<uint64_t>(plane.extensions.size()) * 16;
    extensions.extensions   = &plane.extensions;
    fields.push_back(extensions);
    field_source tensor;
    tensor.encoding     = field_encoding::bytes;
    tensor.path.kind    = tensor_kind;
    tensor.key          = semantic_key(identity, tensor.path);
    tensor.logical_size = plane.tensor_payload.size();
    tensor.bytes        = &plane.tensor_payload;
    fields.push_back(tensor);
}

void add_tensor_sources(std::vector<field_source> &                          fields,
                        const std::vector<llama_dsv4_logical_tensor_state> & tensors,
                        bool                                                 recurrent,
                        uint32_t                                             group,
                        uint32_t                                             tensor_class,
                        const llama_dsv4_segment_identity &                  identity) {
    for (uint32_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        const auto & tensor = tensors[tensor_index];
        for (const auto & chunk : tensor.chunks) {
            field_source source;
            source.encoding            = field_encoding::bytes;
            source.path.kind           = recurrent ? field_kind::recurrent_rows : field_kind::compressed_rows;
            source.path.group          = group;
            source.path.tensor_class   = tensor_class;
            source.path.tensor_ordinal = tensor_index;
            source.path.layer_id       = tensor.layer_id;
            source.path.type           = tensor.type;
            source.path.ne0            = tensor.ne0;
            source.path.row_size       = tensor.row_size;
            source.path.row_begin      = chunk.row_begin;
            source.path.row_count      = chunk.row_count;
            source.key                 = semantic_key(identity, source.path);
            source.logical_size        = chunk.bytes.size();
            source.bytes               = &chunk.bytes;
            fields.push_back(source);
        }
    }
}

std::vector<field_source> make_sources(const llama_dsv4_logical_sequence_state & state,
                                       const llama_dsv4_segment_identity &       identity) {
    std::vector<field_source> fields;
    add_plane_sources(fields, state.raw_swa.base, true, identity);
    add_plane_sources(fields, state.raw_swa.swa, false, identity);
    add_tensor_sources(fields, state.csa.tensors, false, 0, 0, identity);
    add_tensor_sources(fields, state.hca.tensors, false, 1, 0, identity);
    add_tensor_sources(fields, state.lid.tensors, false, 2, 0, identity);
    add_tensor_sources(fields, state.csa_recurrent.kv, true, 0, 0, identity);
    add_tensor_sources(fields, state.csa_recurrent.score, true, 0, 1, identity);
    add_tensor_sources(fields, state.hca_recurrent.kv, true, 1, 0, identity);
    add_tensor_sources(fields, state.hca_recurrent.score, true, 1, 1, identity);
    add_tensor_sources(fields, state.lid_recurrent.kv, true, 2, 0, identity);
    add_tensor_sources(fields, state.lid_recurrent.score, true, 2, 1, identity);
    return fields;
}

llama_dsv4_segment_status validate_state(const llama_dsv4_logical_sequence_state & state,
                                         const llama_dsv4_segment_identity &       identity,
                                         uint64_t                                  max_state_bytes,
                                         sequence_meta &                           meta,
                                         std::vector<uint8_t> &                    envelope,
                                         std::vector<field_source> &               fields,
                                         uint64_t &                                payload_bytes) {
    if (identity.geometry_identity == 0 || digest_zero(identity.model_artifact_digest) ||
        identity.geometry_identity != state.identity) {
        return llama_dsv4_segment_status::identity_mismatch;
    }
    if (state.fingerprint == 0 || state.fingerprint != llama_dsv4_logical_sequence_fingerprint(state)) {
        return llama_dsv4_segment_status::checksum_mismatch;
    }
    envelope    = make_envelope(state);
    auto status = parse_envelope(envelope, meta);
    if (status != llama_dsv4_segment_status::ok) {
        return status;
    }
    uint64_t field_count = 0;
    status               = validate_meta(meta, identity, max_state_bytes, payload_bytes, field_count);
    if (status != llama_dsv4_segment_status::ok) {
        return status;
    }
    fields = make_sources(state, identity);
    if (fields.size() != field_count) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    for (const auto & tensor : { &state.csa.tensors, &state.hca.tensors, &state.lid.tensors, &state.csa_recurrent.kv,
                                 &state.csa_recurrent.score, &state.hca_recurrent.kv, &state.hca_recurrent.score,
                                 &state.lid_recurrent.kv, &state.lid_recurrent.score }) {
        for (const auto & logical : *tensor) {
            for (const auto & chunk : logical.chunks) {
                if (chunk.checksum != logical_row_checksum(chunk.row_begin, chunk.row_count, chunk.bytes)) {
                    return llama_dsv4_segment_status::checksum_mismatch;
                }
            }
        }
    }
    if (state.raw_swa.base.checksum != logical_plane_checksum(state.raw_swa.base) ||
        state.raw_swa.swa.checksum != logical_plane_checksum(state.raw_swa.swa)) {
        return llama_dsv4_segment_status::checksum_mismatch;
    }
    const auto validate_positions = [&](const llama_kv_iswa_logical_plane_state & plane) {
        if (state.accepted_frontier < 0) {
            return plane.positions.empty();
        }
        if (plane.positions.empty() || plane.positions.back() != state.accepted_frontier) {
            return false;
        }
        for (size_t index = 0; index < plane.positions.size(); ++index) {
            if (plane.positions[index] < 0 || plane.positions[index] > state.accepted_frontier ||
                (index != 0 && plane.positions[index - 1] >= plane.positions[index])) {
                return false;
            }
        }
        return true;
    };
    if (!validate_positions(state.raw_swa.base) || !validate_positions(state.raw_swa.swa)) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    return llama_dsv4_segment_status::ok;
}

uint64_t fragment_payload_limit(field_encoding encoding) {
    const uint64_t unit = encoding == field_encoding::positions ? 8 : encoding == field_encoding::extensions ? 16 : 1;
    return LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES - LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES % unit;
}

uint64_t fragment_count(const field_source & field) {
    const uint64_t limit = fragment_payload_limit(field.encoding);
    return field.logical_size == 0 ? 0 : 1 + (field.logical_size - 1) / limit;
}

std::string chunk_filename(const llama_snapshot_digest & digest) {
    return llama_snapshot_digest_hex(digest) + ".chunk";
}

std::string manifest_filename(const llama_snapshot_digest & digest) {
    return llama_snapshot_digest_hex(digest) + ".manifest";
}

bool safe_ref_name(const std::string & name) {
    if (name.empty() || name.size() > 80 || name == "." || name == "..") {
        return false;
    }
    return std::all_of(name.begin(), name.end(), [](unsigned char value) {
        return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') || (value >= '0' && value <= '9') ||
               value == '-' || value == '_' || value == '.';
    });
}

std::string ref_filename(const std::string & name) {
    return name + ".ref";
}

bool private_regular_file(const struct stat & status) {
    return S_ISREG(status.st_mode) && status.st_uid == ::geteuid() && (status.st_mode & 07777) == PRIVATE_FILE_MODE;
}

bool same_file(const struct stat & lhs, const struct stat & rhs) {
    return lhs.st_dev == rhs.st_dev && lhs.st_ino == rhs.st_ino;
}

int unlink_private_same(int directory, const std::string & name, const struct stat & expected) noexcept {
    struct stat current = {};
    if (::fstatat(directory, name.c_str(), &current, AT_SYMLINK_NOFOLLOW) != 0) {
        return -1;
    }
    if (!private_regular_file(current) || !same_file(current, expected)) {
        errno = EACCES;
        return -1;
    }
    return ::unlinkat(directory, name.c_str(), 0);
}

void record_first_errno(int operation_result, int & first_error) noexcept {
    if (operation_result != 0) {
        const int operation_error = errno;
        if (first_error == 0) {
            first_error = operation_error;
        }
    }
}

int close_and_cleanup_temporary(int                 directory,
                                const std::string & name,
                                const struct stat & expected,
                                int                 descriptor) noexcept {
    int first_error = 0;
    record_first_errno(::close(descriptor), first_error);
    record_first_errno(unlink_private_same(directory, name, expected), first_error);
    return first_error;
}

llama_dsv4_segment_status errno_status(int error) {
    if (error == ENOSPC || error == EDQUOT) {
        return llama_dsv4_segment_status::no_space;
    }
    if (error == ELOOP || error == EISDIR || error == ENOTDIR || error == EACCES || error == EPERM) {
        return llama_dsv4_segment_status::path_security;
    }
    return llama_dsv4_segment_status::io_error;
}

bool path_has_traversal(const fs::path & path) {
    if (!path.is_absolute() || path == path.root_path()) {
        return true;
    }
    for (const auto & component : path) {
        const std::string text = component.string();
        if (component == "." || component == ".." || text.empty() || text.find('\0') != std::string::npos) {
            return true;
        }
    }
    return false;
}

llama_dsv4_segment_status open_private_root(const fs::path & root, int & descriptor, int & os_error) {
    descriptor = -1;
    os_error   = 0;
    if (path_has_traversal(root)) {
        os_error = EINVAL;
        return llama_dsv4_segment_status::path_security;
    }
    int current = ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NONBLOCK);
    if (current < 0) {
        os_error = errno;
        return llama_dsv4_segment_status::io_error;
    }
    for (auto iterator = root.begin(); iterator != root.end(); ++iterator) {
        if (*iterator == root.root_path()) {
            continue;
        }
        const std::string name = iterator->string();
        int  next    = ::openat(current, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
        bool created = false;
        if (next < 0 && errno == ENOENT) {
            if (::mkdirat(current, name.c_str(), PRIVATE_DIRECTORY_MODE) == 0) {
                created = true;
            } else if (errno != EEXIST) {
                os_error = errno;
                ::close(current);
                return errno_status(os_error);
            }
            next = ::openat(current, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
        }
        if (next < 0) {
            os_error = errno;
            ::close(current);
            return errno_status(os_error);
        }
        struct stat status = {};
        if (::fstat(next, &status) != 0 || !S_ISDIR(status.st_mode)) {
            os_error = errno == 0 ? ENOTDIR : errno;
            ::close(next);
            ::close(current);
            return llama_dsv4_segment_status::path_security;
        }
        if (created) {
            int creation_error = 0;
            if (::fchmod(next, PRIVATE_DIRECTORY_MODE) != 0) {
                creation_error = errno;
            } else if (::fstat(next, &status) != 0) {
                creation_error = errno;
            } else if (!S_ISDIR(status.st_mode) || status.st_uid != ::geteuid() ||
                       (status.st_mode & 07777) != PRIVATE_DIRECTORY_MODE) {
                creation_error = EACCES;
            }
            if (creation_error == 0) {
                record_first_errno(::fsync(next), creation_error);
                record_first_errno(::fsync(current), creation_error);
            }
            if (creation_error != 0) {
                os_error = creation_error;
                ::close(next);
                ::close(current);
                return errno_status(os_error);
            }
        }
        ::close(current);
        current = next;
    }
    struct stat status = {};
    if (::fstat(current, &status) != 0 || !S_ISDIR(status.st_mode) || status.st_uid != ::geteuid()) {
        os_error = errno == 0 ? EACCES : errno;
        ::close(current);
        return llama_dsv4_segment_status::path_security;
    }
    if ((status.st_mode & 07777) != PRIVATE_DIRECTORY_MODE) {
        os_error = EACCES;
        ::close(current);
        return llama_dsv4_segment_status::path_security;
    }
    descriptor = current;
    return llama_dsv4_segment_status::ok;
}

llama_dsv4_segment_status open_private_directory_at(int parent, const char * name, int & descriptor, int & os_error) {
    descriptor   = ::openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    bool created = false;
    if (descriptor < 0 && errno == ENOENT) {
        if (::mkdirat(parent, name, PRIVATE_DIRECTORY_MODE) == 0) {
            created = true;
        } else if (errno != EEXIST) {
            os_error = errno;
            return errno_status(os_error);
        }
        descriptor = ::openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    }
    if (descriptor < 0) {
        os_error = errno;
        return errno_status(os_error);
    }
    struct stat status = {};
    if (::fstat(descriptor, &status) != 0 || !S_ISDIR(status.st_mode) || status.st_uid != ::geteuid() ||
        (!created && (status.st_mode & 07777) != PRIVATE_DIRECTORY_MODE)) {
        os_error = errno == 0 ? EACCES : errno;
        ::close(descriptor);
        descriptor = -1;
        return llama_dsv4_segment_status::path_security;
    }
    if (created) {
        int creation_error = 0;
        if (::fchmod(descriptor, PRIVATE_DIRECTORY_MODE) != 0) {
            creation_error = errno;
        } else if (::fstat(descriptor, &status) != 0) {
            creation_error = errno;
        } else if (!S_ISDIR(status.st_mode) || status.st_uid != ::geteuid() ||
                   (status.st_mode & 07777) != PRIVATE_DIRECTORY_MODE) {
            creation_error = EACCES;
        }
        if (creation_error == 0) {
            record_first_errno(::fsync(descriptor), creation_error);
            record_first_errno(::fsync(parent), creation_error);
        }
        if (creation_error != 0) {
            os_error = creation_error;
            ::close(descriptor);
            descriptor = -1;
            return errno_status(os_error);
        }
    }
    return llama_dsv4_segment_status::ok;
}

llama_dsv4_segment_status acquire_owner_lock(int root, int & descriptor, int & os_error) {
    bool created = false;
    descriptor   = ::openat(root, "owner.lock", O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK,
                            PRIVATE_FILE_MODE);
    if (descriptor >= 0) {
        created = true;
    } else if (errno == EEXIST) {
        descriptor = ::openat(root, "owner.lock", O_RDWR | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    }
    if (descriptor < 0) {
        os_error = errno;
        return errno_status(os_error);
    }
    struct stat status = {};
    if (::fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode) || status.st_uid != ::geteuid() ||
        (!created && (status.st_mode & 07777) != PRIVATE_FILE_MODE)) {
        os_error = errno == 0 ? EACCES : errno;
        ::close(descriptor);
        descriptor = -1;
        return llama_dsv4_segment_status::path_security;
    }
    if (created) {
        int creation_error = 0;
        if (::fchmod(descriptor, PRIVATE_FILE_MODE) != 0) {
            creation_error = errno;
        } else if (::fstat(descriptor, &status) != 0) {
            creation_error = errno;
        } else if (!private_regular_file(status)) {
            creation_error = EACCES;
        }
        if (creation_error == 0) {
            record_first_errno(::fsync(descriptor), creation_error);
            record_first_errno(::fsync(root), creation_error);
        }
        if (creation_error != 0) {
            os_error = creation_error;
            (void) close_and_cleanup_temporary(root, "owner.lock", status, descriptor);
            descriptor = -1;
            return errno_status(os_error);
        }
    }
    int lock_result;
    do {
        lock_result = ::flock(descriptor, LOCK_EX | LOCK_NB);
    } while (lock_result != 0 && errno == EINTR);
    if (lock_result != 0) {
        os_error = errno;
        ::close(descriptor);
        descriptor = -1;
        return os_error == EWOULDBLOCK || os_error == EAGAIN ? llama_dsv4_segment_status::owner_busy :
                                                               errno_status(os_error);
    }
    return llama_dsv4_segment_status::ok;
}

struct read_result {
    llama_dsv4_segment_status status = llama_dsv4_segment_status::invalid_argument;
    std::vector<uint8_t>      bytes;
    int                       os_error = 0;
};

struct compare_result {
    llama_dsv4_segment_status status   = llama_dsv4_segment_status::invalid_argument;
    bool                      equal    = false;
    int                       os_error = 0;
};

read_result read_private_file(int                       directory,
                              const std::string &       name,
                              uint64_t                  maximum,
                              llama_dsv4_segment_status missing_status) {
    read_result result;
    const int   descriptor = ::openat(directory, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    if (descriptor < 0) {
        result.os_error = errno;
        result.status   = errno == ENOENT ? missing_status : errno_status(errno);
        return result;
    }
    struct stat opened = {};
    if (::fstat(descriptor, &opened) != 0 || !private_regular_file(opened)) {
        result.os_error = errno == 0 ? EACCES : errno;
        result.status   = llama_dsv4_segment_status::path_security;
        ::close(descriptor);
        return result;
    }
    if (opened.st_size < 0 || static_cast<uint64_t>(opened.st_size) > maximum ||
        static_cast<uint64_t>(opened.st_size) > std::numeric_limits<size_t>::max()) {
        result.status = llama_dsv4_segment_status::manifest_too_large;
        ::close(descriptor);
        return result;
    }
    result.bytes.resize(static_cast<size_t>(opened.st_size));
    size_t offset = 0;
    while (offset < result.bytes.size()) {
        const ssize_t count = ::read(descriptor, result.bytes.data() + offset, result.bytes.size() - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            result.os_error = count == 0 ? 0 : errno;
            result.status   = count == 0 ? llama_dsv4_segment_status::truncated : errno_status(errno);
            ::close(descriptor);
            return result;
        }
        offset += static_cast<size_t>(count);
    }
    struct stat after = {};
    struct stat path  = {};
    if (::fstat(descriptor, &after) != 0 || ::fstatat(directory, name.c_str(), &path, AT_SYMLINK_NOFOLLOW) != 0 ||
        !private_regular_file(after) || !private_regular_file(path) || !same_file(opened, after) ||
        !same_file(after, path) || after.st_size != opened.st_size) {
        result.os_error = errno == 0 ? EACCES : errno;
        result.status   = llama_dsv4_segment_status::path_security;
        ::close(descriptor);
        return result;
    }
    if (::close(descriptor) != 0) {
        result.os_error = errno;
        result.status   = llama_dsv4_segment_status::io_error;
        return result;
    }
    result.status = llama_dsv4_segment_status::ok;
    return result;
}

compare_result compare_private_file(int                          directory,
                                    const std::string &          name,
                                    const std::vector<uint8_t> & expected,
                                    uint64_t                     maximum,
                                    llama_dsv4_segment_status    missing_status) {
    compare_result result;
    const int      descriptor = ::openat(directory, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    if (descriptor < 0) {
        result.os_error = errno;
        result.status   = errno == ENOENT ? missing_status : errno_status(errno);
        return result;
    }
    struct stat opened = {};
    if (::fstat(descriptor, &opened) != 0 || !private_regular_file(opened)) {
        result.os_error = errno == 0 ? EACCES : errno;
        result.status   = llama_dsv4_segment_status::path_security;
        (void) ::close(descriptor);
        return result;
    }
    if (opened.st_size < 0 || static_cast<uint64_t>(opened.st_size) > maximum) {
        result.status = llama_dsv4_segment_status::manifest_too_large;
        (void) ::close(descriptor);
        return result;
    }

    result.equal                           = static_cast<uint64_t>(opened.st_size) == expected.size();
    std::array<uint8_t, 16 * 1024> scratch = {};
    size_t                         offset  = 0;
    while (result.equal && offset < expected.size()) {
        const size_t  wanted = std::min(scratch.size(), expected.size() - offset);
        const ssize_t count  = ::read(descriptor, scratch.data(), wanted);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            result.os_error = count == 0 ? 0 : errno;
            result.status   = count == 0 ? llama_dsv4_segment_status::truncated : errno_status(errno);
            (void) ::close(descriptor);
            return result;
        }
        if (std::memcmp(scratch.data(), expected.data() + offset, static_cast<size_t>(count)) != 0) {
            result.equal = false;
            break;
        }
        offset += static_cast<size_t>(count);
    }

    struct stat after = {};
    struct stat path  = {};
    if (::fstat(descriptor, &after) != 0 || ::fstatat(directory, name.c_str(), &path, AT_SYMLINK_NOFOLLOW) != 0 ||
        !private_regular_file(after) || !private_regular_file(path) || !same_file(opened, after) ||
        !same_file(after, path) || after.st_size != opened.st_size) {
        result.os_error = errno == 0 ? EACCES : errno;
        result.status   = llama_dsv4_segment_status::path_security;
        (void) ::close(descriptor);
        return result;
    }
    if (::close(descriptor) != 0) {
        result.os_error = errno;
        result.status   = llama_dsv4_segment_status::io_error;
        return result;
    }
    result.status = llama_dsv4_segment_status::ok;
    return result;
}

struct write_result {
    llama_dsv4_segment_status status    = llama_dsv4_segment_status::invalid_argument;
    bool                      created   = false;
    bool                      committed = false;
    int                       os_error  = 0;
};

bool write_all(int descriptor, const std::vector<uint8_t> & bytes, int & os_error) {
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            os_error = count == 0 ? EIO : errno;
            return false;
        }
        offset += static_cast<size_t>(count);
    }
    return true;
}

std::string temporary_name(const char * category) {
    return std::string(".tmp-") + category + "-" + std::to_string(static_cast<uint64_t>(::getpid())) + "-" +
           std::to_string(temporary_sequence.fetch_add(1, std::memory_order_relaxed));
}

write_result publish_immutable_file(int                          directory,
                                    const std::string &          final_name,
                                    const std::vector<uint8_t> & bytes,
                                    uint64_t                     maximum,
                                    llama_dsv4_segment_status    missing_status) {
    write_result result;
    auto         existing = compare_private_file(directory, final_name, bytes, maximum, missing_status);
    if (existing.status == llama_dsv4_segment_status::ok) {
        result.status = existing.equal ? llama_dsv4_segment_status::ok : llama_dsv4_segment_status::checksum_mismatch;
        return result;
    }
    if (existing.status != missing_status) {
        result.status   = existing.status;
        result.os_error = existing.os_error;
        return result;
    }
    const std::string temporary = temporary_name("cas");
    const int         descriptor =
        ::openat(directory, temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK,
                 PRIVATE_FILE_MODE);
    if (descriptor < 0) {
        result.os_error = errno;
        result.status   = errno_status(errno);
        return result;
    }
    struct stat opened       = {};
    const int   fstat_result = ::fstat(descriptor, &opened);
    if (fstat_result != 0 || !private_regular_file(opened)) {
        result.os_error = fstat_result != 0 ? errno : EACCES;
        result.status   = llama_dsv4_segment_status::path_security;
        // Without a trustworthy descriptor identity it is safer to leave a
        // recognized temporary orphan than unlink a path an attacker may have
        // replaced. Reconciliation rejects or removes it descriptor-relatively.
        if (fstat_result == 0) {
            (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        } else {
            (void) ::close(descriptor);
        }
        return result;
    }
    if (!write_all(descriptor, bytes, result.os_error)) {
        result.status = result.os_error == EIO ? llama_dsv4_segment_status::short_write : errno_status(result.os_error);
        (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        return result;
    }
    if (::fsync(descriptor) != 0) {
        result.os_error = errno;
        result.status   = errno_status(errno);
        (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        return result;
    }
    struct stat path = {};
    if (::fstatat(directory, temporary.c_str(), &path, AT_SYMLINK_NOFOLLOW) != 0 || !private_regular_file(path) ||
        !same_file(opened, path)) {
        result.os_error = errno == 0 ? EACCES : errno;
        result.status   = llama_dsv4_segment_status::path_security;
        (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        return result;
    }
    if (::linkat(directory, temporary.c_str(), directory, final_name.c_str(), 0) != 0) {
        const int error = errno;
        (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        if (error == EEXIST) {
            existing = compare_private_file(directory, final_name, bytes, maximum, missing_status);
            result.status =
                existing.status == llama_dsv4_segment_status::ok && existing.equal ?
                    llama_dsv4_segment_status::ok :
                    (existing.status == llama_dsv4_segment_status::ok ? llama_dsv4_segment_status::checksum_mismatch :
                                                                        existing.status);
            result.os_error = existing.os_error;
            return result;
        }
        result.os_error = error;
        result.status   = errno_status(error);
        return result;
    }
    result.created   = true;
    result.committed = true;
    int final_error  = 0;
    record_first_errno(unlink_private_same(directory, temporary, opened), final_error);
    record_first_errno(::close(descriptor), final_error);
    record_first_errno(::fsync(directory), final_error);
    if (final_error != 0) {
        result.os_error = final_error;
        result.status   = llama_dsv4_segment_status::commit_uncertain;
        return result;
    }
    result.status = llama_dsv4_segment_status::ok;
    return result;
}

}  // namespace

struct llama_dsv4_segment_store_impl {
    explicit llama_dsv4_segment_store_impl(llama_dsv4_segment_store_config input) : cfg(std::move(input)) {}

    ~llama_dsv4_segment_store_impl() {
        if (refs_fd >= 0) {
            ::close(refs_fd);
        }
        if (manifests_fd >= 0) {
            ::close(manifests_fd);
        }
        if (chunks_fd >= 0) {
            ::close(chunks_fd);
        }
        if (lock_fd >= 0) {
            ::close(lock_fd);
        }
        if (root_fd >= 0) {
            ::close(root_fd);
        }
    }

    llama_dsv4_segment_store_config cfg;
    llama_dsv4_segment_status       init_status  = llama_dsv4_segment_status::invalid_argument;
    int                             init_error   = 0;
    int                             root_fd      = -1;
    int                             lock_fd      = -1;
    int                             chunks_fd    = -1;
    int                             manifests_fd = -1;
    int                             refs_fd      = -1;
    mutable std::mutex              mutex;
};

const char * llama_dsv4_segment_status_name(llama_dsv4_segment_status status) {
    switch (status) {
        case llama_dsv4_segment_status::ok:
            return "ok";
        case llama_dsv4_segment_status::invalid_argument:
            return "invalid_argument";
        case llama_dsv4_segment_status::unsupported_version:
            return "unsupported_version";
        case llama_dsv4_segment_status::invalid_schema:
            return "invalid_schema";
        case llama_dsv4_segment_status::identity_mismatch:
            return "identity_mismatch";
        case llama_dsv4_segment_status::coverage_mismatch:
            return "coverage_mismatch";
        case llama_dsv4_segment_status::state_too_large:
            return "state_too_large";
        case llama_dsv4_segment_status::manifest_too_large:
            return "manifest_too_large";
        case llama_dsv4_segment_status::too_many_chunks:
            return "too_many_chunks";
        case llama_dsv4_segment_status::no_current_manifest:
            return "no_current_manifest";
        case llama_dsv4_segment_status::missing_parent:
            return "missing_parent";
        case llama_dsv4_segment_status::missing_manifest:
            return "missing_manifest";
        case llama_dsv4_segment_status::missing_chunk:
            return "missing_chunk";
        case llama_dsv4_segment_status::malformed:
            return "malformed";
        case llama_dsv4_segment_status::truncated:
            return "truncated";
        case llama_dsv4_segment_status::trailing_data:
            return "trailing_data";
        case llama_dsv4_segment_status::checksum_mismatch:
            return "checksum_mismatch";
        case llama_dsv4_segment_status::path_security:
            return "path_security";
        case llama_dsv4_segment_status::owner_busy:
            return "owner_busy";
        case llama_dsv4_segment_status::resource_exhausted:
            return "resource_exhausted";
        case llama_dsv4_segment_status::no_space:
            return "no_space";
        case llama_dsv4_segment_status::short_write:
            return "short_write";
        case llama_dsv4_segment_status::io_error:
            return "io_error";
        case llama_dsv4_segment_status::commit_uncertain:
            return "commit_uncertain";
        case llama_dsv4_segment_status::injected_failure:
            return "injected_failure";
        case llama_dsv4_segment_status::import_rejected:
            return "import_rejected";
    }
    return "unknown";
}

bool operator==(const llama_dsv4_segment_identity & lhs, const llama_dsv4_segment_identity & rhs) {
    return lhs.geometry_identity == rhs.geometry_identity && lhs.model_artifact_digest == rhs.model_artifact_digest;
}

bool operator!=(const llama_dsv4_segment_identity & lhs, const llama_dsv4_segment_identity & rhs) {
    return !(lhs == rhs);
}

llama_dsv4_segment_store::llama_dsv4_segment_store(llama_dsv4_segment_store_config config) :
    pimpl(new llama_dsv4_segment_store_impl(std::move(config))) {
    try {
        uint64_t   chunk_payload_capacity = 0;
        const bool chunk_capacity_valid =
            checked_multiply(pimpl->cfg.max_chunks, LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES, chunk_payload_capacity);
        if (pimpl->cfg.root_path.empty() || pimpl->cfg.max_chunks == 0 ||
            pimpl->cfg.max_chunks > LLAMA_DSV4_SEGMENT_MAX_CHUNKS || pimpl->cfg.max_manifest_bytes < 1024 ||
            pimpl->cfg.max_manifest_bytes > LLAMA_DSV4_SEGMENT_MAX_MANIFEST_BYTES || pimpl->cfg.max_state_bytes == 0 ||
            pimpl->cfg.max_state_bytes > LLAMA_DSV4_SEGMENT_MAX_STATE_BYTES || !chunk_capacity_valid ||
            pimpl->cfg.max_state_bytes > chunk_payload_capacity) {
            pimpl->init_status = llama_dsv4_segment_status::invalid_argument;
            return;
        }
        pimpl->init_status = open_private_root(pimpl->cfg.root_path, pimpl->root_fd, pimpl->init_error);
        if (pimpl->init_status != llama_dsv4_segment_status::ok) {
            return;
        }
        pimpl->init_status = acquire_owner_lock(pimpl->root_fd, pimpl->lock_fd, pimpl->init_error);
        if (pimpl->init_status != llama_dsv4_segment_status::ok) {
            return;
        }
        pimpl->init_status = open_private_directory_at(pimpl->root_fd, "chunks", pimpl->chunks_fd, pimpl->init_error);
        if (pimpl->init_status != llama_dsv4_segment_status::ok) {
            return;
        }
        pimpl->init_status =
            open_private_directory_at(pimpl->root_fd, "manifests", pimpl->manifests_fd, pimpl->init_error);
        if (pimpl->init_status != llama_dsv4_segment_status::ok) {
            return;
        }
        pimpl->init_status = open_private_directory_at(pimpl->root_fd, "refs", pimpl->refs_fd, pimpl->init_error);
    } catch (const std::bad_alloc &) {
        pimpl->init_status = llama_dsv4_segment_status::resource_exhausted;
    } catch (const std::length_error &) {
        pimpl->init_status = llama_dsv4_segment_status::invalid_argument;
    } catch (...) {
        pimpl->init_status = llama_dsv4_segment_status::path_security;
        pimpl->init_error  = EINVAL;
    }
}

llama_dsv4_segment_store::~llama_dsv4_segment_store() = default;

const llama_dsv4_segment_store_config & llama_dsv4_segment_store::config() const {
    return pimpl->cfg;
}

namespace {

void add_plane_field_records(std::vector<field_record> &         fields,
                             const plane_meta &                  plane,
                             bool                                base,
                             const llama_dsv4_segment_identity & identity) {
    field_record positions;
    positions.encoding  = field_encoding::positions;
    positions.path.kind = base ? field_kind::raw_base_positions : field_kind::raw_swa_positions;
    positions.key       = semantic_key(identity, positions.path);
    checked_multiply(plane.position_count, 8, positions.logical_size);
    fields.push_back(positions);
    field_record extensions;
    extensions.encoding  = field_encoding::extensions;
    extensions.path.kind = base ? field_kind::raw_base_extensions : field_kind::raw_swa_extensions;
    extensions.key       = semantic_key(identity, extensions.path);
    checked_multiply(plane.extension_count, 16, extensions.logical_size);
    fields.push_back(extensions);
    field_record tensor;
    tensor.encoding     = field_encoding::bytes;
    tensor.path.kind    = base ? field_kind::raw_base_tensor : field_kind::raw_swa_tensor;
    tensor.key          = semantic_key(identity, tensor.path);
    tensor.logical_size = plane.tensor_bytes;
    fields.push_back(tensor);
}

void add_tensor_field_records(std::vector<field_record> &         fields,
                              const std::vector<tensor_meta> &    tensors,
                              bool                                recurrent,
                              uint32_t                            group,
                              uint32_t                            tensor_class,
                              const llama_dsv4_segment_identity & identity) {
    for (uint32_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        const auto & tensor = tensors[tensor_index];
        for (const auto & chunk : tensor.chunks) {
            field_record field;
            field.encoding            = field_encoding::bytes;
            field.path.kind           = recurrent ? field_kind::recurrent_rows : field_kind::compressed_rows;
            field.path.group          = group;
            field.path.tensor_class   = tensor_class;
            field.path.tensor_ordinal = tensor_index;
            field.path.layer_id       = tensor.layer_id;
            field.path.type           = tensor.type;
            field.path.ne0            = tensor.ne0;
            field.path.row_size       = tensor.row_size;
            field.path.row_begin      = chunk.row_begin;
            field.path.row_count      = chunk.row_count;
            field.key                 = semantic_key(identity, field.path);
            field.logical_size        = chunk.byte_count;
            fields.push_back(field);
        }
    }
}

std::vector<field_record> make_field_records(const sequence_meta & meta, const llama_dsv4_segment_identity & identity) {
    std::vector<field_record> fields;
    add_plane_field_records(fields, meta.base, true, identity);
    add_plane_field_records(fields, meta.swa, false, identity);
    add_tensor_field_records(fields, meta.csa.tensors, false, 0, 0, identity);
    add_tensor_field_records(fields, meta.hca.tensors, false, 1, 0, identity);
    add_tensor_field_records(fields, meta.lid.tensors, false, 2, 0, identity);
    add_tensor_field_records(fields, meta.csa_recurrent.kv, true, 0, 0, identity);
    add_tensor_field_records(fields, meta.csa_recurrent.score, true, 0, 1, identity);
    add_tensor_field_records(fields, meta.hca_recurrent.kv, true, 1, 0, identity);
    add_tensor_field_records(fields, meta.hca_recurrent.score, true, 1, 1, identity);
    add_tensor_field_records(fields, meta.lid_recurrent.kv, true, 2, 0, identity);
    add_tensor_field_records(fields, meta.lid_recurrent.score, true, 2, 1, identity);
    return fields;
}

uint64_t fragment_count(const field_record & field) {
    const uint64_t limit = fragment_payload_limit(field.encoding);
    return field.logical_size == 0 ? 0 : 1 + (field.logical_size - 1) / limit;
}

void append_field_path(std::vector<uint8_t> & bytes, const field_path & path) {
    append_u32(bytes, static_cast<uint32_t>(path.kind));
    append_u32(bytes, path.group);
    append_u32(bytes, path.tensor_class);
    append_u32(bytes, path.tensor_ordinal);
    append_u32(bytes, path.layer_id);
    append_u32(bytes, static_cast<uint32_t>(path.type));
    append_u64(bytes, path.ne0);
    append_u64(bytes, path.row_size);
    append_u64(bytes, path.row_begin);
    append_u64(bytes, path.row_count);
}

bool read_field_path(cursor & input, field_path & path) {
    uint32_t kind = 0;
    uint32_t type = 0;
    if (!input.u32(kind) || !input.u32(path.group) || !input.u32(path.tensor_class) ||
        !input.u32(path.tensor_ordinal) || !input.u32(path.layer_id) || !input.u32(type) || !input.u64(path.ne0) ||
        !input.u64(path.row_size) || !input.u64(path.row_begin) || !input.u64(path.row_count)) {
        return false;
    }
    path.kind = static_cast<field_kind>(kind);
    path.type = static_cast<int32_t>(type);
    return true;
}

std::vector<uint8_t> serialize_manifest(const parsed_manifest & manifest) {
    uint64_t total_segments = 0;
    for (const auto & field : manifest.fields) {
        total_segments += field.fragments.size();
    }
    std::vector<uint8_t> bytes;
    bytes.insert(bytes.end(), MANIFEST_MAGIC.begin(), MANIFEST_MAGIC.end());
    append_u32(bytes, LLAMA_DSV4_SEGMENT_FORMAT_VERSION);
    append_u32(bytes, manifest.has_parent ? 1U : 0U);
    append_u64(bytes, manifest.identity.geometry_identity);
    append_digest(bytes, manifest.identity.model_artifact_digest);
    append_u64(bytes, manifest.prefix.token_count);
    append_u32(bytes, manifest.prefix.radix_depth);
    append_u32(bytes, 0);
    append_digest(bytes, manifest.prefix.token_digest);
    append_digest(bytes, manifest.parent);
    append_u64(bytes, manifest.fingerprint);
    append_u64(bytes, manifest.total_payload);
    append_u64(bytes, manifest.envelope.size());
    append_u32(bytes, static_cast<uint32_t>(manifest.fields.size()));
    append_u32(bytes, static_cast<uint32_t>(total_segments));
    bytes.insert(bytes.end(), manifest.envelope.begin(), manifest.envelope.end());
    for (const auto & field : manifest.fields) {
        append_u32(bytes, static_cast<uint32_t>(field.encoding));
        append_u32(bytes, 0);
        append_field_path(bytes, field.path);
        append_digest(bytes, field.key);
        append_u64(bytes, field.logical_size);
        append_u32(bytes, static_cast<uint32_t>(field.fragments.size()));
        append_u32(bytes, 0);
        for (const auto & fragment : field.fragments) {
            append_u64(bytes, fragment.offset);
            append_u64(bytes, fragment.size);
            append_digest(bytes, fragment.digest);
            bytes.push_back(fragment.reused ? 1 : 0);
            bytes.insert(bytes.end(), 7, 0);
        }
    }
    return bytes;
}

llama_dsv4_segment_status validate_prefix(const llama_dsv4_segment_prefix_metadata & prefix) {
    if (prefix.radix_depth > prefix.token_count || (prefix.token_count != 0 && digest_zero(prefix.token_digest))) {
        return llama_dsv4_segment_status::invalid_argument;
    }
    return llama_dsv4_segment_status::ok;
}

llama_dsv4_segment_status parse_manifest_bytes(const std::vector<uint8_t> &            bytes,
                                               const llama_dsv4_segment_store_config & cfg,
                                               parsed_manifest &                       manifest) {
    if (bytes.size() > cfg.max_manifest_bytes) {
        return llama_dsv4_segment_status::manifest_too_large;
    }
    cursor                 input{ bytes };
    std::array<uint8_t, 8> magic          = {};
    uint32_t               version        = 0;
    uint32_t               flags          = 0;
    uint32_t               reserved       = 0;
    uint64_t               envelope_size  = 0;
    uint32_t               field_count    = 0;
    uint32_t               total_segments = 0;
    if (!input.take(magic.data(), magic.size()) || !input.u32(version) || !input.u32(flags)) {
        return llama_dsv4_segment_status::truncated;
    }
    if (magic != MANIFEST_MAGIC) {
        return llama_dsv4_segment_status::malformed;
    }
    if (version != LLAMA_DSV4_SEGMENT_FORMAT_VERSION) {
        return llama_dsv4_segment_status::unsupported_version;
    }
    if (flags > 1 || !input.u64(manifest.identity.geometry_identity) ||
        !input.digest(manifest.identity.model_artifact_digest) || !input.u64(manifest.prefix.token_count) ||
        !input.u32(manifest.prefix.radix_depth) || !input.u32(reserved) || reserved != 0 ||
        !input.digest(manifest.prefix.token_digest) || !input.digest(manifest.parent) ||
        !input.u64(manifest.fingerprint) || !input.u64(manifest.total_payload) || !input.u64(envelope_size) ||
        !input.u32(field_count) || !input.u32(total_segments)) {
        return llama_dsv4_segment_status::truncated;
    }
    manifest.has_parent = flags != 0;
    if ((!manifest.has_parent && !digest_zero(manifest.parent)) ||
        (manifest.has_parent && digest_zero(manifest.parent)) || digest_zero(manifest.identity.model_artifact_digest) ||
        manifest.identity.geometry_identity == 0 || validate_prefix(manifest.prefix) != llama_dsv4_segment_status::ok) {
        return llama_dsv4_segment_status::malformed;
    }
    if (envelope_size > cfg.max_manifest_bytes || envelope_size > bytes.size() - input.offset ||
        field_count > MAX_DESCRIPTOR_COUNT || total_segments > cfg.max_chunks) {
        return envelope_size > cfg.max_manifest_bytes ? llama_dsv4_segment_status::manifest_too_large :
                                                        llama_dsv4_segment_status::too_many_chunks;
    }
    manifest.envelope.resize(static_cast<size_t>(envelope_size));
    if (!input.take(manifest.envelope.data(), manifest.envelope.size())) {
        return llama_dsv4_segment_status::truncated;
    }
    auto status = parse_envelope(manifest.envelope, manifest.meta);
    if (status != llama_dsv4_segment_status::ok) {
        return status;
    }
    uint64_t payload         = 0;
    uint64_t expected_fields = 0;
    status = validate_meta(manifest.meta, manifest.identity, cfg.max_state_bytes, payload, expected_fields);
    if (status != llama_dsv4_segment_status::ok) {
        return status;
    }
    if (payload != manifest.total_payload || manifest.meta.fingerprint != manifest.fingerprint ||
        expected_fields != field_count) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    auto expected = make_field_records(manifest.meta, manifest.identity);
    manifest.fields.resize(field_count);
    uint64_t observed_segments = 0;
    for (uint32_t index = 0; index < field_count; ++index) {
        auto &   field                = manifest.fields[index];
        uint32_t encoding             = 0;
        uint32_t fragment_count_value = 0;
        if (!input.u32(encoding) || !read_reserved(input) || !read_field_path(input, field.path) ||
            !input.digest(field.key) || !input.u64(field.logical_size) || !input.u32(fragment_count_value) ||
            !read_reserved(input)) {
            return llama_dsv4_segment_status::truncated;
        }
        field.encoding = static_cast<field_encoding>(encoding);
        if (field.encoding != expected[index].encoding || field.path.kind != expected[index].path.kind ||
            field.path.group != expected[index].path.group ||
            field.path.tensor_class != expected[index].path.tensor_class ||
            field.path.tensor_ordinal != expected[index].path.tensor_ordinal ||
            field.path.layer_id != expected[index].path.layer_id || field.path.type != expected[index].path.type ||
            field.path.ne0 != expected[index].path.ne0 || field.path.row_size != expected[index].path.row_size ||
            field.path.row_begin != expected[index].path.row_begin ||
            field.path.row_count != expected[index].path.row_count || field.key != expected[index].key ||
            field.logical_size != expected[index].logical_size ||
            fragment_count_value != fragment_count(expected[index]) ||
            observed_segments > cfg.max_chunks - fragment_count_value) {
            return llama_dsv4_segment_status::coverage_mismatch;
        }
        observed_segments += fragment_count_value;
        field.fragments.resize(fragment_count_value);
        uint64_t next_offset = 0;
        for (auto & fragment : field.fragments) {
            uint8_t reused    = 0;
            uint8_t zeroes[7] = {};
            if (!input.u64(fragment.offset) || !input.u64(fragment.size) || !input.digest(fragment.digest) ||
                !input.take(&reused, 1) || !input.take(zeroes, sizeof(zeroes))) {
                return llama_dsv4_segment_status::truncated;
            }
            if (reused > 1 ||
                std::any_of(std::begin(zeroes), std::end(zeroes), [](uint8_t value) { return value != 0; }) ||
                fragment.offset != next_offset || fragment.size == 0 ||
                fragment.size > fragment_payload_limit(field.encoding) ||
                fragment.size > field.logical_size - fragment.offset || digest_zero(fragment.digest)) {
                return llama_dsv4_segment_status::coverage_mismatch;
            }
            const uint64_t unit = field.encoding == field_encoding::positions  ? 8 :
                                  field.encoding == field_encoding::extensions ? 16 :
                                                                                 1;
            if (fragment.offset % unit != 0 || fragment.size % unit != 0) {
                return llama_dsv4_segment_status::coverage_mismatch;
            }
            fragment.reused = reused != 0;
            next_offset += fragment.size;
        }
        if (next_offset != field.logical_size) {
            return llama_dsv4_segment_status::coverage_mismatch;
        }
    }
    if (observed_segments != total_segments) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    return input.offset == bytes.size() ? llama_dsv4_segment_status::ok : llama_dsv4_segment_status::trailing_data;
}

llama_dsv4_segment_manifest public_manifest(const parsed_manifest &       parsed,
                                            const llama_snapshot_digest & digest,
                                            uint64_t                      encoded_bytes) {
    llama_dsv4_segment_manifest result;
    result.format_version         = LLAMA_DSV4_SEGMENT_FORMAT_VERSION;
    result.digest                 = digest;
    result.identity               = parsed.identity;
    result.prefix                 = parsed.prefix;
    result.has_parent             = parsed.has_parent;
    result.parent_digest          = parsed.parent;
    result.logical_fingerprint    = parsed.fingerprint;
    result.logical_payload_bytes  = parsed.total_payload;
    result.encoded_manifest_bytes = encoded_bytes;
    for (const auto & field : parsed.fields) {
        for (const auto & fragment : field.fragments) {
            result.segments.push_back({ field.key, fragment.digest, fragment.offset, fragment.size, fragment.reused });
            if (fragment.reused) {
                ++result.reused_segments;
            }
        }
    }
    return result;
}

struct parsed_open_result {
    llama_dsv4_segment_status status = llama_dsv4_segment_status::invalid_argument;
    parsed_manifest           parsed;
    uint64_t                  encoded_bytes = 0;
    int                       os_error      = 0;
};

parsed_open_result open_parsed_manifest_locked(const llama_dsv4_segment_store_impl & store,
                                               const llama_snapshot_digest &         digest,
                                               const llama_dsv4_segment_identity *   expected) {
    parsed_open_result result;
    if (store.init_status != llama_dsv4_segment_status::ok) {
        result.status   = store.init_status;
        result.os_error = store.init_error;
        return result;
    }
    if (digest_zero(digest)) {
        result.status = llama_dsv4_segment_status::invalid_argument;
        return result;
    }
    const auto read = read_private_file(store.manifests_fd, manifest_filename(digest), store.cfg.max_manifest_bytes,
                                        llama_dsv4_segment_status::missing_manifest);
    if (read.status != llama_dsv4_segment_status::ok) {
        result.status   = read.status;
        result.os_error = read.os_error;
        return result;
    }
    if (llama_snapshot_sha256(read.bytes.data(), read.bytes.size()) != digest) {
        result.status = llama_dsv4_segment_status::checksum_mismatch;
        return result;
    }
    result.status = parse_manifest_bytes(read.bytes, store.cfg, result.parsed);
    if (result.status != llama_dsv4_segment_status::ok) {
        return result;
    }
    if (expected != nullptr && result.parsed.identity != *expected) {
        result.status = llama_dsv4_segment_status::identity_mismatch;
        return result;
    }
    result.encoded_bytes = read.bytes.size();
    return result;
}

llama_dsv4_segment_open_result open_manifest_locked(const llama_dsv4_segment_store_impl & store,
                                                    const llama_snapshot_digest &         digest,
                                                    const llama_dsv4_segment_identity *   expected) {
    llama_dsv4_segment_open_result result;
    const auto                     parsed = open_parsed_manifest_locked(store, digest, expected);
    result.status                         = parsed.status;
    result.os_error                       = parsed.os_error;
    if (parsed.status == llama_dsv4_segment_status::ok) {
        result.manifest = public_manifest(parsed.parsed, digest, parsed.encoded_bytes);
    }
    return result;
}

std::vector<uint8_t> make_ref_bytes(const llama_snapshot_digest &       manifest,
                                    const llama_dsv4_segment_identity & identity) {
    std::vector<uint8_t> bytes;
    bytes.insert(bytes.end(), REF_MAGIC.begin(), REF_MAGIC.end());
    append_u32(bytes, LLAMA_DSV4_SEGMENT_FORMAT_VERSION);
    append_u32(bytes, 0);
    append_digest(bytes, manifest);
    append_u64(bytes, identity.geometry_identity);
    append_digest(bytes, identity.model_artifact_digest);
    const auto checksum = llama_snapshot_sha256(bytes.data(), bytes.size());
    append_digest(bytes, checksum);
    return bytes;
}

llama_dsv4_segment_status parse_ref_bytes(const std::vector<uint8_t> &  bytes,
                                          llama_snapshot_digest &       manifest,
                                          llama_dsv4_segment_identity & identity) {
    if (bytes.size() < REF_BYTES) {
        return llama_dsv4_segment_status::truncated;
    }
    if (bytes.size() > REF_BYTES) {
        return llama_dsv4_segment_status::trailing_data;
    }
    const auto            actual   = llama_snapshot_sha256(bytes.data(), bytes.size() - 32);
    llama_snapshot_digest expected = {};
    std::copy_n(bytes.end() - 32, 32, expected.begin());
    if (actual != expected) {
        return llama_dsv4_segment_status::checksum_mismatch;
    }
    cursor                 input{ bytes };
    std::array<uint8_t, 8> magic    = {};
    uint32_t               version  = 0;
    uint32_t               reserved = 0;
    llama_snapshot_digest  ignored  = {};
    if (!input.take(magic.data(), magic.size()) || !input.u32(version) || !input.u32(reserved) ||
        !input.digest(manifest) || !input.u64(identity.geometry_identity) ||
        !input.digest(identity.model_artifact_digest) || !input.digest(ignored)) {
        return llama_dsv4_segment_status::truncated;
    }
    if (magic != REF_MAGIC || reserved != 0) {
        return llama_dsv4_segment_status::malformed;
    }
    if (version != LLAMA_DSV4_SEGMENT_FORMAT_VERSION) {
        return llama_dsv4_segment_status::unsupported_version;
    }
    return digest_zero(manifest) || digest_zero(identity.model_artifact_digest) || identity.geometry_identity == 0 ?
               llama_dsv4_segment_status::malformed :
               llama_dsv4_segment_status::ok;
}

write_result publish_ref_file(int                          directory,
                              const std::string &          name,
                              const std::vector<uint8_t> & bytes,
                              bool                         fail_after_rename) {
    write_result      result;
    const std::string temporary = temporary_name("ref");
    const int         descriptor =
        ::openat(directory, temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK,
                 PRIVATE_FILE_MODE);
    if (descriptor < 0) {
        result.os_error = errno;
        result.status   = errno_status(errno);
        return result;
    }
    struct stat opened       = {};
    const int   fstat_result = ::fstat(descriptor, &opened);
    if (fstat_result != 0 || !private_regular_file(opened)) {
        result.os_error = fstat_result != 0 ? errno : EACCES;
        result.status   = llama_dsv4_segment_status::path_security;
        if (fstat_result == 0) {
            (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        } else {
            (void) ::close(descriptor);
        }
        return result;
    }
    if (!write_all(descriptor, bytes, result.os_error)) {
        result.status = result.os_error == EIO ? llama_dsv4_segment_status::short_write : errno_status(result.os_error);
        (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        return result;
    }
    if (::fsync(descriptor) != 0) {
        result.os_error = errno;
        result.status   = errno_status(result.os_error);
        (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        return result;
    }
    struct stat temporary_path = {};
    if (::fstatat(directory, temporary.c_str(), &temporary_path, AT_SYMLINK_NOFOLLOW) != 0 ||
        !private_regular_file(temporary_path) || !same_file(opened, temporary_path)) {
        result.os_error = errno == 0 ? EACCES : errno;
        result.status   = llama_dsv4_segment_status::path_security;
        (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        return result;
    }
    struct stat current        = {};
    const int   current_result = ::fstatat(directory, name.c_str(), &current, AT_SYMLINK_NOFOLLOW);
    if (current_result == 0 && !private_regular_file(current)) {
        result.status   = llama_dsv4_segment_status::path_security;
        result.os_error = EACCES;
        (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        return result;
    }
    if (current_result != 0 && errno != ENOENT) {
        result.os_error = errno;
        result.status   = errno_status(errno);
        (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        return result;
    }
    if (::renameat(directory, temporary.c_str(), directory, name.c_str()) != 0) {
        result.os_error = errno;
        result.status   = errno_status(errno);
        (void) close_and_cleanup_temporary(directory, temporary, opened, descriptor);
        return result;
    }
    result.committed = true;
    result.created   = true;
    int final_error  = 0;
    record_first_errno(::close(descriptor), final_error);
    if (!fail_after_rename) {
        record_first_errno(::fsync(directory), final_error);
    }
    if (fail_after_rename || final_error != 0) {
        result.os_error = fail_after_rename ? 0 : final_error;
        result.status   = llama_dsv4_segment_status::commit_uncertain;
        return result;
    }
    result.status = llama_dsv4_segment_status::ok;
    return result;
}

std::vector<uint8_t> make_chunk_bytes(const llama_dsv4_segment_identity & identity,
                                      const field_source &                field,
                                      uint64_t                            offset,
                                      uint64_t                            size) {
    std::vector<uint8_t> bytes;
    bytes.reserve(static_cast<size_t>(CHUNK_HEADER_BYTES + size));
    bytes.insert(bytes.end(), CHUNK_MAGIC.begin(), CHUNK_MAGIC.end());
    append_u32(bytes, LLAMA_DSV4_SEGMENT_CHUNK_FORMAT_VERSION);
    append_u32(bytes, LLAMA_DSV4_LOGICAL_STATE_SCHEMA);
    append_u64(bytes, identity.geometry_identity);
    append_digest(bytes, identity.model_artifact_digest);
    append_digest(bytes, field.key);
    append_u64(bytes, offset);
    append_u64(bytes, size);
    if (field.encoding == field_encoding::bytes) {
        bytes.insert(bytes.end(), field.bytes->begin() + static_cast<std::ptrdiff_t>(offset),
                     field.bytes->begin() + static_cast<std::ptrdiff_t>(offset + size));
    } else if (field.encoding == field_encoding::positions) {
        const uint64_t first = offset / 8;
        for (uint64_t index = 0; index < size / 8; ++index) {
            append_i64(bytes, field.positions->at(static_cast<size_t>(first + index)));
        }
    } else {
        const uint64_t first = offset / 16;
        for (uint64_t index = 0; index < size / 16; ++index) {
            const auto & extension = field.extensions->at(static_cast<size_t>(first + index));
            append_i64(bytes, extension.x);
            append_i64(bytes, extension.y);
        }
    }
    return bytes;
}

llama_dsv4_segment_status validate_chunk_bytes(const std::vector<uint8_t> &        bytes,
                                               const llama_dsv4_segment_identity & identity,
                                               const field_record &                field,
                                               const fragment_record &             fragment) {
    if (bytes.size() < CHUNK_HEADER_BYTES) {
        return llama_dsv4_segment_status::truncated;
    }
    if (bytes.size() != CHUNK_HEADER_BYTES + fragment.size) {
        return bytes.size() < CHUNK_HEADER_BYTES + fragment.size ? llama_dsv4_segment_status::truncated :
                                                                   llama_dsv4_segment_status::trailing_data;
    }
    if (llama_snapshot_sha256(bytes.data(), bytes.size()) != fragment.digest) {
        return llama_dsv4_segment_status::checksum_mismatch;
    }
    cursor                      input{ bytes };
    std::array<uint8_t, 8>      magic   = {};
    uint32_t                    version = 0;
    uint32_t                    schema  = 0;
    llama_dsv4_segment_identity stored;
    llama_snapshot_digest       key    = {};
    uint64_t                    offset = 0;
    uint64_t                    size   = 0;
    if (!input.take(magic.data(), magic.size()) || !input.u32(version) || !input.u32(schema) ||
        !input.u64(stored.geometry_identity) || !input.digest(stored.model_artifact_digest) || !input.digest(key) ||
        !input.u64(offset) || !input.u64(size)) {
        return llama_dsv4_segment_status::truncated;
    }
    if (magic != CHUNK_MAGIC || version != LLAMA_DSV4_SEGMENT_CHUNK_FORMAT_VERSION ||
        schema != LLAMA_DSV4_LOGICAL_STATE_SCHEMA || stored != identity || key != field.key ||
        offset != fragment.offset || size != fragment.size || input.offset != CHUNK_HEADER_BYTES) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    return llama_dsv4_segment_status::ok;
}

using parent_key = std::pair<std::pair<llama_snapshot_digest, uint64_t>, uint64_t>;

std::map<parent_key, llama_snapshot_digest> parent_segments(const parsed_manifest & parent) {
    std::map<parent_key, llama_snapshot_digest> result;
    for (const auto & field : parent.fields) {
        for (const auto & fragment : field.fragments) {
            result.emplace(
                parent_key{
                    { field.key, fragment.offset },
                    fragment.size
            },
                fragment.digest);
        }
    }
    return result;
}

}  // namespace

llama_dsv4_segment_measurement llama_dsv4_segment_store::measure(
    const llama_dsv4_logical_sequence_state &  state,
    const llama_dsv4_segment_identity &        identity,
    const llama_dsv4_segment_prefix_metadata & prefix) const {
    llama_dsv4_segment_measurement result;
    try {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        if (pimpl->init_status != llama_dsv4_segment_status::ok) {
            result.status = pimpl->init_status;
            return result;
        }
        result.status = validate_prefix(prefix);
        if (result.status != llama_dsv4_segment_status::ok) {
            return result;
        }
        sequence_meta             meta;
        std::vector<uint8_t>      envelope;
        std::vector<field_source> sources;
        uint64_t                  payload = 0;
        result.status = validate_state(state, identity, pimpl->cfg.max_state_bytes, meta, envelope, sources, payload);
        if (result.status != llama_dsv4_segment_status::ok) {
            return result;
        }
        parsed_manifest manifest;
        manifest.identity      = identity;
        manifest.prefix        = prefix;
        manifest.fingerprint   = state.fingerprint;
        manifest.total_payload = payload;
        manifest.envelope      = std::move(envelope);
        manifest.meta          = std::move(meta);
        manifest.fields        = make_field_records(manifest.meta, identity);
        uint64_t chunks        = 0;
        for (auto & field : manifest.fields) {
            const uint64_t count = fragment_count(field);
            if (count > pimpl->cfg.max_chunks - chunks) {
                result.status = llama_dsv4_segment_status::too_many_chunks;
                return result;
            }
            chunks += count;
            field.fragments.resize(static_cast<size_t>(count));
        }
        const auto serialized = serialize_manifest(manifest);
        if (serialized.size() > pimpl->cfg.max_manifest_bytes) {
            result.status = llama_dsv4_segment_status::manifest_too_large;
            return result;
        }
        result.status                   = llama_dsv4_segment_status::ok;
        result.logical_payload_bytes    = payload;
        result.encoded_manifest_bytes   = serialized.size();
        result.segment_count            = static_cast<uint32_t>(chunks);
        result.peak_codec_scratch_bytes = std::max<uint64_t>(
            serialized.size(), chunks == 0 ? 0 : CHUNK_HEADER_BYTES + LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_segment_status::resource_exhausted;
        return result;
    } catch (const std::length_error &) {
        result.status = llama_dsv4_segment_status::state_too_large;
        return result;
    }
}

llama_dsv4_segment_publish_result llama_dsv4_segment_store::publish(const std::string &                        ref_name,
                                                                    const llama_dsv4_logical_sequence_state &  state,
                                                                    const llama_dsv4_segment_identity &        identity,
                                                                    const llama_dsv4_segment_prefix_metadata & prefix,
                                                                    const llama_dsv4_segment_manifest *        parent,
                                                                    const llama_dsv4_segment_faults &          faults) {
    llama_dsv4_segment_publish_result result;
    try {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        if (pimpl->init_status != llama_dsv4_segment_status::ok) {
            result.status   = pimpl->init_status;
            result.os_error = pimpl->init_error;
            return result;
        }
        if (!safe_ref_name(ref_name)) {
            result.status = llama_dsv4_segment_status::invalid_argument;
            return result;
        }
        result.status = validate_prefix(prefix);
        if (result.status != llama_dsv4_segment_status::ok) {
            return result;
        }
        sequence_meta             meta;
        std::vector<uint8_t>      envelope;
        std::vector<field_source> sources;
        uint64_t                  payload = 0;
        result.status = validate_state(state, identity, pimpl->cfg.max_state_bytes, meta, envelope, sources, payload);
        if (result.status != llama_dsv4_segment_status::ok) {
            return result;
        }

        parsed_manifest manifest;
        manifest.identity      = identity;
        manifest.prefix        = prefix;
        manifest.fingerprint   = state.fingerprint;
        manifest.total_payload = payload;
        manifest.envelope      = std::move(envelope);
        manifest.meta          = std::move(meta);
        manifest.fields        = make_field_records(manifest.meta, identity);
        std::map<parent_key, llama_snapshot_digest> inherited;
        if (parent != nullptr) {
            const auto opened = open_parsed_manifest_locked(*pimpl, parent->digest, &identity);
            if (opened.status != llama_dsv4_segment_status::ok) {
                result.status   = opened.status == llama_dsv4_segment_status::missing_manifest ?
                                      llama_dsv4_segment_status::missing_parent :
                                      opened.status;
                result.os_error = opened.os_error;
                return result;
            }
            manifest.has_parent = true;
            manifest.parent     = parent->digest;
            inherited           = parent_segments(opened.parsed);
        }

        uint64_t total_chunks = 0;
        for (const auto & source : sources) {
            const uint64_t count = fragment_count(source);
            if (count > pimpl->cfg.max_chunks - total_chunks) {
                result.status = llama_dsv4_segment_status::too_many_chunks;
                return result;
            }
            total_chunks += count;
        }
        if (manifest.fields.size() != sources.size()) {
            result.status = llama_dsv4_segment_status::coverage_mismatch;
            return result;
        }
        uint32_t ordinal = 0;
        for (size_t field_index = 0; field_index < sources.size(); ++field_index) {
            const auto &   source = sources[field_index];
            auto &         field  = manifest.fields[field_index];
            const uint64_t limit  = fragment_payload_limit(source.encoding);
            for (uint64_t offset = 0; offset < source.logical_size; offset += limit) {
                const uint64_t size = std::min<uint64_t>(limit, source.logical_size - offset);
                if (faults.fail_before_segment == ordinal) {
                    result.status = llama_dsv4_segment_status::injected_failure;
                    return result;
                }
                auto chunk                      = make_chunk_bytes(identity, source, offset, size);
                result.peak_codec_scratch_bytes = std::max<uint64_t>(result.peak_codec_scratch_bytes, chunk.size());
                fragment_record fragment;
                fragment.offset         = offset;
                fragment.size           = size;
                fragment.digest         = llama_snapshot_sha256(chunk.data(), chunk.size());
                const auto inherited_it = inherited.find(parent_key{
                    { field.key, offset },
                    size
                });
                fragment.reused         = inherited_it != inherited.end() && inherited_it->second == fragment.digest;
                const auto written = publish_immutable_file(pimpl->chunks_fd, chunk_filename(fragment.digest), chunk,
                                                            CHUNK_HEADER_BYTES + LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES,
                                                            llama_dsv4_segment_status::missing_chunk);
                if (written.status != llama_dsv4_segment_status::ok) {
                    result.status   = written.status;
                    result.os_error = written.os_error;
                    return result;
                }
                if (written.created) {
                    ++result.segments_created;
                }
                if (fragment.reused) {
                    ++result.segments_reused;
                }
                field.fragments.push_back(fragment);
                ++ordinal;
            }
        }
        auto serialized                 = serialize_manifest(manifest);
        result.peak_codec_scratch_bytes = std::max<uint64_t>(result.peak_codec_scratch_bytes, serialized.size());
        if (serialized.size() > pimpl->cfg.max_manifest_bytes) {
            result.status = llama_dsv4_segment_status::manifest_too_large;
            return result;
        }
        const auto digest = llama_snapshot_sha256(serialized.data(), serialized.size());
        if (manifest.has_parent && digest == manifest.parent) {
            result.status = llama_dsv4_segment_status::malformed;
            return result;
        }
        result.manifest = public_manifest(manifest, digest, serialized.size());
        if (faults.fail_before_manifest_publish) {
            result.status = llama_dsv4_segment_status::injected_failure;
            return result;
        }
        const auto manifest_write =
            publish_immutable_file(pimpl->manifests_fd, manifest_filename(digest), serialized,
                                   pimpl->cfg.max_manifest_bytes, llama_dsv4_segment_status::missing_manifest);
        if (manifest_write.status != llama_dsv4_segment_status::ok) {
            result.status   = manifest_write.status;
            result.os_error = manifest_write.os_error;
            return result;
        }
        if (faults.fail_before_ref_publish) {
            result.status = llama_dsv4_segment_status::injected_failure;
            return result;
        }
        const auto ref = make_ref_bytes(digest, identity);
        const auto ref_write =
            publish_ref_file(pimpl->refs_fd, ref_filename(ref_name), ref, faults.fail_after_ref_rename);
        result.status    = ref_write.status;
        result.os_error  = ref_write.os_error;
        result.committed = ref_write.committed;
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_segment_status::resource_exhausted;
        return result;
    } catch (const std::length_error &) {
        result.status = llama_dsv4_segment_status::state_too_large;
        return result;
    }
}

llama_dsv4_segment_open_result llama_dsv4_segment_store::open_manifest(
    const llama_snapshot_digest &       digest,
    const llama_dsv4_segment_identity & expected_identity) const {
    try {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        return open_manifest_locked(*pimpl, digest, &expected_identity);
    } catch (const std::bad_alloc &) {
        llama_dsv4_segment_open_result result;
        result.status = llama_dsv4_segment_status::resource_exhausted;
        return result;
    } catch (const std::length_error &) {
        llama_dsv4_segment_open_result result;
        result.status = llama_dsv4_segment_status::manifest_too_large;
        return result;
    }
}

llama_dsv4_segment_open_result llama_dsv4_segment_store::open_current(
    const std::string &                 ref_name,
    const llama_dsv4_segment_identity & expected_identity) const {
    llama_dsv4_segment_open_result result;
    try {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        if (pimpl->init_status != llama_dsv4_segment_status::ok) {
            result.status   = pimpl->init_status;
            result.os_error = pimpl->init_error;
            return result;
        }
        if (!safe_ref_name(ref_name)) {
            result.status = llama_dsv4_segment_status::invalid_argument;
            return result;
        }
        const auto read = read_private_file(pimpl->refs_fd, ref_filename(ref_name), REF_BYTES,
                                            llama_dsv4_segment_status::no_current_manifest);
        if (read.status != llama_dsv4_segment_status::ok) {
            result.status   = read.status;
            result.os_error = read.os_error;
            return result;
        }
        llama_snapshot_digest       manifest = {};
        llama_dsv4_segment_identity identity;
        result.status = parse_ref_bytes(read.bytes, manifest, identity);
        if (result.status != llama_dsv4_segment_status::ok) {
            return result;
        }
        if (identity != expected_identity) {
            result.status = llama_dsv4_segment_status::identity_mismatch;
            return result;
        }
        return open_manifest_locked(*pimpl, manifest, &expected_identity);
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_segment_status::resource_exhausted;
        return result;
    } catch (const std::length_error &) {
        result.status = llama_dsv4_segment_status::manifest_too_large;
        return result;
    }
}

namespace {

struct field_target {
    field_encoding                   encoding     = field_encoding::bytes;
    llama_snapshot_digest            key          = {};
    uint64_t                         logical_size = 0;
    std::vector<uint8_t> *           bytes        = nullptr;
    std::vector<llama_pos> *         positions    = nullptr;
    std::vector<llama_kv_cell_ext> * extensions   = nullptr;
};

llama_dsv4_logical_tensor_state materialize_tensor_meta(const tensor_meta & meta) {
    llama_dsv4_logical_tensor_state result;
    result.layer_id = meta.layer_id;
    result.type     = meta.type;
    result.ne0      = meta.ne0;
    result.row_size = meta.row_size;
    result.chunks.reserve(meta.chunks.size());
    for (const auto & chunk : meta.chunks) {
        llama_dsv4_logical_row_chunk logical;
        logical.row_begin = chunk.row_begin;
        logical.row_count = chunk.row_count;
        logical.checksum  = chunk.checksum;
        logical.bytes.resize(static_cast<size_t>(chunk.byte_count));
        result.chunks.push_back(std::move(logical));
    }
    return result;
}

std::vector<llama_dsv4_logical_tensor_state> materialize_tensor_list(const std::vector<tensor_meta> & tensors) {
    std::vector<llama_dsv4_logical_tensor_state> result;
    result.reserve(tensors.size());
    for (const auto & tensor : tensors) {
        result.push_back(materialize_tensor_meta(tensor));
    }
    return result;
}

llama_dsv4_logical_component_state materialize_component(const component_meta & meta) {
    llama_dsv4_logical_component_state result;
    result.schema_version = meta.schema_version;
    result.row_begin      = meta.row_begin;
    result.row_end        = meta.row_end;
    result.tensors        = materialize_tensor_list(meta.tensors);
    return result;
}

llama_dsv4_recurrent_sequence_state materialize_recurrent(const recurrent_meta & meta) {
    llama_dsv4_recurrent_sequence_state result;
    result.schema_version = meta.schema_version;
    result.ratio          = meta.ratio;
    result.state_size     = meta.state_size;
    result.n_embd_state   = meta.n_embd_state;
    result.n_rs_seq       = meta.n_rs_seq;
    result.state_identity = meta.state_identity;
    result.kv             = materialize_tensor_list(meta.kv);
    result.score          = materialize_tensor_list(meta.score);
    return result;
}

llama_dsv4_logical_sequence_state materialize_state(const sequence_meta & meta) {
    llama_dsv4_logical_sequence_state result;
    result.schema_version         = meta.schema_version;
    result.identity               = meta.identity;
    result.accepted_frontier      = static_cast<llama_pos>(meta.accepted_frontier);
    result.rollback_index         = meta.rollback_index;
    result.active_rollback_depth  = meta.active_rollback_depth;
    result.n_rs_seq               = meta.n_rs_seq;
    result.fingerprint            = meta.fingerprint;
    result.raw_swa.schema_version = meta.raw_schema_version;
    const auto materialize_plane  = [](const plane_meta & source, llama_kv_iswa_logical_plane_state & destination) {
        destination.schema_version = source.schema_version;
        destination.positions.resize(static_cast<size_t>(source.position_count));
        destination.extensions.resize(static_cast<size_t>(source.extension_count));
        destination.tensor_payload.resize(static_cast<size_t>(source.tensor_bytes));
        destination.checksum = source.checksum;
    };
    materialize_plane(meta.base, result.raw_swa.base);
    materialize_plane(meta.swa, result.raw_swa.swa);
    result.csa           = materialize_component(meta.csa);
    result.hca           = materialize_component(meta.hca);
    result.lid           = materialize_component(meta.lid);
    result.csa_recurrent = materialize_recurrent(meta.csa_recurrent);
    result.hca_recurrent = materialize_recurrent(meta.hca_recurrent);
    result.lid_recurrent = materialize_recurrent(meta.lid_recurrent);
    return result;
}

void add_plane_targets(std::vector<field_target> &         fields,
                       llama_kv_iswa_logical_plane_state & plane,
                       bool                                base,
                       const llama_dsv4_segment_identity & identity) {
    field_target positions;
    positions.encoding = field_encoding::positions;
    field_path position_path;
    position_path.kind     = base ? field_kind::raw_base_positions : field_kind::raw_swa_positions;
    positions.key          = semantic_key(identity, position_path);
    positions.logical_size = static_cast<uint64_t>(plane.positions.size()) * 8;
    positions.positions    = &plane.positions;
    fields.push_back(positions);
    field_target extensions;
    extensions.encoding = field_encoding::extensions;
    field_path extension_path;
    extension_path.kind     = base ? field_kind::raw_base_extensions : field_kind::raw_swa_extensions;
    extensions.key          = semantic_key(identity, extension_path);
    extensions.logical_size = static_cast<uint64_t>(plane.extensions.size()) * 16;
    extensions.extensions   = &plane.extensions;
    fields.push_back(extensions);
    field_target tensor;
    tensor.encoding = field_encoding::bytes;
    field_path tensor_path;
    tensor_path.kind    = base ? field_kind::raw_base_tensor : field_kind::raw_swa_tensor;
    tensor.key          = semantic_key(identity, tensor_path);
    tensor.logical_size = plane.tensor_payload.size();
    tensor.bytes        = &plane.tensor_payload;
    fields.push_back(tensor);
}

void add_tensor_targets(std::vector<field_target> &                    fields,
                        std::vector<llama_dsv4_logical_tensor_state> & tensors,
                        bool                                           recurrent,
                        uint32_t                                       group,
                        uint32_t                                       tensor_class,
                        const llama_dsv4_segment_identity &            identity) {
    for (uint32_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        auto & tensor = tensors[tensor_index];
        for (auto & chunk : tensor.chunks) {
            field_target field;
            field.encoding = field_encoding::bytes;
            field_path path;
            path.kind           = recurrent ? field_kind::recurrent_rows : field_kind::compressed_rows;
            path.group          = group;
            path.tensor_class   = tensor_class;
            path.tensor_ordinal = tensor_index;
            path.layer_id       = tensor.layer_id;
            path.type           = tensor.type;
            path.ne0            = tensor.ne0;
            path.row_size       = tensor.row_size;
            path.row_begin      = chunk.row_begin;
            path.row_count      = chunk.row_count;
            field.key           = semantic_key(identity, path);
            field.logical_size  = chunk.bytes.size();
            field.bytes         = &chunk.bytes;
            fields.push_back(field);
        }
    }
}

std::vector<field_target> make_targets(llama_dsv4_logical_sequence_state & state,
                                       const llama_dsv4_segment_identity & identity) {
    std::vector<field_target> fields;
    add_plane_targets(fields, state.raw_swa.base, true, identity);
    add_plane_targets(fields, state.raw_swa.swa, false, identity);
    add_tensor_targets(fields, state.csa.tensors, false, 0, 0, identity);
    add_tensor_targets(fields, state.hca.tensors, false, 1, 0, identity);
    add_tensor_targets(fields, state.lid.tensors, false, 2, 0, identity);
    add_tensor_targets(fields, state.csa_recurrent.kv, true, 0, 0, identity);
    add_tensor_targets(fields, state.csa_recurrent.score, true, 0, 1, identity);
    add_tensor_targets(fields, state.hca_recurrent.kv, true, 1, 0, identity);
    add_tensor_targets(fields, state.hca_recurrent.score, true, 1, 1, identity);
    add_tensor_targets(fields, state.lid_recurrent.kv, true, 2, 0, identity);
    add_tensor_targets(fields, state.lid_recurrent.score, true, 2, 1, identity);
    return fields;
}

llama_dsv4_segment_status fill_target(field_target & target, uint64_t offset, const uint8_t * payload, uint64_t size) {
    if (offset > target.logical_size || size > target.logical_size - offset) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    if (target.encoding == field_encoding::bytes) {
        if (size != 0) {
            std::memcpy(target.bytes->data() + offset, payload, static_cast<size_t>(size));
        }
        return llama_dsv4_segment_status::ok;
    }
    const uint64_t unit = target.encoding == field_encoding::positions ? 8 : 16;
    if (offset % unit != 0 || size % unit != 0) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    const auto decode_i64 = [](const uint8_t * data) {
        uint64_t value = 0;
        for (uint32_t index = 0; index < 8; ++index) {
            value |= static_cast<uint64_t>(data[index]) << (8 * index);
        }
        return static_cast<int64_t>(value);
    };
    if (target.encoding == field_encoding::positions) {
        const size_t first = static_cast<size_t>(offset / 8);
        for (size_t index = 0; index < size / 8; ++index) {
            const int64_t position = decode_i64(payload + 8 * index);
            if (position < std::numeric_limits<llama_pos>::min() || position > std::numeric_limits<llama_pos>::max()) {
                return llama_dsv4_segment_status::coverage_mismatch;
            }
            target.positions->at(first + index) = static_cast<llama_pos>(position);
        }
    } else {
        const size_t first = static_cast<size_t>(offset / 16);
        for (size_t index = 0; index < size / 16; ++index) {
            const int64_t x = decode_i64(payload + 16 * index);
            const int64_t y = decode_i64(payload + 16 * index + 8);
            if (x < std::numeric_limits<llama_pos>::min() || x > std::numeric_limits<llama_pos>::max() ||
                y < std::numeric_limits<llama_pos>::min() || y > std::numeric_limits<llama_pos>::max()) {
                return llama_dsv4_segment_status::coverage_mismatch;
            }
            target.extensions->at(first + index) = {
                static_cast<llama_pos>(x),
                static_cast<llama_pos>(y),
            };
        }
    }
    return llama_dsv4_segment_status::ok;
}

llama_dsv4_segment_status validate_all_chunks_locked(const llama_dsv4_segment_store_impl & store,
                                                     const parsed_manifest &               manifest,
                                                     uint64_t &                            peak_scratch,
                                                     int &                                 os_error,
                                                     llama_dsv4_logical_sequence_state *   destination,
                                                     uint32_t *                            segments_read = nullptr) {
    std::vector<field_target> targets;
    if (destination != nullptr) {
        targets = make_targets(*destination, manifest.identity);
        if (targets.size() != manifest.fields.size()) {
            return llama_dsv4_segment_status::coverage_mismatch;
        }
    }
    for (size_t field_index = 0; field_index < manifest.fields.size(); ++field_index) {
        const auto & field = manifest.fields[field_index];
        if (destination != nullptr &&
            (targets[field_index].key != field.key || targets[field_index].logical_size != field.logical_size ||
             targets[field_index].encoding != field.encoding)) {
            return llama_dsv4_segment_status::coverage_mismatch;
        }
        for (const auto & fragment : field.fragments) {
            const auto read = read_private_file(store.chunks_fd, chunk_filename(fragment.digest),
                                                CHUNK_HEADER_BYTES + LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES,
                                                llama_dsv4_segment_status::missing_chunk);
            peak_scratch    = std::max<uint64_t>(peak_scratch, read.bytes.size());
            if (read.status != llama_dsv4_segment_status::ok) {
                os_error = read.os_error;
                return read.status;
            }
            if (segments_read != nullptr) {
                ++*segments_read;
            }
            auto status = validate_chunk_bytes(read.bytes, manifest.identity, field, fragment);
            if (status != llama_dsv4_segment_status::ok) {
                return status;
            }
            if (destination != nullptr) {
                status = fill_target(targets[field_index], fragment.offset, read.bytes.data() + CHUNK_HEADER_BYTES,
                                     fragment.size);
                if (status != llama_dsv4_segment_status::ok) {
                    return status;
                }
            }
        }
    }
    return llama_dsv4_segment_status::ok;
}

bool tensor_geometry_matches(const tensor_meta & actual, const llama_dsv4_logical_tensor_state & expected) {
    return actual.layer_id == expected.layer_id && actual.type == expected.type && actual.ne0 == expected.ne0 &&
           actual.row_size == expected.row_size;
}

bool tensor_list_geometry_matches(const std::vector<tensor_meta> &                     actual,
                                  const std::vector<llama_dsv4_logical_tensor_state> & expected) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!tensor_geometry_matches(actual[index], expected[index])) {
            return false;
        }
    }
    return true;
}

std::vector<llama_dsv4_logical_tensor_state> cache_tensor_geometry(const llama_kv_cache * cache) {
    std::vector<llama_dsv4_logical_tensor_state> result;
    for (uint32_t layer : cache->get_layer_ids()) {
        const ggml_tensor *             tensor = cache->get_k_storage(static_cast<int32_t>(layer));
        llama_dsv4_logical_tensor_state geometry;
        geometry.layer_id = layer;
        geometry.type     = static_cast<int32_t>(tensor->type);
        geometry.ne0      = tensor->ne[0];
        geometry.row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        result.push_back(std::move(geometry));
    }
    return result;
}

bool raw_payload_size(const llama_kv_cache * cache, uint64_t rows, uint64_t & result) {
    result = 8;  // v_trans + layer count
    for (uint32_t layer : cache->get_layer_ids()) {
        const ggml_tensor * tensor    = cache->get_k_storage(static_cast<int32_t>(layer));
        uint64_t            row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
        uint64_t            payload   = 0;
        if (!checked_multiply(rows, row_bytes, payload) || !checked_add(result, 12) || !checked_add(result, payload)) {
            return false;
        }
    }
    return true;
}

llama_dsv4_segment_status validate_cache_geometry(const sequence_meta &               meta,
                                                  const llama_dsv4_segment_identity & identity,
                                                  const llama_kv_cache_dsv4 &         cache) {
    if (!cache.is_aggregate_compressed()) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    if (cache.get_logical_state_identity() != identity.geometry_identity ||
        meta.identity != identity.geometry_identity) {
        return llama_dsv4_segment_status::identity_mismatch;
    }
    if (meta.n_rs_seq != cache.get_n_rs_seq() || meta.active_rollback_depth != cache.get_active_rs_depth() ||
        meta.rollback_index > meta.active_rollback_depth) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    const uint64_t accepted = meta.accepted_frontier < 0 ? 0 : static_cast<uint64_t>(meta.accepted_frontier) + 1;
    if (accepted / C4_RATIO > cache.get_c4_logical_rows() || accepted / HCA_RATIO > cache.get_hca_logical_rows()) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    const auto csa_geometry = cache_tensor_geometry(cache.get_csa());
    const auto hca_geometry = cache_tensor_geometry(cache.get_hca());
    const auto lid_geometry = cache_tensor_geometry(cache.get_lid());
    if (!tensor_list_geometry_matches(meta.csa.tensors, csa_geometry) ||
        !tensor_list_geometry_matches(meta.hca.tensors, hca_geometry) ||
        !tensor_list_geometry_matches(meta.lid.tensors, lid_geometry)) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    const auto validate_recurrent = [](const recurrent_meta & actual, const llama_dsv4_comp_state * expected) {
        return actual.ratio == expected->get_ratio() && actual.state_size == expected->get_state_size() &&
               actual.n_rs_seq == expected->get_n_rs_seq() && actual.state_identity == expected->state_identity() &&
               tensor_list_geometry_matches(actual.kv, expected->logical_tensor_geometry(false)) &&
               tensor_list_geometry_matches(actual.score, expected->logical_tensor_geometry(true));
    };
    if (!validate_recurrent(meta.csa_recurrent, cache.get_csa_state()) ||
        !validate_recurrent(meta.hca_recurrent, cache.get_hca_state()) ||
        !validate_recurrent(meta.lid_recurrent, cache.get_lid_state())) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    auto * raw = cache.get_raw();
    if (meta.base.position_count > raw->get_base()->get_size() ||
        meta.swa.position_count > raw->get_swa()->get_size()) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    uint64_t base_bytes = 0;
    uint64_t swa_bytes  = 0;
    if (!raw_payload_size(raw->get_base(), meta.base.position_count, base_bytes) ||
        !raw_payload_size(raw->get_swa(), meta.swa.position_count, swa_bytes) || base_bytes != meta.base.tensor_bytes ||
        swa_bytes != meta.swa.tensor_bytes) {
        return llama_dsv4_segment_status::coverage_mismatch;
    }
    return llama_dsv4_segment_status::ok;
}

}  // namespace

llama_dsv4_segment_status llama_dsv4_segment_store::validate(const llama_dsv4_segment_manifest & manifest,
                                                             int *                               os_error) const {
    try {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        const auto                  opened = open_parsed_manifest_locked(*pimpl, manifest.digest, &manifest.identity);
        if (opened.status != llama_dsv4_segment_status::ok) {
            if (os_error != nullptr) {
                *os_error = opened.os_error;
            }
            return opened.status;
        }
        uint64_t   peak   = 0;
        int        error  = 0;
        const auto status = validate_all_chunks_locked(*pimpl, opened.parsed, peak, error, nullptr);
        if (os_error != nullptr) {
            *os_error = error;
        }
        return status;
    } catch (const std::bad_alloc &) {
        return llama_dsv4_segment_status::resource_exhausted;
    } catch (const std::length_error &) {
        return llama_dsv4_segment_status::manifest_too_large;
    }
}

llama_dsv4_segment_load_result llama_dsv4_segment_store::load(
    const llama_dsv4_segment_manifest & manifest,
    const llama_dsv4_segment_identity & expected_identity) const {
    llama_dsv4_segment_load_result result;
    try {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        const auto                  opened = open_parsed_manifest_locked(*pimpl, manifest.digest, &expected_identity);
        if (opened.status != llama_dsv4_segment_status::ok) {
            result.status   = opened.status;
            result.os_error = opened.os_error;
            return result;
        }
        result.peak_codec_scratch_bytes = opened.encoded_bytes;
        const auto & parsed             = opened.parsed;
        try {
            result.state = materialize_state(parsed.meta);
        } catch (const std::bad_alloc &) {
            result.status = llama_dsv4_segment_status::resource_exhausted;
            return result;
        } catch (const std::length_error &) {
            result.status = llama_dsv4_segment_status::state_too_large;
            return result;
        }
        result.status = validate_all_chunks_locked(*pimpl, parsed, result.peak_codec_scratch_bytes, result.os_error,
                                                   &result.state, &result.segments_read);
        if (result.status != llama_dsv4_segment_status::ok) {
            result.state = {};
            return result;
        }
        sequence_meta             verified_meta;
        std::vector<uint8_t>      verified_envelope;
        std::vector<field_source> verified_sources;
        uint64_t                  verified_payload = 0;
        result.status = validate_state(result.state, expected_identity, pimpl->cfg.max_state_bytes, verified_meta,
                                       verified_envelope, verified_sources, verified_payload);
        if (result.status != llama_dsv4_segment_status::ok || verified_envelope != parsed.envelope ||
            verified_payload != parsed.total_payload) {
            if (result.status == llama_dsv4_segment_status::ok) {
                result.status = llama_dsv4_segment_status::checksum_mismatch;
            }
            result.state = {};
            return result;
        }
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_segment_status::resource_exhausted;
        result.state  = {};
        return result;
    } catch (const std::length_error &) {
        result.status = llama_dsv4_segment_status::state_too_large;
        result.state  = {};
        return result;
    }
}

llama_dsv4_segment_restore_result llama_dsv4_segment_store::restore(
    const llama_dsv4_segment_manifest & manifest,
    const llama_dsv4_segment_identity & expected_identity,
    llama_kv_cache_dsv4 &               cache,
    llama_seq_id                        destination) const {
    llama_dsv4_segment_restore_result result;
    try {
        {
            std::lock_guard<std::mutex> lock(pimpl->mutex);
            const auto opened = open_parsed_manifest_locked(*pimpl, manifest.digest, &expected_identity);
            if (opened.status != llama_dsv4_segment_status::ok) {
                result.status   = opened.status;
                result.os_error = opened.os_error;
                return result;
            }
            result.status = validate_cache_geometry(opened.parsed.meta, expected_identity, cache);
            if (result.status != llama_dsv4_segment_status::ok) {
                return result;
            }
        }
        auto loaded                     = load(manifest, expected_identity);
        result.status                   = loaded.status;
        result.os_error                 = loaded.os_error;
        result.peak_codec_scratch_bytes = loaded.peak_codec_scratch_bytes;
        result.segments_read            = loaded.segments_read;
        if (loaded.status != llama_dsv4_segment_status::ok) {
            return result;
        }
        auto quote           = cache.quote_logical_import(destination, loaded.state);
        result.import_status = quote.status;
        if (quote.status != llama_dsv4_logical_state_status::ok) {
            result.status = llama_dsv4_segment_status::import_rejected;
            return result;
        }
        result.import_status = cache.import_logical_sequence(quote, loaded.state);
        result.status        = result.import_status == llama_dsv4_logical_state_status::ok ?
                                   llama_dsv4_segment_status::ok :
                                   llama_dsv4_segment_status::import_rejected;
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_segment_status::resource_exhausted;
        return result;
    } catch (const std::length_error &) {
        result.status = llama_dsv4_segment_status::state_too_large;
        return result;
    } catch (...) {
        result.status        = llama_dsv4_segment_status::import_rejected;
        result.import_status = llama_dsv4_logical_state_status::backend_error;
        return result;
    }
}

namespace {

struct scanned_entry {
    std::string name;
    struct stat status = {};
};

struct scan_result {
    llama_dsv4_segment_status  status = llama_dsv4_segment_status::ok;
    std::vector<scanned_entry> entries;
    int                        os_error = 0;
};

scan_result scan_directory(int descriptor) {
    scan_result result;
    // dup() would share the directory stream offset with the held authority
    // fd, making a second reconciliation silently start at EOF. Open an
    // independent description descriptor-relatively instead.
    const int   duplicate = ::openat(descriptor, ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    if (duplicate < 0) {
        result.status   = llama_dsv4_segment_status::io_error;
        result.os_error = errno;
        return result;
    }
    DIR * directory = ::fdopendir(duplicate);
    if (directory == nullptr) {
        result.status   = llama_dsv4_segment_status::io_error;
        result.os_error = errno;
        ::close(duplicate);
        return result;
    }
    errno = 0;
    while (dirent * entry = ::readdir(directory)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        struct stat status = {};
        if (::fstatat(descriptor, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
            result.status   = llama_dsv4_segment_status::io_error;
            result.os_error = errno;
            break;
        }
        result.entries.push_back({ name, status });
        errno = 0;
    }
    if (result.status == llama_dsv4_segment_status::ok && errno != 0) {
        result.status   = llama_dsv4_segment_status::io_error;
        result.os_error = errno;
    }
    if (::closedir(directory) != 0 && result.status == llama_dsv4_segment_status::ok) {
        result.status   = llama_dsv4_segment_status::io_error;
        result.os_error = errno;
    }
    return result;
}

bool lowercase_hex(char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
}

bool digest_filename(const std::string & name, const char * suffix, llama_snapshot_digest & digest) {
    const size_t suffix_size = std::strlen(suffix);
    if (name.size() != 64 + suffix_size || name.compare(64, suffix_size, suffix) != 0 ||
        !std::all_of(name.begin(), name.begin() + 64, lowercase_hex)) {
        return false;
    }
    for (size_t index = 0; index < digest.size(); ++index) {
        const auto nibble = [](char value) -> uint8_t {
            return value <= '9' ? static_cast<uint8_t>(value - '0') : static_cast<uint8_t>(value - 'a' + 10);
        };
        digest[index] = static_cast<uint8_t>((nibble(name[2 * index]) << 4) | nibble(name[2 * index + 1]));
    }
    return true;
}

bool recognized_temp(const std::string & name) {
    if (name.rfind(".tmp-", 0) != 0 || name.size() > 128) {
        return false;
    }
    return std::all_of(name.begin() + 5, name.end(), [](unsigned char value) {
        return (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9') || value == '-';
    });
}

llama_dsv4_segment_status remove_checked(int                   directory,
                                         const scanned_entry & entry,
                                         bool                  dry_run,
                                         bool &                removed,
                                         int &                 os_error) {
    removed = false;
    if (!private_regular_file(entry.status)) {
        os_error = EACCES;
        return llama_dsv4_segment_status::path_security;
    }
    struct stat current = {};
    if (::fstatat(directory, entry.name.c_str(), &current, AT_SYMLINK_NOFOLLOW) != 0) {
        if (errno == ENOENT) {
            return llama_dsv4_segment_status::ok;
        }
        os_error = errno;
        return errno_status(errno);
    }
    if (!private_regular_file(current) || !same_file(current, entry.status)) {
        os_error = EACCES;
        return llama_dsv4_segment_status::path_security;
    }
    if (!dry_run && ::unlinkat(directory, entry.name.c_str(), 0) != 0) {
        os_error = errno;
        return errno_status(errno);
    }
    removed = true;
    return llama_dsv4_segment_status::ok;
}

}  // namespace

llama_dsv4_segment_reconcile_result llama_dsv4_segment_store::reconcile(
    const std::vector<llama_snapshot_digest> & pinned_manifests,
    bool                                       dry_run) {
    llama_dsv4_segment_reconcile_result result;
    try {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        if (pimpl->init_status != llama_dsv4_segment_status::ok) {
            result.status   = pimpl->init_status;
            result.os_error = pimpl->init_error;
            return result;
        }
        std::set<llama_snapshot_digest> live_manifests(pinned_manifests.begin(), pinned_manifests.end());
        if (live_manifests.count({}) != 0) {
            result.status = llama_dsv4_segment_status::invalid_argument;
            return result;
        }

        const auto refs = scan_directory(pimpl->refs_fd);
        if (refs.status != llama_dsv4_segment_status::ok) {
            result.status   = refs.status;
            result.os_error = refs.os_error;
            return result;
        }
        std::vector<scanned_entry> ref_temps;
        for (const auto & entry : refs.entries) {
            if (recognized_temp(entry.name)) {
                if (!private_regular_file(entry.status)) {
                    result.status   = llama_dsv4_segment_status::path_security;
                    result.os_error = EACCES;
                    return result;
                }
                ref_temps.push_back(entry);
                continue;
            }
            if (entry.name.size() <= 4 || entry.name.compare(entry.name.size() - 4, 4, ".ref") != 0) {
                continue;
            }
            const std::string logical = entry.name.substr(0, entry.name.size() - 4);
            if (!safe_ref_name(logical) || !private_regular_file(entry.status)) {
                result.status   = llama_dsv4_segment_status::path_security;
                result.os_error = EACCES;
                return result;
            }
            const auto read = read_private_file(pimpl->refs_fd, entry.name, REF_BYTES,
                                                llama_dsv4_segment_status::no_current_manifest);
            if (read.status != llama_dsv4_segment_status::ok) {
                result.status   = read.status;
                result.os_error = read.os_error;
                return result;
            }
            llama_snapshot_digest       digest = {};
            llama_dsv4_segment_identity identity;
            result.status = parse_ref_bytes(read.bytes, digest, identity);
            if (result.status != llama_dsv4_segment_status::ok) {
                return result;
            }
            live_manifests.insert(digest);
        }

        std::set<llama_snapshot_digest> live_chunks;
        for (const auto & digest : live_manifests) {
            const auto opened = open_parsed_manifest_locked(*pimpl, digest, nullptr);
            if (opened.status != llama_dsv4_segment_status::ok) {
                result.status   = opened.status;
                result.os_error = opened.os_error;
                return result;
            }
            ++result.manifests_scanned;
            uint64_t peak = 0;
            result.status = validate_all_chunks_locked(*pimpl, opened.parsed, peak, result.os_error, nullptr);
            if (result.status != llama_dsv4_segment_status::ok) {
                return result;
            }
            for (const auto & field : opened.parsed.fields) {
                for (const auto & fragment : field.fragments) {
                    live_chunks.insert(fragment.digest);
                }
            }
        }

        const auto manifests = scan_directory(pimpl->manifests_fd);
        const auto chunks    = scan_directory(pimpl->chunks_fd);
        if (manifests.status != llama_dsv4_segment_status::ok || chunks.status != llama_dsv4_segment_status::ok) {
            const auto & failed = manifests.status != llama_dsv4_segment_status::ok ? manifests : chunks;
            result.status       = failed.status;
            result.os_error     = failed.os_error;
            return result;
        }
        std::vector<scanned_entry> manifest_remove;
        std::vector<scanned_entry> chunk_remove;
        std::vector<scanned_entry> manifest_temps;
        std::vector<scanned_entry> chunk_temps;
        for (const auto & entry : manifests.entries) {
            if (recognized_temp(entry.name)) {
                if (!private_regular_file(entry.status)) {
                    result.status   = llama_dsv4_segment_status::path_security;
                    result.os_error = EACCES;
                    return result;
                }
                manifest_temps.push_back(entry);
                continue;
            }
            llama_snapshot_digest digest = {};
            if (!digest_filename(entry.name, ".manifest", digest)) {
                continue;
            }
            if (!private_regular_file(entry.status)) {
                result.status   = llama_dsv4_segment_status::path_security;
                result.os_error = EACCES;
                return result;
            }
            ++result.manifests_scanned;
            if (live_manifests.count(digest) == 0) {
                manifest_remove.push_back(entry);
            }
        }
        for (const auto & entry : chunks.entries) {
            if (recognized_temp(entry.name)) {
                if (!private_regular_file(entry.status)) {
                    result.status   = llama_dsv4_segment_status::path_security;
                    result.os_error = EACCES;
                    return result;
                }
                chunk_temps.push_back(entry);
                continue;
            }
            llama_snapshot_digest digest = {};
            if (!digest_filename(entry.name, ".chunk", digest)) {
                continue;
            }
            if (!private_regular_file(entry.status)) {
                result.status   = llama_dsv4_segment_status::path_security;
                result.os_error = EACCES;
                return result;
            }
            ++result.segments_scanned;
            if (live_chunks.count(digest) == 0) {
                chunk_remove.push_back(entry);
            }
        }
        bool       changed_refs      = false;
        bool       changed_manifests = false;
        bool       changed_chunks    = false;
        const auto remove_group      = [&](int directory, const std::vector<scanned_entry> & entries, uint32_t & count,
                                      bool & changed) -> llama_dsv4_segment_status {
            for (const auto & entry : entries) {
                bool       removed = false;
                const auto status  = remove_checked(directory, entry, dry_run, removed, result.os_error);
                if (status != llama_dsv4_segment_status::ok) {
                    return status;
                }
                if (removed) {
                    ++count;
                    changed = changed || !dry_run;
                }
            }
            return llama_dsv4_segment_status::ok;
        };
        result.status = remove_group(pimpl->manifests_fd, manifest_remove, result.manifests_removed, changed_manifests);
        if (result.status != llama_dsv4_segment_status::ok) {
            return result;
        }
        result.status = remove_group(pimpl->chunks_fd, chunk_remove, result.segments_removed, changed_chunks);
        if (result.status != llama_dsv4_segment_status::ok) {
            return result;
        }
        result.status = remove_group(pimpl->refs_fd, ref_temps, result.temporary_files_removed, changed_refs);
        if (result.status != llama_dsv4_segment_status::ok) {
            return result;
        }
        result.status =
            remove_group(pimpl->manifests_fd, manifest_temps, result.temporary_files_removed, changed_manifests);
        if (result.status != llama_dsv4_segment_status::ok) {
            return result;
        }
        result.status = remove_group(pimpl->chunks_fd, chunk_temps, result.temporary_files_removed, changed_chunks);
        if (result.status != llama_dsv4_segment_status::ok) {
            return result;
        }
        int fence_error = 0;
        if (!dry_run && changed_refs) {
            record_first_errno(::fsync(pimpl->refs_fd), fence_error);
        }
        if (!dry_run && changed_manifests) {
            record_first_errno(::fsync(pimpl->manifests_fd), fence_error);
        }
        if (!dry_run && changed_chunks) {
            record_first_errno(::fsync(pimpl->chunks_fd), fence_error);
        }
        if (!dry_run && (changed_refs || changed_manifests || changed_chunks)) {
            record_first_errno(::fsync(pimpl->root_fd), fence_error);
        }
        if (fence_error != 0) {
            result.status   = llama_dsv4_segment_status::commit_uncertain;
            result.os_error = fence_error;
            return result;
        }
        result.status = llama_dsv4_segment_status::ok;
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_segment_status::resource_exhausted;
        return result;
    } catch (const std::length_error &) {
        result.status = llama_dsv4_segment_status::manifest_too_large;
        return result;
    }
}
