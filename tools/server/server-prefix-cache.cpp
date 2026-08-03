#include "server-prefix-cache.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>

namespace server_prefix_cache {

namespace {

using token_block = std::array<int32_t, SERVER_PREFIX_BLOCK_TOKENS>;

size_t hash_mix(size_t hash, uint64_t value) {
    // Stable enough for in-process buckets. Exact equality below remains the
    // compatibility authority and is intentionally exercised under collisions.
    hash ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

size_t hash_bytes(size_t hash, const uint8_t * bytes, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        hash = hash_mix(hash, bytes[i]);
    }
    return hash;
}

token_block make_block(const std::vector<int32_t> & tokens, size_t offset) {
    token_block result;
    std::copy_n(tokens.data() + offset, result.size(), result.data());
    return result;
}

bool valid_entry(const server_prefix_entry & entry) {
    return entry.handle_id != 0 && entry.generation != 0 && std::isfinite(entry.saved_prefill_ms) &&
           entry.saved_prefill_ms >= 0.0;
}

bool digest_present(const server_prefix_digest & digest) {
    return std::any_of(digest.begin(), digest.end(), [](uint8_t byte) { return byte != 0; });
}

bool valid_identity(const server_prefix_identity & identity) {
    // An all-zero template, draft layout, or LoRA digest explicitly represents
    // an absent optional artifact. The remaining inputs are mandatory.
    return !identity.architecture.empty() && digest_present(identity.model) && digest_present(identity.tokenizer) &&
           digest_present(identity.runtime_build) && digest_present(identity.target_kv_layout) &&
           digest_present(identity.rope) && identity.context_size != 0 && identity.dsv4_state_version != 0;
}

}  // namespace

bool server_prefix_identity::operator==(const server_prefix_identity & other) const {
    return architecture == other.architecture && model == other.model && tokenizer == other.tokenizer &&
           chat_template == other.chat_template && runtime_build == other.runtime_build &&
           target_kv_layout == other.target_kv_layout && draft_kv_layout == other.draft_kv_layout &&
           rope == other.rope && lora == other.lora && context_size == other.context_size &&
           dsv4_state_version == other.dsv4_state_version;
}

bool server_prefix_entry::operator==(const server_prefix_entry & other) const {
    return handle_id == other.handle_id && generation == other.generation && unique_bytes == other.unique_bytes &&
           saved_prefill_ms == other.saved_prefill_ms;
}

struct identity_hash {
    bool constant = false;

    size_t operator()(const server_prefix_identity & identity) const {
        if (constant) {
            return 0;
        }
        size_t hash = std::hash<std::string>{}(identity.architecture);
        hash        = hash_bytes(hash, identity.model.data(), identity.model.size());
        hash        = hash_bytes(hash, identity.tokenizer.data(), identity.tokenizer.size());
        hash        = hash_bytes(hash, identity.chat_template.data(), identity.chat_template.size());
        hash        = hash_bytes(hash, identity.runtime_build.data(), identity.runtime_build.size());
        hash        = hash_bytes(hash, identity.target_kv_layout.data(), identity.target_kv_layout.size());
        hash        = hash_bytes(hash, identity.draft_kv_layout.data(), identity.draft_kv_layout.size());
        hash        = hash_bytes(hash, identity.rope.data(), identity.rope.size());
        hash        = hash_bytes(hash, identity.lora.data(), identity.lora.size());
        hash        = hash_mix(hash, identity.context_size);
        return hash_mix(hash, identity.dsv4_state_version);
    }
};

struct token_block_hash {
    bool constant = false;

    size_t operator()(const token_block & block) const {
        if (constant) {
            return 0;
        }
        size_t hash = 0;
        for (int32_t token : block) {
            hash = hash_mix(hash, static_cast<uint32_t>(token));
        }
        return hash;
    }
};

struct radix_node {
    explicit radix_node(bool force_collisions) : children(0, token_block_hash{ force_collisions }) {}

    std::unordered_map<token_block, std::unique_ptr<radix_node>, token_block_hash> children;
    std::optional<server_prefix_entry>                                             entry;
};

struct server_prefix_radix::impl {
    explicit impl(server_prefix_radix_config config) :
        cfg(config),
        roots(0, identity_hash{ config.force_hash_collisions }) {}

    server_prefix_radix_config                                                             cfg;
    std::unordered_map<server_prefix_identity, std::unique_ptr<radix_node>, identity_hash> roots;
    size_t                                                                                 node_count  = 0;
    size_t                                                                                 entry_count = 0;
};

server_prefix_radix::server_prefix_radix(server_prefix_radix_config config) : data(std::make_unique<impl>(config)) {}

server_prefix_radix::~server_prefix_radix() = default;

server_prefix_insert_status server_prefix_radix::insert(const server_prefix_identity & identity,
                                                        const std::vector<int32_t> &   tokens,
                                                        const server_prefix_entry &    entry) {
    auto & cfg         = data->cfg;
    auto & roots       = data->roots;
    auto & node_count  = data->node_count;
    auto & entry_count = data->entry_count;
    if (!valid_identity(identity) || !valid_entry(entry) || tokens.empty() ||
        tokens.size() % SERVER_PREFIX_BLOCK_TOKENS != 0 || tokens.size() > identity.context_size) {
        return server_prefix_insert_status::invalid_argument;
    }

    const auto         root_found    = roots.find(identity);
    const radix_node * cursor        = root_found == roots.end() ? nullptr : root_found->second.get();
    size_t             missing_nodes = 0;
    for (size_t offset = 0; offset < tokens.size(); offset += SERVER_PREFIX_BLOCK_TOKENS) {
        if (cursor == nullptr) {
            ++missing_nodes;
            continue;
        }
        const auto child = cursor->children.find(make_block(tokens, offset));
        if (child == cursor->children.end()) {
            cursor = nullptr;
            ++missing_nodes;
        } else {
            cursor = child->second.get();
        }
    }

    if (cursor != nullptr && cursor->entry.has_value()) {
        return cursor->entry.value() == entry ? server_prefix_insert_status::already_present :
                                                server_prefix_insert_status::conflict;
    }
    if (missing_nodes > cfg.max_nodes - std::min(cfg.max_nodes, node_count) || entry_count >= cfg.max_entries) {
        return server_prefix_insert_status::capacity;
    }

    auto         root           = roots.try_emplace(identity, std::make_unique<radix_node>(cfg.force_hash_collisions));
    radix_node * mutable_cursor = root.first->second.get();
    for (size_t offset = 0; offset < tokens.size(); offset += SERVER_PREFIX_BLOCK_TOKENS) {
        const token_block block = make_block(tokens, offset);
        auto              child = mutable_cursor->children.find(block);
        if (child == mutable_cursor->children.end()) {
            auto inserted =
                mutable_cursor->children.emplace(block, std::make_unique<radix_node>(cfg.force_hash_collisions));
            child = inserted.first;
            ++node_count;
        }
        mutable_cursor = child->second.get();
    }
    mutable_cursor->entry = entry;
    ++entry_count;
    return server_prefix_insert_status::inserted;
}

std::optional<server_prefix_match> server_prefix_radix::lookup(const server_prefix_identity & identity,
                                                               const std::vector<int32_t> &   tokens) const {
    const auto & roots = data->roots;
    const auto   root  = roots.find(identity);
    if (root == roots.end()) {
        return std::nullopt;
    }

    const radix_node *                 cursor = root->second.get();
    std::optional<server_prefix_match> result;
    const size_t                       complete_tokens = tokens.size() - tokens.size() % SERVER_PREFIX_BLOCK_TOKENS;
    for (size_t offset = 0; offset < complete_tokens; offset += SERVER_PREFIX_BLOCK_TOKENS) {
        const auto child = cursor->children.find(make_block(tokens, offset));
        if (child == cursor->children.end()) {
            break;
        }
        cursor = child->second.get();
        if (cursor->entry.has_value()) {
            result = server_prefix_match{ cursor->entry.value(), offset + SERVER_PREFIX_BLOCK_TOKENS };
        }
    }
    return result;
}

bool server_prefix_radix::erase(const server_prefix_identity & identity,
                                const std::vector<int32_t> &   tokens,
                                uint64_t                       expected_handle_id,
                                uint64_t                       expected_generation) {
    auto & roots       = data->roots;
    auto & node_count  = data->node_count;
    auto & entry_count = data->entry_count;
    if (tokens.empty() || tokens.size() % SERVER_PREFIX_BLOCK_TOKENS != 0) {
        return false;
    }
    const auto root = roots.find(identity);
    if (root == roots.end()) {
        return false;
    }

    std::vector<std::pair<radix_node *, token_block>> path;
    radix_node *                                      cursor = root->second.get();
    for (size_t offset = 0; offset < tokens.size(); offset += SERVER_PREFIX_BLOCK_TOKENS) {
        const token_block block = make_block(tokens, offset);
        const auto        child = cursor->children.find(block);
        if (child == cursor->children.end()) {
            return false;
        }
        path.emplace_back(cursor, block);
        cursor = child->second.get();
    }
    if (!cursor->entry.has_value() || cursor->entry->handle_id != expected_handle_id ||
        cursor->entry->generation != expected_generation) {
        return false;
    }
    cursor->entry.reset();
    --entry_count;

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        radix_node * parent = it->first;
        const auto   child  = parent->children.find(it->second);
        if (child == parent->children.end() || child->second->entry.has_value() || !child->second->children.empty()) {
            break;
        }
        parent->children.erase(child);
        --node_count;
    }
    if (root->second->children.empty() && !root->second->entry.has_value()) {
        roots.erase(root);
    }
    return true;
}

void server_prefix_radix::clear() {
    data->roots.clear();
    data->node_count  = 0;
    data->entry_count = 0;
}

server_prefix_radix_stats server_prefix_radix::stats() const {
    return { data->roots.size(), data->node_count, data->entry_count };
}

}  // namespace server_prefix_cache
