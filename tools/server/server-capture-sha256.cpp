#include "server-capture-sha256.h"

#include <algorithm>
#include <cstring>

namespace server_capture {
namespace capture_sha256 {
namespace {

struct sha256_context {
    uint32_t state[8]      = {};
    uint64_t bit_length    = 0;
    uint8_t  buffer[64]    = {};
    size_t   buffer_length = 0;
};

constexpr std::array<uint32_t, 64> SHA256_K = {
    {
     0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
     0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
     0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
     0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
     0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
     0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
     0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
     0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
     }
};

uint32_t rotate_right(uint32_t value, unsigned shift) noexcept {
    return (value >> shift) | (value << (32U - shift));
}

void compress(uint32_t state[8], const uint8_t block[64]) noexcept {
    uint32_t words[64] = {};
    for (size_t index = 0; index < 16; ++index) {
        words[index] = (static_cast<uint32_t>(block[index * 4]) << 24U) |
                       (static_cast<uint32_t>(block[index * 4 + 1]) << 16U) |
                       (static_cast<uint32_t>(block[index * 4 + 2]) << 8U) |
                       static_cast<uint32_t>(block[index * 4 + 3]);
    }
    for (size_t index = 16; index < 64; ++index) {
        const uint32_t s0 = rotate_right(words[index - 15], 7U) ^ rotate_right(words[index - 15], 18U) ^
                            (words[index - 15] >> 3U);
        const uint32_t s1 = rotate_right(words[index - 2], 17U) ^ rotate_right(words[index - 2], 19U) ^
                            (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];
    for (size_t index = 0; index < 64; ++index) {
        const uint32_t sum1     = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
        const uint32_t choose   = (e & f) ^ ((~e) & g);
        const uint32_t temp1    = h + sum1 + choose + SHA256_K[index] + words[index];
        const uint32_t sum0     = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2    = sum0 + majority;
        h                       = g;
        g                       = f;
        f                       = e;
        e                       = d + temp1;
        d                       = c;
        c                       = b;
        b                       = a;
        a                       = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void init(sha256_context & context) noexcept {
    context.state[0]      = 0x6a09e667U;
    context.state[1]      = 0xbb67ae85U;
    context.state[2]      = 0x3c6ef372U;
    context.state[3]      = 0xa54ff53aU;
    context.state[4]      = 0x510e527fU;
    context.state[5]      = 0x9b05688cU;
    context.state[6]      = 0x1f83d9abU;
    context.state[7]      = 0x5be0cd19U;
    context.bit_length    = 0;
    context.buffer_length = 0;
}

void update(sha256_context & context, const void * input, size_t size) noexcept {
    const auto * bytes = static_cast<const uint8_t *>(input);
    context.bit_length += static_cast<uint64_t>(size) * 8U;
    if (context.buffer_length != 0) {
        const size_t count = std::min<size_t>(64 - context.buffer_length, size);
        std::memcpy(context.buffer + context.buffer_length, bytes, count);
        context.buffer_length += count;
        bytes += count;
        size -= count;
        if (context.buffer_length == 64) {
            compress(context.state, context.buffer);
            context.buffer_length = 0;
        }
    }
    while (size >= 64) {
        compress(context.state, bytes);
        bytes += 64;
        size -= 64;
    }
    if (size != 0) {
        std::memcpy(context.buffer, bytes, size);
        context.buffer_length = size;
    }
}

digest finish(sha256_context & context) noexcept {
    const uint64_t bit_length               = context.bit_length;
    context.buffer[context.buffer_length++] = 0x80;
    if (context.buffer_length > 56) {
        while (context.buffer_length < 64) {
            context.buffer[context.buffer_length++] = 0;
        }
        compress(context.state, context.buffer);
        context.buffer_length = 0;
    }
    while (context.buffer_length < 56) {
        context.buffer[context.buffer_length++] = 0;
    }
    for (int shift = 7; shift >= 0; --shift) {
        context.buffer[context.buffer_length++] =
                static_cast<uint8_t>(bit_length >> (static_cast<unsigned>(shift) * 8U));
    }
    compress(context.state, context.buffer);
    digest result = {};
    for (size_t index = 0; index < 8; ++index) {
        result[index * 4]     = static_cast<uint8_t>(context.state[index] >> 24U);
        result[index * 4 + 1] = static_cast<uint8_t>(context.state[index] >> 16U);
        result[index * 4 + 2] = static_cast<uint8_t>(context.state[index] >> 8U);
        result[index * 4 + 3] = static_cast<uint8_t>(context.state[index]);
    }
    return result;
}

}  // namespace

digest hash(const void * data, size_t size) noexcept {
    sha256_context context;
    init(context);
    update(context, data, size);
    return finish(context);
}

bool equal(const digest & left, const digest & right) noexcept {
    uint8_t difference = 0;
    for (size_t index = 0; index < left.size(); ++index) {
        difference = static_cast<uint8_t>(difference | static_cast<uint8_t>(left[index] ^ right[index]));
    }
    return difference == 0;
}

}  // namespace capture_sha256
}  // namespace server_capture
