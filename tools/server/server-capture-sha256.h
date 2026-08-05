#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace server_capture {
namespace capture_sha256 {

using digest = std::array<uint8_t, 32>;

// Shared by the capture store and the dormant resync journal so every
// sidecar uses the standard SHA-256 constants and padding rules.
digest hash(const void * data, size_t size) noexcept;
bool   equal(const digest & left, const digest & right) noexcept;

}  // namespace capture_sha256
}  // namespace server_capture
