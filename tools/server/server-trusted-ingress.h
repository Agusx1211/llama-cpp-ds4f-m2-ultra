#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-trusted-scheduling.h"

namespace server_trusted_scheduling {

// Token-authenticated scheduling adapter used outside trusted-LAN profile
// mode. Its input deliberately excludes the body: ordinary JSON controls
// cannot participate in header-based trusted scheduling.
classification classify_http_request(const control & ingress, const server_http_req & request);

// Trusted-LAN profile adapter. The exact JSON model allowlist is authoritative;
// legacy lane headers are rejected as a conflicting scheduling mechanism.
classification classify_http_model_profile(const control & ingress, const server_http_req & request, const json & data);

// Rejected classifications leave the task's default normal lane and response
// model untouched. Exact public profiles set both atomically at task creation.
bool apply_to_task(const classification & classified, server_task & task);

} // namespace server_trusted_scheduling
