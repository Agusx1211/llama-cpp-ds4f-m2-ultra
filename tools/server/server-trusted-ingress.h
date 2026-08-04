#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-trusted-scheduling.h"

namespace server_trusted_scheduling {

// This is the exact HTTP-route adapter.  Its input deliberately excludes the
// request body: JSON cannot participate in trusted scheduling classification.
classification classify_http_request(const control & ingress, const server_http_req & request);

// Rejected classifications leave the task's default normal lane untouched.
bool apply_to_task(const classification & classified, server_task & task);

} // namespace server_trusted_scheduling
