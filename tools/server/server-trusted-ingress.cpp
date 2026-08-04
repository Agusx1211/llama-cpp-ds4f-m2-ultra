#include "server-trusted-ingress.h"

namespace server_trusted_scheduling {

classification classify_http_request(const control & ingress, const server_http_req & request) {
    if (request.ambiguous_trusted_scheduling_headers) {
        return { classification_status::rejected, lane::normal, {},
                 "ambiguous trusted scheduling headers" };
    }
    return ingress.classify(request.remote_addr, request.headers);
}

bool apply_to_task(const classification & classified, server_task & task) {
    if (!classified) {
        return false;
    }
    switch (classified.priority) {
        case lane::low:
            task.scheduling.lane = server_task::trusted_lane::low;
            return true;
        case lane::normal:
            task.scheduling.lane = server_task::trusted_lane::normal;
            return true;
        case lane::fast:
            task.scheduling.lane = server_task::trusted_lane::fast;
            return true;
    }
    return false;
}

} // namespace server_trusted_scheduling
