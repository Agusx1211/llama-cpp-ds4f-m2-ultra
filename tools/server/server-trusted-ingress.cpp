#include "server-trusted-ingress.h"

namespace server_trusted_scheduling {

classification classify_http_request(const control & ingress, const server_http_req & request) {
    if (request.ambiguous_trusted_scheduling_headers) {
        return { classification_status::rejected, lane::normal, {},
                 "ambiguous trusted scheduling headers" };
    }
    return ingress.classify(request.remote_addr, request.headers);
}

classification classify_http_model_profile(const control &         ingress,
                                           const server_http_req & request,
                                           const json &            data) {
    if (!ingress.trust_lan()) {
        return {};
    }
    if (request.ambiguous_trusted_scheduling_headers) {
        return {
            classification_status::rejected,
            lane::normal,
            {},
            "ambiguous trusted scheduling headers",
        };
    }
    if (has_lane_header(request.headers)) {
        return {
            classification_status::rejected,
            lane::normal,
            {},
            "trusted-LAN scheduling is selected by model ID; lane headers are not accepted",
        };
    }

    std::optional<std::string> model;
    if (data.contains("model")) {
        if (!data.at("model").is_string()) {
            return {
                classification_status::rejected,
                lane::normal,
                {},
                "model must be a string",
            };
        }
        model = data.at("model").get<std::string>();
    }
    return classify_model_profile(true, model);
}

bool apply_to_task(const classification & classified, server_task & task) {
    if (!classified) {
        return false;
    }
    switch (classified.priority) {
        case lane::low:
            task.scheduling.lane = server_task::trusted_lane::low;
            break;
        case lane::normal:
            task.scheduling.lane = server_task::trusted_lane::normal;
            break;
        case lane::fast:
            task.scheduling.lane = server_task::trusted_lane::fast;
            break;
        default:
            return false;
    }
    if (!classified.public_model.empty()) {
        task.params.oaicompat_model = classified.public_model;
    }
    return true;
}

} // namespace server_trusted_scheduling
