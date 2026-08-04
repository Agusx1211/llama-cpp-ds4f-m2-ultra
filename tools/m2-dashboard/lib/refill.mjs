import { formatDuration, setSafeText } from "./safe-dom.mjs";

function node(documentRef, tag, className, text) {
    const result = documentRef.createElement(tag);
    result.className = className;
    setSafeText(result, text);
    return result;
}

function refillMetric(documentRef, label, value) {
    const result = node(documentRef, "div", "metric", "");
    result.append(
        node(documentRef, "span", "metric-label", label),
        node(documentRef, "span", "metric-value", value),
    );
    return result;
}

export function fastRefillView(value, elapsedMs = 0) {
    if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
        throw new RangeError("elapsedMs must be a finite non-negative number");
    }

    const { configuration, cohort, refill } = value;
    const remainingMs = refill.deadline_at === null
        ? 0
        : Math.max(0, refill.remaining_ms - elapsedMs);
    const windowOpen = refill.window_open && remainingMs > 0;
    const oneMemberEligibleNow = refill.one_member_eligible_now && windowOpen;

    let state = "closed";
    if (!configuration.enabled) {
        state = "disabled";
    } else if (!cohort.active) {
        state = "idle";
    } else if (refill.deadline_expired || (refill.deadline_at !== null && remainingMs === 0)) {
        state = "expired";
    } else if (refill.fast_members_remaining === 0) {
        state = "quota_exhausted";
    } else if (oneMemberEligibleNow) {
        state = "eligible";
    } else if (windowOpen) {
        state = "width_full";
    }

    return Object.freeze({
        state,
        remainingMs,
        windowOpen,
        oneMemberEligibleNow,
    });
}

export function renderFastRefill(target, value, elapsedMs = 0, documentRef = globalThis.document) {
    const view = fastRefillView(value, elapsedMs);
    const { configuration, cohort, refill } = value;
    const state = node(documentRef, "div", "metric", "");
    state.append(
        node(documentRef, "span", "metric-label", "Refill state"),
        node(documentRef, "span", `badge badge-${view.state}`, view.state.replaceAll("_", " ")),
    );

    target.replaceChildren(
        state,
        refillMetric(
            documentRef,
            "Configuration",
            configuration.enabled
                ? `${configuration.max_members} members / ${formatDuration(configuration.window_ms)}`
                : "disabled (default)",
        ),
        refillMetric(
            documentRef,
            "Cohort",
            cohort.active ? `${cohort.dominant_lane} · limit ${cohort.limit}` : "inactive",
        ),
        refillMetric(documentRef, "Initial selection", cohort.selection_open ? "open" : "closed"),
        refillMetric(
            documentRef,
            "Fast members",
            `${refill.fast_members_used} used · ${refill.fast_members_remaining} remaining`,
        ),
        refillMetric(
            documentRef,
            "Refill window",
            refill.deadline_at === null
                ? "not started"
                : `${view.windowOpen ? "open" : "closed"} · ${formatDuration(view.remainingMs)} left`,
        ),
        refillMetric(
            documentRef,
            "One member",
            view.oneMemberEligibleNow ? "eligible at sample" : "not eligible at sample",
        ),
    );
    return view;
}
