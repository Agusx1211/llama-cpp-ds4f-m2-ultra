import { AdminStateClient } from "./lib/client.mjs";
import { ControlIntentBuffer, createControlIntent } from "./lib/controls.mjs";
import { createLiveDashboardTransport } from "./lib/live.mjs";
import {
    contentForDisplay,
    formatBytes,
    formatDuration,
    formatRate,
    setSafeText,
} from "./lib/safe-dom.mjs";

const fixtureRoot = new URL("./fixtures/", import.meta.url);
const intentBuffer = new ControlIntentBuffer(20);

let selectedRequestId = null;
let liveMode = false;
let activeClient = null;
let activeLiveTransport = null;

function available(snapshot, field) {
    return snapshot.availability?.[field] !== false;
}

function unavailable(target, label) {
    target.replaceChildren(element("p", "unavailable", `${label} is unavailable in the registry-only live view.`));
}

function element(tag, className = "", text = null) {
    const node = document.createElement(tag);
    if (className) {
        node.className = className;
    }
    if (text !== null) {
        setSafeText(node, text);
    }
    return node;
}

function valueBlock(label, value, wide = false) {
    const block = element("div", `detail-block${wide ? " detail-block-wide" : ""}`);
    block.append(element("span", "metric-label", label), element("strong", "", value));
    return block;
}

function metric(label, value) {
    const block = element("div", "metric");
    block.append(element("span", "metric-label", label), element("span", "metric-value", value));
    return block;
}

function badge(value) {
    return element("span", `badge badge-${value}`, value.replaceAll("_", " "));
}

function renderHealth(snapshot, state) {
    const server = snapshot.server;
    const strip = document.querySelector("#health-strip");
    if (available(snapshot, "server_metrics")) {
        const health = element("div", "metric");
        health.append(element("span", "metric-label", "Health"), badge(server.health));
        strip.replaceChildren(
            health,
            metric("Model", server.model),
            metric("Build", server.build),
            metric("RSS", formatBytes(server.rss_bytes)),
            metric("Pressure", `${server.memory_pressure_percent.toFixed(0)}%`),
            metric("Swap", formatBytes(server.swap_bytes)),
            metric("Decode width", String(server.decode_width)),
            metric("Aggregate", formatRate(server.aggregate_tokens_per_second, " tok/s")),
        );
    } else {
        const registry = snapshot.registry;
        strip.replaceChildren(
            metric("Source", "request registry"),
            metric("Requests", registry.active_requests.toLocaleString()),
            metric("Occupied slots", registry.occupied_slots.toLocaleString()),
            metric("Permits", registry.total_permits.toLocaleString()),
            metric("Event sequence", registry.total_events.toLocaleString()),
            metric("Retained events", `${registry.retained_events} / ${registry.event_capacity}`),
            metric("Dropped events", registry.dropped_events.toLocaleString()),
            metric("Process metrics", "unavailable"),
        );
    }

    const connection = document.querySelector("#connection-status");
    const recovery = state.recovery.lastError === null
        ? ""
        : ` · ${state.recovery.reason}: ${state.recovery.lastError}`;
    setSafeText(connection, `${state.connection} · event ${state.lastEventId}${recovery}`);
}

function requestCard(request) {
    const card = element("button", "request-card");
    card.type = "button";
    if (request.id === selectedRequestId) {
        card.classList.add("selected");
    }
    card.dataset.requestId = request.id;

    const title = element("div", "request-title");
    title.append(element("strong", "monospace", request.id), badge(request.state));

    const metadata = element("div", "request-meta");
    metadata.append(
        element("span", "", `${request.prompt_tokens.toLocaleString()} prompt`),
        element("span", "", `${request.output_tokens.toLocaleString()} output`),
        element("span", "", `${request.cache_hit_tokens.toLocaleString()} reused`),
        element("span", "", request.blocker || "runnable"),
    );
    card.append(title, metadata);
    card.addEventListener("click", () => {
        selectedRequestId = request.id;
        renderCurrentState();
    });
    return card;
}

function renderLanes(snapshot) {
    const columns = document.querySelector("#lane-columns");
    const laneNodes = snapshot.lanes.map((lane) => {
        const section = element("section", `lane lane-${lane.id}`);
        const heading = element("div", "request-title");
        heading.append(element("h3", "", lane.id), element("span", "muted", `${lane.queued} queued · ${lane.active} active`));
        const summary = element("div", "lane-summary");
        const prediction = available(snapshot, "scheduler_predictions")
            ? `start ${formatDuration(lane.predicted_start_ms[0])}–${formatDuration(lane.predicted_start_ms[1])}`
            : `${lane.bound_permits ?? 0} bound / ${lane.claimed_permits ?? 0} claimed`;
        summary.append(
            element("span", "", `oldest ${formatDuration(lane.oldest_wait_ms)}`),
            element("span", "", prediction),
        );
        section.append(heading, summary);

        const requests = snapshot.requests.filter((request) => request.lane === lane.id);
        if (requests.length === 0) {
            section.append(element("p", "muted", "No resident requests."));
        } else {
            section.append(...requests.map(requestCard));
        }
        return section;
    });
    columns.replaceChildren(...laneNodes);
}

function draftIntent(action, targetId, parameters = {}) {
    intentBuffer.add(createControlIntent(action, targetId, parameters));
    renderIntents();
}

function controlButton(label, action, targetId, parameters = {}) {
    const button = element("button", "", label);
    button.type = "button";
    button.addEventListener("click", () => draftIntent(action, targetId, parameters));
    return button;
}

function contentBlock(label, content) {
    const block = element("div", "detail-block detail-block-wide");
    block.append(element("span", "metric-label", label));
    const pre = element("pre", "content-box");
    setSafeText(pre, contentForDisplay(content));
    block.append(pre, element("small", "muted", "Reveal disabled in fixture foundation."));
    return block;
}

function renderRequestDetail(snapshot) {
    const detail = document.querySelector("#request-detail");
    const selected = snapshot.requests.find((request) => request.id === selectedRequestId);
    if (!selected) {
        selectedRequestId = snapshot.requests[0]?.id ?? null;
    }
    const request = snapshot.requests.find((candidate) => candidate.id === selectedRequestId);
    if (!request) {
        detail.className = "detail-empty";
        detail.replaceChildren(element("p", "muted", "No requests in this snapshot."));
        return;
    }

    detail.className = "detail-grid";
    const reasons = request.scheduler_reasons.length > 0
        ? request.scheduler_reasons.join(" · ")
        : "none";
    const blocks = [
        valueBlock("Request", request.id),
        valueBlock("State", request.state),
        valueBlock("Lane", request.lane),
        valueBlock("Prompt", `${request.prompt_tokens.toLocaleString()} tokens`),
        valueBlock("Output", `${request.output_tokens.toLocaleString()} / ${request.requested_output_tokens ?? "∞"}`),
        valueBlock("Scheduler reasons", reasons, true),
    ];
    if (available(snapshot, "request_latency")) {
        blocks.push(valueBlock("TTFT / TBT", `${formatDuration(request.ttft_ms)} / ${formatDuration(request.tbt_ms)}`));
    }
    if (available(snapshot, "request_kv")) {
        blocks.push(
            valueBlock("KV", `${formatBytes(request.kv.unique_bytes)} unique / ${formatBytes(request.kv.logical_bytes)} logical`),
            valueBlock("Lineage", request.kv.lineage),
        );
    }
    if (available(snapshot, "request_preemption") || available(snapshot, "dspark")) {
        blocks.push(valueBlock("Preemptions / DSpark", `${request.preemptions} / ${request.dspark_cycles}`));
    }
    if (available(snapshot, "content")) {
        blocks.push(contentBlock("Prompt content", request.content.prompt), contentBlock("Output content", request.content.output));
    } else {
        blocks.push(valueBlock("Content", "not collected by this route", true));
    }
    if (!liveMode) {
        const controls = element("div", "detail-block detail-block-wide");
        controls.append(
            element("span", "metric-label", "Local control intent preview"),
            element("p", "notice", "No authenticated mutation adapter is attached."),
        );
        const controlRow = element("div", "control-row");
        controlRow.append(
            controlButton("Cancel", "request.cancel", request.id),
            controlButton(request.state === "parked" ? "Resume" : "Pause", request.state === "parked" ? "request.resume" : "request.pause", request.id),
            controlButton("Move to fast", "request.reprioritize", request.id, { lane: "fast" }),
        );
        controls.append(controlRow);
        blocks.push(controls);
    }
    detail.replaceChildren(...blocks);
}

function progressCard(title, detail, used, capacity, footer) {
    const card = element("article", "stack-card");
    const heading = element("div", "row");
    heading.append(element("strong", "", title), element("span", "muted", detail));
    const track = element("progress", "progress-track");
    track.max = Math.max(1, capacity);
    track.value = Math.min(track.max, Math.max(0, used));
    card.append(heading, track, element("div", "kv-row", footer));
    return card;
}

function renderAllocator(snapshot) {
    const target = document.querySelector("#allocator-pools");
    if (!available(snapshot, "allocator")) {
        unavailable(target, "Allocator state");
        return;
    }
    const cards = snapshot.allocator.pools.map((pool) => progressCard(
        pool.id,
        `${pool.mapped_pages.toLocaleString()} / ${pool.capacity_pages.toLocaleString()} pages`,
        pool.mapped_pages + pool.reserved_pages,
        pool.capacity_pages,
        `free ${pool.free_pages.toLocaleString()} · reserved ${pool.reserved_pages.toLocaleString()} · shared ${pool.shared_pages.toLocaleString()} · COW ${pool.cow_pages.toLocaleString()}`,
    ));
    target.replaceChildren(...cards);
}

function renderCache(snapshot) {
    const target = document.querySelector("#cache-objects");
    if (!available(snapshot, "cache")) {
        unavailable(target, "Cache state");
        return;
    }
    const cards = snapshot.cache.objects.map((object) => {
        const card = element("article", "stack-card");
        const heading = element("div", "row");
        heading.append(element("strong", "monospace", object.id), badge(object.tier));
        const details = element("div", "kv-row");
        details.append(
            element("span", "", `${object.kind} · ${object.hits} hits · score ${object.score.toFixed(2)}`),
            element("span", "", `${formatBytes(object.unique_bytes)} unique · ${formatBytes(object.shared_bytes)} shared`),
        );
        const controls = element("div", "control-row");
        controls.append(
            controlButton(object.pinned ? "Draft unpin" : "Draft pin", object.pinned ? "cache.unpin" : "cache.pin", object.id),
            controlButton("Draft eviction", "cache.evict", object.id),
        );
        card.append(heading, details, controls);
        return card;
    });
    target.replaceChildren(...cards);
}

function renderDisks(snapshot) {
    const target = document.querySelector("#disk-list");
    if (!available(snapshot, "disks")) {
        unavailable(target, "Storage state");
        return;
    }
    const cards = snapshot.disks.map((disk) => {
        const health = disk.healthy ? "healthy" : "degraded";
        const card = progressCard(
            disk.path,
            health,
            disk.capacity_bytes - disk.free_bytes,
            disk.capacity_bytes,
            `queue ${disk.queue_depth} · read ${formatBytes(disk.read_bps)}/s · write ${formatBytes(disk.write_bps)}/s · ${disk.errors} errors`,
        );
        card.prepend(badge(health));
        return card;
    });
    target.replaceChildren(...cards);
}

function renderAux(snapshot) {
    const target = document.querySelector("#aux-status");
    if (!available(snapshot, "dspark") && !available(snapshot, "capture")) {
        unavailable(target, "DSpark and capture state");
        return;
    }
    const dspark = element("article", "stack-card");
    const accepted = snapshot.dspark.proposals > 0
        ? snapshot.dspark.accepted / snapshot.dspark.proposals * 100
        : 0;
    dspark.append(
        element("span", "metric-label", "DSpark"),
        element("strong", "", `${snapshot.dspark.mode} · width ${snapshot.dspark.scheduled_decode_width}`),
        element("p", "muted", `${snapshot.dspark.accepted.toLocaleString()} / ${snapshot.dspark.proposals.toLocaleString()} accepted (${accepted.toFixed(1)}%)`),
    );

    const capture = element("article", "stack-card");
    capture.append(
        element("span", "metric-label", "Capture"),
        element("strong", "", `${snapshot.capture.mode} · ${snapshot.capture.healthy ? "healthy" : "degraded"}`),
        element("p", "muted", `${snapshot.capture.written_records.toLocaleString()} written · ${snapshot.capture.dropped_records.toLocaleString()} dropped · ${formatBytes(snapshot.capture.bytes_written)}`),
    );
    if (!liveMode) {
        const controls = element("div", "control-row");
        controls.append(controlButton("Draft request exclusion", "capture.exclude", selectedRequestId ?? "none"));
        capture.append(controls);
    }
    target.replaceChildren(dspark, capture);
}

function timelineItem(title, detail) {
    const item = element("li", "timeline-item");
    item.append(element("strong", "", title), element("span", "muted", detail));
    return item;
}

function renderTimeline(snapshot) {
    const target = document.querySelector("#timeline");
    const items = [...snapshot.timeline].reverse().map((item) => timelineItem(
        item.label,
        `${item.at} · ${item.type}${item.request_id ? ` · ${item.request_id}` : ""}`,
    ));
    target.replaceChildren(...items);
}

function renderHistory(state) {
    setSafeText(document.querySelector("#event-sequence"), `last ${state.lastEventId}`);
    const target = document.querySelector("#event-history");
    const items = [...state.history].reverse().map((event) => timelineItem(
        event.type,
        `#${event.id} · ${event.reason_code}${event.request_id ? ` · ${event.request_id}` : ""}`,
    ));
    if (items.length === 0) {
        items.push(timelineItem("Snapshot loaded", `#${state.lastEventId}`));
    }
    target.replaceChildren(...items);
}

function renderIntents() {
    const target = document.querySelector("#intent-list");
    const items = [...intentBuffer.snapshot()].reverse().map((intent) => timelineItem(
        intent.action,
        `${intent.targetId} · ${intent.transport}`,
    ));
    if (items.length === 0) {
        items.push(timelineItem("No intents", "Controls remain local and inert."));
    }
    target.replaceChildren(...items);
}

let currentState = null;

function renderCurrentState() {
    if (currentState === null) {
        return;
    }
    const snapshot = currentState.snapshot;
    if (!snapshot.requests.some((request) => request.id === selectedRequestId)) {
        selectedRequestId = snapshot.requests[0]?.id ?? null;
    }
    renderHealth(snapshot, currentState);
    renderLanes(snapshot);
    renderRequestDetail(snapshot);
    renderAllocator(snapshot);
    renderCache(snapshot);
    renderDisks(snapshot);
    renderAux(snapshot);
    renderTimeline(snapshot);
    renderHistory(currentState);
    renderIntents();
}

async function loadJson(url) {
    const response = await fetch(url, {
        method: "GET",
        cache: "no-store",
        credentials: "same-origin",
        headers: { Accept: "application/json" },
    });
    if (!response.ok) {
        throw new Error(`fixture fetch failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
}

function fixtureTransport(events) {
    return ({ lastEventId, onEvent }) => {
        const remaining = events.filter((event) => event.sequence > Number(lastEventId));
        const timers = remaining.map((event, index) => setTimeout(() => {
            onEvent({ data: JSON.stringify(event), lastEventId: event.id });
        }, 700 * (index + 1)));
        return () => timers.forEach((timer) => clearTimeout(timer));
    };
}

async function loadFixtures() {
    return Promise.all([
        loadJson(new URL("state.json", fixtureRoot)),
        loadJson(new URL("events.json", fixtureRoot)),
    ]);
}

function stopActiveConnection() {
    activeClient?.stop();
    activeClient = null;
    activeLiveTransport?.clear();
    activeLiveTransport = null;
}

async function startClient(transport, mode) {
    stopActiveConnection();
    liveMode = mode === "live";
    if (liveMode) {
        activeLiveTransport = transport;
    }
    const badgeNode = document.querySelector("#mode-badge");
    badgeNode.className = `badge badge-${liveMode ? "live" : "fixture"}`;
    setSafeText(badgeNode, liveMode ? "live read-only" : "fixture data");

    activeClient = new AdminStateClient({
        getSnapshot: transport.getSnapshot,
        openEvents: transport.openEvents,
        onState: (state) => {
            currentState = state;
            renderCurrentState();
        },
        historyLimit: 32,
        pendingLimit: 16,
    });
    await activeClient.start();
}

async function connectFixture() {
    const [snapshot, events] = await loadFixtures();
    await startClient({
        getSnapshot: async () => snapshot,
        openEvents: fixtureTransport(events),
    }, "fixture");
}

async function main() {
    document.querySelector("#fixture-button").addEventListener("click", () => {
        void connectFixture().catch(showConnectionError);
    });
    document.querySelector("#live-form").addEventListener("submit", (event) => {
        event.preventDefault();
        const serverUrl = document.querySelector("#server-url").value;
        const apiKeyInput = document.querySelector("#api-key");
        const operatorInput = document.querySelector("#operator-token");
        try {
            const transport = createLiveDashboardTransport({
                baseUrl: serverUrl,
                apiKey: apiKeyInput.value,
                operatorToken: operatorInput.value,
            });
            apiKeyInput.value = "";
            operatorInput.value = "";
            void startClient(transport, "live").catch((error) => {
                transport.clear();
                showConnectionError(error);
            });
        } catch (error) {
            apiKeyInput.value = "";
            operatorInput.value = "";
            showConnectionError(error);
        }
    });
    globalThis.addEventListener("pagehide", stopActiveConnection, { once: true });
    await connectFixture();
}

function showConnectionError(error) {
    const status = document.querySelector("#connection-status");
    setSafeText(status, `connection error: ${error.message ?? error}`);
    status.className = "badge badge-failed";
}

main().catch(showConnectionError);
