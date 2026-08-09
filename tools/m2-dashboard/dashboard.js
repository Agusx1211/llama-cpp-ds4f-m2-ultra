// M2 llama-server operations dashboard — browser entry point.
//
// v2: three views. "Live" keeps the v1 polling dashboard (/slots + registry
// snapshot). "Timeline" and "Cache" ride the realtime /m2-dashboard/events
// SSE stream (schema v1) with the poller as degraded fallback. No build
// step, no external dependencies. `?fixture=1|replay|demo` replays recorded
// slot payloads; `?fixture=v2` additionally replays a recorded event stream
// for the new views. The API key comes from `#key=...` in the URL hash or
// the inline form and is kept in memory only.

import {
    LANE_ORDER,
    PHASE,
    SlotTracker,
    formatClock,
    formatDuration,
    formatPercent,
    formatRate,
    formatTokens,
    parseMetrics,
    registryView,
    shortModelName,
    sparkPath,
} from "./lib/core.mjs";
import {
    CACHE_STATE_PATH,
    EventStore,
    LiveEventStream,
    cacheView,
    hitRateSeries,
    phaseBreakdown,
    timeSegments,
    tokenSpans,
    wastedRestore,
} from "./lib/events.mjs";
import {
    CACHE_PREVIEW_PATH,
    CONTENT_PATH,
    WATCH_REASON_LABEL,
    WatchStream,
    cachePreviewLabel,
    extentLabel,
    parseCachePreview,
    parseContent,
} from "./lib/content.mjs";

const POLL_MS = 2000;
const SSE_POLL_MS = 5000;     // slots poll relaxes once the event stream is live
const HIDDEN_POLL_MS = 15000;
const MAX_BACKOFF_MS = 30000;
const SPARK_WINDOW_MS = 4 * 60 * 1000;
const METRICS_EVERY = 5; // poll /metrics every N ticks when available
const TIMELINE_ROWS = 30;
const CACHE_POLL_MS = 5000;

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Transports
// ---------------------------------------------------------------------------

function liveTransport(apiKey) {
    const headers = { Accept: "application/json" };
    if (apiKey) headers.Authorization = "Bearer " + apiKey;
    const get = async (path, type = "json") => {
        const res = await fetch(path, { headers, cache: "no-store" });
        if (!res.ok) {
            const err = new Error(path + " -> " + res.status);
            err.status = res.status;
            throw err;
        }
        return type === "json" ? res.json() : res.text();
    };
    return {
        mode: "live",
        props: () => get("/props"),
        slots: () => get("/slots"),
        registry: () => get("/internal/admin/dashboard/snapshot"),
        metrics: () => get("/metrics", "text"),
        cacheState: () => get(CACHE_STATE_PATH),
        // v3 content routes: only ever called from an explicit reveal action
        content: (id) => get(CONTENT_PATH + "?req=" + encodeURIComponent(String(id))),
        cachePreview: (id) => get(CACHE_PREVIEW_PATH + "?id=" + encodeURIComponent(String(id))),
        requestDetail: async (id) => {
            const res = await fetch("/internal/admin/dashboard/request-detail", {
                method: "POST",
                headers: { ...headers, "Content-Type": "application/json" },
                body: JSON.stringify({ request_id: id }),
            });
            if (!res.ok) { const e = new Error("detail " + res.status); e.status = res.status; throw e; }
            return res.json();
        },
        cancelRequest: async (id) => {
            const res = await fetch("/internal/admin/dashboard/request-control", {
                method: "POST",
                headers: { ...headers, "Content-Type": "application/json" },
                body: JSON.stringify({ request_id: id, action: "cancel" }),
            });
            if (!res.ok) { const e = new Error("cancel " + res.status); e.status = res.status; throw e; }
            return res.json();
        },
    };
}

function fixtureTransport(replay, startIndex = 0) {
    let i = startIndex;
    const tick = () => replay.ticks[Math.min(i, replay.ticks.length - 1)];
    return {
        mode: "fixture",
        props: async () => replay.props,
        slots: async () => {
            const t = tick();
            if (i < replay.ticks.length - 1) i += 1;
            return t.slots;
        },
        registry: async () => tick().registry,
        metrics: async () => { const e = new Error("fixture"); e.status = 501; throw e; },
        cacheState: async () => {
            if (!state.fixtureEvents?.cache_state) { const e = new Error("fixture"); e.status = 404; throw e; }
            return state.fixtureEvents.cache_state;
        },
        content: async (id) => {
            const body = state.fixtureContent?.requests?.[String(id)];
            if (!body) { const e = new Error("unknown"); e.status = 404; throw e; }
            return body;
        },
        cachePreview: async (id) => {
            const body = state.fixtureContent?.cache?.[String(id)];
            if (!body) return { v: 1, id, found: false };
            return body;
        },
        requestDetail: async (id) => {
            const slot = tick().bindings?.[id];
            if (slot === undefined) { const e = new Error("unknown"); e.status = 404; throw e; }
            return { registry: { bindings: [{ slot_id: slot }] } };
        },
        cancelRequest: async () => ({ status: "accepted" }),
    };
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

const state = {
    transport: null,
    tracker: new SlotTracker(),
    props: null,
    registry: null,
    lastView: null,
    metrics: null,
    metricsEnabled: null, // null = untested, false = 501/absent, true = on
    bindings: new Map(),  // request id -> slot id (from request-detail)
    bindingFailed: false, // detail route denied: stop trying
    failures: 0,
    tickCount: 0,
    pollTimer: null,
    lastError: null,
    cancelPending: new Set(),
    // v2
    view: "live",
    events: new EventStore(),
    stream: null,
    streamStatus: "off",  // off | connecting | open | retrying | degraded
    fixtureEvents: null,
    tlWindow: "auto",
    tlScale: "wall",      // wall = shared clock axis | fit = per-request normalized
    tlExpanded: null,     // expanded request id
    tlTimer: null,
    tlDirty: false,
    cacheState: null,
    cacheTimer: null,
    // v3 content. Everything here is memory-only and explicitly bounded: a
    // reveal is a deliberate act, it is never persisted, and the page keeps at
    // most MAX_REVEALS of them alive at once.
    contentEnabled: null,       // from /props; null = unknown yet
    reveal: new Map(),          // req id -> {status, view?, error?}
    contentNodes: new Map(),    // req id -> the live DOM node for that block
    watch: null,                // {req, stream, state} while a mirror is open
    cacheExpanded: null,        // expanded cache entry id
    cachePreview: new Map(),    // entry id -> {status, view?, error?}
    fixtureContent: null,
};

// A page left open for a day must not accumulate prompts. Reveals are capped
// and evicted oldest-first, mirroring the server store's own bound.
const MAX_REVEALS = 8;
const MAX_CACHE_PREVIEWS = 8;

function currentApiKey() {
    const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
    return state.enteredKey ?? hash.get("key") ?? "";
}

// ---------------------------------------------------------------------------
// Poll loop (v1 views; also the degraded path when SSE is down)
// ---------------------------------------------------------------------------

async function poll() {
    clearTimeout(state.pollTimer);
    const t = state.transport;
    try {
        if (state.props === null) {
            state.props = await t.props();
            // v3: the server tells us up front whether the content routes will
            // answer, so the page shows "disclosure disabled" instead of
            // offering a button that always fails
            if (typeof state.props.dashboard_content === "boolean") {
                state.contentEnabled = state.props.dashboard_content;
            } else if (state.transport.mode === "live") {
                state.contentEnabled = null; // older server: find out on first use
            }
            renderHeader();
        }
        const [slots, registry] = await Promise.all([
            t.slots(),
            t.registry().catch(() => null), // registry is additive, not required
        ]);
        state.registry = registryView(registry);
        state.lastView = state.tracker.feed(performance.now(), slots);
        state.failures = 0;
        state.lastError = null;
        state.tickCount += 1;
        await maybeJoinBindings();
        maybePollMetrics();
        render();
    } catch (error) {
        state.failures += 1;
        state.lastError = error;
        if (error.status === 401) {
            showKeyForm("this server requires an API key");
            return; // resume via the form
        }
        renderStatus();
        document.querySelector("main").classList.add("stale");
    }
    schedule();
}

function schedule() {
    let base = document.hidden ? HIDDEN_POLL_MS : POLL_MS;
    // the event stream carries request-level realtime; slot occupancy can
    // relax to a slower poll unless the Live view is front and center
    if (!document.hidden && state.streamStatus === "open" && state.view !== "live") base = SSE_POLL_MS;
    const backoff = state.failures > 0 ? Math.min(MAX_BACKOFF_MS, POLL_MS * 2 ** state.failures) : base;
    state.pollTimer = setTimeout(poll, backoff);
}

async function maybeJoinBindings() {
    if (state.bindingFailed || !state.registry) return;
    for (const req of state.registry.active) {
        if (state.bindings.has(req.id)) continue;
        try {
            const detail = await state.transport.requestDetail(req.id);
            const slot = detail?.registry?.bindings?.[0]?.slot_id;
            if (typeof slot === "number") state.bindings.set(req.id, slot);
            // not bound yet: leave unset so the next poll retries
        } catch (error) {
            if (error.status === 403 || error.status === 401) { state.bindingFailed = true; }
            // transient failure (request may have finished between the snapshot
            // and the detail call): retry on the next poll while it stays live
        }
    }
    // trim mappings for departed requests
    const live = new Set(state.registry.active.map((r) => r.id).concat(state.registry.queued.map((r) => r.id)));
    for (const id of [...state.bindings.keys()]) {
        if (!live.has(id)) state.bindings.delete(id);
    }
}

function maybePollMetrics() {
    if (state.metricsEnabled === false) return;
    if (state.tickCount % METRICS_EVERY !== 1) return;
    state.transport.metrics().then((text) => {
        state.metricsEnabled = true;
        state.metrics = parseMetrics(text);
        renderFooter();
    }).catch(() => {
        state.metricsEnabled = false;
        renderFooter();
    });
}

// ---------------------------------------------------------------------------
// Event stream (v2 realtime backbone)
// ---------------------------------------------------------------------------

function startStream() {
    if (state.transport.mode !== "live") return;
    state.stream = new LiveEventStream({
        apiKey: currentApiKey(),
        onEvent: (evt) => {
            state.events.feed(evt);
            state.tlDirty = true;
            if (evt.k === "cache_op" && state.view === "cache") scheduleCachePoll(1000);
        },
        onStatus: (status) => {
            state.streamStatus = status;
            renderStatus();
        },
    });
    state.stream.start();
}

function restartStream() {
    if (state.stream) state.stream.stop();
    state.streamStatus = "off";
    startStream();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
}

function formatBytes(n) {
    if (n === null || n === undefined) return "—";
    if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(n >= 10 * 1024 ** 3 ? 0 : 1) + " GiB";
    if (n >= 1024 ** 2) return (n / 1024 ** 2).toFixed(0) + " MiB";
    if (n >= 1024) return (n / 1024).toFixed(0) + " KiB";
    return n + " B";
}

function serverNow() {
    return Date.now() + state.events.clockSkewMs;
}

function renderHeader() {
    const model = state.props ? shortModelName(state.props.model_alias ?? state.props.model_path) : "";
    $("model").textContent = model;
    $("model").title = state.props?.model_path ?? "";
    $("build").textContent = state.props?.build_info ?? "";
    document.title = "M2 llama-server · " + model;
}

// A standing, always-visible marker of whether any prompt or generated text is
// currently on screen. Content is the one thing on this page that is not
// metadata, so its presence is never implicit.
function renderContentFlag() {
    const flag = $("content-flag");
    if (!flag) return;
    const revealed = [...state.reveal.values()].filter((e) => e.status === "ok" && e.view?.found).length;
    const watching = state.watch ? 1 : 0;
    const previews = [...state.cachePreview.values()].filter((e) => e.status === "ok" && e.view?.found).length;
    const total = revealed + watching + previews;
    flag.hidden = total === 0;
    if (total === 0) return;
    flag.replaceChildren(
        el("span", "reveal-flag", "content shown"),
        document.createTextNode(" " + total + " "),
    );
    const hideAll = el("button", "", "hide all");
    hideAll.type = "button";
    hideAll.addEventListener("click", () => {
        stopWatch();
        const reqs = [...state.reveal.keys()];
        state.reveal.clear();
        state.cachePreview.clear();
        for (const id of reqs) {
            const node = state.contentNodes.get(id);
            if (node) { node.dataset.rev = ""; node.replaceChildren(); }
        }
        state.contentNodes.clear();
        renderContentFlag();
        if (state.view === "timeline") renderTimeline({ force: true });
        if (state.view === "cache") renderCache();
    });
    flag.append(hideAll);
}

function renderStatus() {
    renderContentFlag();
    const status = $("status");
    if (state.failures > 0) {
        const wait = Math.min(MAX_BACKOFF_MS, POLL_MS * 2 ** state.failures) / 1000;
        status.className = "status err";
        status.textContent = "unreachable · retrying in " + Math.round(wait) + "s";
        return;
    }
    if (state.transport.mode === "fixture") {
        status.className = "status warn";
        status.textContent = "fixture replay";
        return;
    }
    if (state.streamStatus === "open") {
        status.className = "status ok";
        status.textContent = "live · push";
        return;
    }
    if (state.streamStatus === "degraded") {
        status.className = "status warn";
        status.textContent = "live · polling fallback (stream down)";
        return;
    }
    status.className = "status ok";
    status.textContent = "live · " + (document.hidden ? HIDDEN_POLL_MS : POLL_MS) / 1000 + "s poll";
}

function render() {
    document.querySelector("main").classList.remove("stale");
    renderStatus();
    if (state.view === "live") {
        renderTiles();
        renderWork();
        renderTail();
        renderLanes();
    } else if (state.view === "timeline") {
        renderTimeline();
    } else if (state.view === "cache") {
        renderCache();
    }
    renderFooter();
}

// ---- tiles ----

function tile(className, label, value, unit, sub) {
    const node = el("div", "tile " + className);
    node.append(el("span", "label", label));
    const v = el("span", "value", value);
    if (unit) v.append(el("small", "", unit));
    node.append(v);
    if (sub) node.append(el("span", "sub", sub));
    return node;
}

function miniSvg(points, { width = 200, height = 34 } = {}) {
    // shared tiny line+area chart over pre-scaled {x, y} points is not needed;
    // callers use sparkPath-compatible history instead. Kept minimal.
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.setAttribute("preserveAspectRatio", "none");
    return svg;
}

function sparkTile(className, label, rateKey, current) {
    const node = tile(className, label, formatRate(current), " tok/s");
    const geo = sparkPath(state.tracker.history, rateKey, {
        width: 200, height: 34, windowMs: SPARK_WINDOW_MS, now: performance.now(),
    });
    const svg = miniSvg(null, {});
    svg.setAttribute("aria-label", label + ", last 4 minutes, peak " + formatRate(geo.max) + " tok/s");
    if (geo.path) {
        const area = document.createElementNS("http://www.w3.org/2000/svg", "path");
        area.setAttribute("d", geo.area);
        area.setAttribute("class", "spark-area");
        const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
        line.setAttribute("d", geo.path);
        line.setAttribute("class", "spark-line");
        const endDot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        const lastPt = geo.points[geo.points.length - 1];
        endDot.setAttribute("cx", lastPt.x);
        endDot.setAttribute("cy", lastPt.y);
        endDot.setAttribute("r", 3);
        endDot.setAttribute("class", "spark-dot");
        svg.append(area, line, endDot);
        attachSparkHover(svg, node, geo, label);
    }
    node.append(svg);
    return node;
}

function attachSparkHover(svg, tileNode, geo, label) {
    const tooltip = $("tooltip");
    const hair = document.createElementNS("http://www.w3.org/2000/svg", "line");
    hair.setAttribute("class", "spark-hair");
    hair.setAttribute("y1", 0);
    hair.setAttribute("y2", 34);
    hair.setAttribute("visibility", "hidden");
    svg.append(hair);
    svg.addEventListener("pointermove", (ev) => {
        const rect = svg.getBoundingClientRect();
        const x = ((ev.clientX - rect.left) / rect.width) * 200;
        let best = geo.points[0];
        for (const p of geo.points) if (Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
        hair.setAttribute("x1", best.x);
        hair.setAttribute("x2", best.x);
        hair.setAttribute("visibility", "visible");
        const ageS = Math.round((performance.now() - best.t) / 1000);
        tooltip.hidden = false;
        tooltip.replaceChildren(el("b", "", formatRate(best.v) + " tok/s"), el("span", "", " · " + label + " · " + ageS + "s ago"));
        tooltip.style.left = Math.min(ev.clientX + 12, window.innerWidth - tooltip.offsetWidth - 8) + "px";
        tooltip.style.top = (ev.clientY + 14) + "px";
    });
    svg.addEventListener("pointerleave", () => {
        hair.setAttribute("visibility", "hidden");
        tooltip.hidden = true;
    });
}

function renderTiles() {
    const view = state.lastView;
    if (!view) return;
    const tiles = $("tiles");
    const reg = state.registry;
    const nodes = [];

    const warmedUp = state.tracker.history.length > 0;
    nodes.push(sparkTile("gen", "Generation", "gen", warmedUp ? view.genRate : null));
    nodes.push(sparkTile("pp", "Prompt processing", "pp", warmedUp ? view.ppRate : null));

    const total = view.slots.length;
    const queued = reg ? reg.queued.length : 0;
    const reqTile = tile("", "Requests", String(view.processingCount), " / " + total + " slots",
        queued > 0 ? queued + " queued" : "queue empty");
    nodes.push(reqTile);

    // context: unified KV budget shared by all slots
    const budget = view.slots[0]?.slot.nCtx ?? 0;
    const used = view.contextTokens;
    const frac = budget > 0 ? Math.min(1, used / budget) : 0;
    const ctxTile = tile("", "KV context", formatTokens(used), " / " + formatTokens(budget),
        formatPercent(frac) + " of shared budget" + (frac >= 0.9 ? " · ▲ near limit" : ""));
    const meter = el("div", "meter" + (frac >= 0.9 ? " crit" : frac >= 0.75 ? " warn" : ""));
    const fill = el("i");
    fill.style.width = (frac * 100).toFixed(1) + "%";
    meter.append(fill);
    ctxTile.append(meter);
    nodes.push(ctxTile);

    const reuse = state.tracker.cacheReuse(view.slots);
    nodes.push(tile("", "Prefix cache reuse",
        reuse ? formatPercent(reuse.fraction) : "—", "",
        reuse ? formatTokens(reuse.cachedTokens) + " of " + formatTokens(reuse.promptTokens) + " prompt tok reused" : "no requests observed yet"));

    tiles.replaceChildren(...nodes);
}

// ---- work table (slots + queued requests) ----

function slotRequestId(slotId) {
    for (const [reqId, sid] of state.bindings) {
        if (sid === slotId) return reqId;
    }
    return null;
}

function laneOf(reqId) {
    if (!reqId || !state.registry) return null;
    const req = state.registry.active.find((r) => r.id === reqId);
    return req ? req.lane : null;
}

function cancelButton(reqId) {
    const btn = el("button", "", "cancel");
    btn.disabled = state.cancelPending.has(reqId);
    btn.addEventListener("click", () => {
        if (!confirm("Cancel request " + reqId + "?")) return;
        state.cancelPending.add(reqId);
        btn.disabled = true;
        state.transport.cancelRequest(reqId).catch(() => {}).finally(() => {
            state.cancelPending.delete(reqId);
        });
    });
    return btn;
}

function renderWork() {
    const view = state.lastView;
    if (!view) return;
    const body = $("work-body");
    const rows = [];

    for (const v of view.slots) {
        const slot = v.slot;
        const tr = el("tr", "phase-" + v.phase + (v.phase === PHASE.IDLE ? " idle" : ""));
        tr.append(el("td", "strong", String(slot.id)));

        const reqId = slot.processing ? slotRequestId(slot.id) : null;
        tr.append(el("td", "", reqId ? laneOf(reqId) ?? "—" : "—"));
        tr.append(el("td", "", reqId ?? (slot.taskId !== null ? "task " + slot.taskId : "—")));

        const phaseTd = el("td", "");
        phaseTd.append(el("i", "phase-dot"), document.createTextNode(v.phase));
        tr.append(phaseTd);

        const rate = v.phase === PHASE.PREFILL ? v.ppRate : v.genRate;
        tr.append(el("td", "num" + (rate ? " strong" : ""), rate ? formatRate(rate) : "—"));

        tr.append(progressCell(v));
        tr.append(el("td", "num", slot.taskId !== null ? formatTokens(v.contextTokens) : "—"));

        const cachedTd = el("td", "num");
        if (slot.processing && slot.cached > 0) {
            const promptTotal = Math.max(slot.promptTokens, slot.cached + slot.processed);
            cachedTd.append(document.createTextNode(formatTokens(slot.cached) + " "),
                el("span", "u", "(" + formatPercent(slot.cached / promptTotal) + ")"));
        } else {
            cachedTd.textContent = slot.processing ? "0" : "—";
        }
        tr.append(cachedTd);

        const req = reqId && state.registry ? state.registry.active.find((r) => r.id === reqId) : null;
        tr.append(el("td", "num", req ? formatDuration(req.ageMs) : "—"));

        const actionTd = el("td", "");
        if (reqId && state.transport.mode === "live" && !state.bindingFailed) actionTd.append(cancelButton(reqId));
        tr.append(actionTd);
        rows.push(tr);
    }

    // queued registry requests (no slot yet)
    if (state.registry) {
        for (const req of state.registry.queued) {
            const tr = el("tr", "phase-queued");
            tr.append(el("td", "", "—"));
            tr.append(el("td", "", req.lane));
            tr.append(el("td", "", req.id));
            const phaseTd = el("td", "");
            phaseTd.append(el("i", "phase-dot"), document.createTextNode("queued"));
            tr.append(phaseTd);
            tr.append(el("td", "num", "—"));
            tr.append(el("td", "", "waiting"));
            tr.append(el("td", "num", req.promptTokens > 0 ? formatTokens(req.promptTokens) : "—"));
            tr.append(el("td", "num", "—"));
            tr.append(el("td", "num", formatDuration(req.ageMs)));
            const actionTd = el("td", "");
            if (state.transport.mode === "live" && !state.bindingFailed) actionTd.append(cancelButton(req.id));
            tr.append(actionTd);
            rows.push(tr);
        }
    }

    body.replaceChildren(...rows);
}

function progressCell(v) {
    const td = el("td", "");
    const slot = v.slot;
    if (v.phase === PHASE.PREFILL && v.prefillFraction !== null) {
        const bar = el("span", "bar pp");
        const fill = el("i");
        fill.style.width = (v.prefillFraction * 100).toFixed(1) + "%";
        bar.append(fill);
        td.append(bar, document.createTextNode(
            formatPercent(v.prefillFraction) + " · " + formatTokens(slot.cached + slot.processed) + "/" + formatTokens(slot.promptTokens)));
    } else if (v.phase === PHASE.GENERATING) {
        if (v.decodeFraction !== null) {
            const bar = el("span", "bar gen");
            const fill = el("i");
            fill.style.width = (v.decodeFraction * 100).toFixed(1) + "%";
            bar.append(fill);
            td.append(bar);
        }
        const budget = slot.remain !== null && slot.remain >= 0 ? "/" + formatTokens(slot.decoded + slot.remain) : "";
        td.append(document.createTextNode(formatTokens(slot.decoded) + budget + " out"));
    } else if (slot.taskId !== null) {
        td.append(el("span", "u", formatTokens(slot.decoded) + " out · resident"));
    } else {
        td.textContent = "—";
    }
    return td;
}

// ---- tail ----

function renderTail() {
    const body = $("tail-body");
    const tail = state.tracker.tail.slice(0, 15);
    $("tail-note").textContent = tail.length > 0 ? "· observed since page load" : "";
    const rows = tail.map((rec) => {
        const tr = el("tr", "");
        tr.append(el("td", "", formatClock(rec.endedAt)));
        tr.append(el("td", "", String(rec.slot)));
        tr.append(el("td", "num", formatTokens(rec.promptTokens)));
        const cachedTd = el("td", "num");
        if (rec.cachedTokens > 0 && rec.promptTokens > 0) {
            cachedTd.append(document.createTextNode(formatTokens(rec.cachedTokens) + " "),
                el("span", "u", "(" + formatPercent(rec.cachedTokens / rec.promptTokens) + ")"));
        } else {
            cachedTd.textContent = "0";
        }
        tr.append(cachedTd);
        tr.append(el("td", "num", formatTokens(rec.generated)));
        tr.append(el("td", "num", rec.ppRate ? formatRate(rec.ppRate) : "—"));
        tr.append(el("td", "num strong", rec.genRate ? formatRate(rec.genRate) : "—"));
        tr.append(el("td", "num", formatDuration(rec.durationMs) + (rec.partial ? "+" : "")));
        return tr;
    });
    body.replaceChildren(...rows);
    if (rows.length === 0) {
        const tr = el("tr");
        const td = el("td", "empty", "requests that finish while this page is open appear here");
        td.colSpan = 8;
        tr.append(td);
        body.append(tr);
    }
}

// ---- lanes + footer ----

function renderLanes() {
    const target = $("lanes");
    if (!state.registry) { target.textContent = ""; return; }
    const parts = LANE_ORDER.map((name) => {
        const lane = state.registry.lanes[name] ?? { active: 0, queued: 0, oldestWaitMs: 0 };
        let text = lane.active + " active";
        if (lane.queued > 0) text += " · " + lane.queued + " queued (oldest " + formatDuration(lane.oldestWaitMs) + ")";
        return name + " " + text;
    });
    target.replaceChildren(document.createTextNode("lanes: "));
    parts.forEach((p, i) => {
        if (i > 0) target.append(document.createTextNode("   ·   "));
        const [name, ...rest] = p.split(" ");
        target.append(el("b", "", name), document.createTextNode(" " + rest.join(" ")));
    });
}

function renderFooter() {
    const bits = [];
    if (state.transport.mode === "fixture") {
        bits.push("replaying recorded fixtures — append #key=… and drop ?fixture for live data");
    } else if (state.streamStatus === "open") {
        bits.push("event stream live (schema v1) · slots poll " + SSE_POLL_MS / 1000 + "s");
    } else {
        bits.push("polling /slots + request registry");
    }
    if (state.metricsEnabled === true && state.metrics) {
        const p = state.metrics;
        bits.push("KV pressure events " + (p.kv_physical_pressure_total ?? 0) +
            " · retries " + (p.kv_physical_pressure_retries_total ?? 0) +
            " · victims " + (p.kv_physical_pressure_victims_total ?? 0));
    } else if (state.metricsEnabled === false && state.transport.mode === "live") {
        bits.push("/metrics disabled on this server (start with --metrics for KV-pressure counters)");
    }
    if (state.contentEnabled === false) {
        bits.push("content disclosure disabled (--no-dashboard-content)");
    } else if (state.contentEnabled === true) {
        bits.push("prompt/output text is fetched only when you ask for it, per request");
    }
    $("foot").textContent = bits.join("  ·  ");
}

// ---------------------------------------------------------------------------
// Timeline view
// ---------------------------------------------------------------------------

const SEG_CLASS = { queued: "q", prefill: "pp", decode: "gen" };
const SRC_CLASS = { resident: "c0", donor: "c0", ram: "c1", disk: "c2" };
const SRC_LABEL = {
    resident: "resident reuse",
    donor: "zero-copy donor",
    ram: "RAM tier",
    disk: "SSD tier",
};
// why an elastic admission cleared the slot instead of reusing its KV
// (server_dashboard::clear_code)
const CLEAR_LABEL = {
    empty_slot: "slot was empty",
    family: "parent/child family admission",
    no_lcp: "no shared prefix with resident KV",
    lcp_covers_task: "task ends below the shared prefix",
    mtmd: "multimodal residue",
    cache_reuse_opt: "request set n_cache_reuse",
    no_cache_prompt: "request disabled prompt cache",
    no_pos_min: "sequence had no min position",
    no_covering_checkpoint: "no checkpoint covers the landing",
    rs_window: "tail trim exceeds rollback window",
};
const WHY_LABEL = {
    checkpoint_gap: "checkpoint landing gap",
    reset: "no usable checkpoint — full re-process",
    last_token: "full prefix match — 1 token re-evaluated",
    divergent_tail: "divergent tail (new tokens)",
    no_cache: "no cache entry",
};

function segClass(seg) {
    if (seg.kind === "restore") return SRC_CLASS[seg.src] ?? "c1";
    return SEG_CLASS[seg.kind] ?? "q";
}

function renderTimelineLegend() {
    const legend = $("tl-legend");
    if (legend.childElementCount > 0) return;
    const keys = [
        ["swatch", "var(--queued)", "queued"],
        ["swatch", "var(--cache-0)", "reuse zero-copy"],
        ["swatch", "var(--cache-1)", "restore RAM"],
        ["swatch", "var(--cache-2)", "restore SSD"],
        ["swatch", "var(--pp)", "prefill / recompute"],
        ["swatch", "var(--gen)", "decode"],
    ];
    legend.replaceChildren(...keys.map(([cls, color, label]) => {
        const key = el("span", "key");
        const sw = el("i", cls);
        sw.style.background = color;
        key.append(sw, document.createTextNode(label));
        return key;
    }));
}

function timelineWindow(rows, now) {
    if (state.tlWindow !== "auto" && state.tlWindow !== "all") {
        return { w0: now - Number(state.tlWindow), w1: now };
    }
    let minStart = Infinity;
    const considered = state.tlWindow === "all" ? rows : rows.slice(0, 12);
    for (const r of considered) {
        const seg = timeSegments(r, now);
        if (seg) minStart = Math.min(minStart, seg.start);
    }
    if (!Number.isFinite(minStart)) minStart = now - 60000;
    const span = Math.max(10000, now - minStart);
    return { w0: now - span * 1.02, w1: now };
}

// While a row's content is open, the 500 ms rebuild would move the content DOM
// on every tick — which drops text selection and resets the live mirror's
// scroll. Reading content wins; the row still updates, just not the whole board.
function timelinePaused() {
    if (state.tlExpanded === null) return false;
    return state.reveal.has(state.tlExpanded) || (state.watch !== null && state.watch.req === state.tlExpanded);
}

function renderTimeline({ force = false } = {}) {
    if (!force && timelinePaused()) {
        $("tl-note").textContent = "timeline paused while request #" + state.tlExpanded +
            "'s content is open · close it to resume";
        return;
    }
    renderTimelineLegend();
    const container = $("timeline");
    const rows = state.events.rows().slice(0, TIMELINE_ROWS);
    const now = serverNow();
    const fit = state.tlScale === "fit";
    $("tl-window-group").hidden = fit;
    $("tl-note").textContent = rows.length === 0
        ? (state.streamStatus === "open" ? "waiting for requests…" : "no events yet — stream " + state.streamStatus)
        : rows.length + " requests · " + (state.events.gapped ? "gap in stream (ring overwrote events) · " : "") +
          (fit ? "each bar = that request's own duration · " : "") + "click a row for detail";

    const nodes = [];
    if (fit) {
        // per-request normalized: no shared axis, every bar is full width
        $("tl-axis").replaceChildren(el("span", "tick fit-note", "0%"), (() => {
            const t = el("span", "tick fit-note", "100%");
            t.style.left = "100%";
            t.style.transform = "translateX(-100%)";
            return t;
        })());
        for (const req of rows) {
            const bd = phaseBreakdown(req, now);
            if (!bd) continue;
            nodes.push(timelineRowFit(req, bd));
            if (state.tlExpanded === req.id) nodes.push(timelineDetail(req, timeSegments(req, now)));
        }
    } else {
        const { w0, w1 } = timelineWindow(rows, now);
        renderTimelineAxis(w0, w1, now);
        const x = (t) => Math.max(0, Math.min(1, (t - w0) / (w1 - w0)));
        for (const req of rows) {
            const seg = timeSegments(req, now);
            if (!seg) continue;
            nodes.push(timelineRow(req, seg, x));
            if (state.tlExpanded === req.id) nodes.push(timelineDetail(req, seg));
        }
    }
    if (nodes.length === 0) {
        nodes.push(el("p", "empty", "requests appear here as soon as the server reports them"));
    }
    container.replaceChildren(...nodes);
}

function renderTimelineAxis(w0, w1, now) {
    const axis = $("tl-axis");
    const span = w1 - w0;
    const ticks = [];
    for (const f of [0, 0.25, 0.5, 0.75]) {
        const t = w0 + span * f;
        const tick = el("span", "tick", "-" + formatDuration(now - t));
        tick.style.left = (f * 100).toFixed(2) + "%";
        ticks.push(tick);
    }
    const nowTick = el("span", "tick now", "now");
    nowTick.style.left = "100%";
    ticks.push(nowTick);
    axis.replaceChildren(...ticks);
}

function requestStatus(req) {
    if (req.terminal) {
        const reason = req.terminal.reason;
        if (req.terminal.state === "completed") return { text: req.finished?.stop === "limit" ? "done (limit)" : "done", cls: "ok" };
        if (reason === "run_stall") return { text: "▲ stalled → killed", cls: "err" };
        if (reason === "run_timeout") return { text: "▲ run cap", cls: "err" };
        if (reason === "queue_timeout") return { text: "▲ queue timeout", cls: "err" };
        if (req.terminal.state === "cancelled") return { text: "cancelled", cls: "warn" };
        if (req.terminal.state === "failed") return { text: "▲ failed", cls: "err" };
        return { text: req.terminal.state, cls: "warn" };
    }
    if (req.decode.length > 0 || (req.finished === null && req.promptStart && req.prefill.length > 0 &&
        req.prefill[req.prefill.length - 1].done >= (req.promptStart.nPrompt ?? Infinity))) {
        return { text: "decoding", cls: "ok" };
    }
    if (req.promptStart) return { text: "prefill", cls: "ok" };
    if (req.dispatchedAt) return { text: "starting", cls: "ok" };
    if (req.deferred.length > 0) return { text: "deferred", cls: "warn" };
    return { text: "queued", cls: "ok" };
}

// row identity block, shared by both scales
function timelineWho(req) {
    const who = el("span", "who");
    const line1 = el("span", "l1");
    line1.append(el("b", "", "#" + req.id));
    line1.append(el("span", "lane", (req.lane ?? "—") + (req.slot !== null ? " · s" + req.slot : "")));
    const st = requestStatus(req);
    line1.append(el("span", "st " + st.cls, st.text));
    who.append(line1);
    who.append(el("span", "lane l2", req.promptTokens !== null ? formatTokens(req.promptTokens) + " tok prompt" : ""));
    return who;
}

// duration + rate block, shared by both scales
function timelineMeta(req, durMs) {
    const meta = el("span", "meta");
    const rate = req.finished && req.finished.tgMs > 0 ? req.finished.nDecoded / (req.finished.tgMs / 1000)
        : req.decode.length > 0 ? req.decode[req.decode.length - 1].tps : null;
    meta.append(el("b", "", formatDuration(durMs)));
    meta.append(document.createTextNode(rate ? " · " + formatRate(rate) + " t/s" : ""));
    return meta;
}

// rows are expandable controls: keyboard-reachable, not mouse-only
function expandOnClick(row, req) {
    const toggle = () => {
        const collapsing = state.tlExpanded === req.id;
        state.tlExpanded = collapsing ? null : req.id;
        // collapsing a row drops any content it was showing: a reveal is an
        // action taken for one look, not a setting that stays on
        if (collapsing) hideReveal(req);
        renderTimeline({ force: true });
        const next = $("timeline").querySelector('[data-req="' + req.id + '"]');
        if (next) next.focus();
    };
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-expanded", state.tlExpanded === req.id ? "true" : "false");
    row.setAttribute("aria-label", "request " + req.id + " timeline detail");
    row.addEventListener("click", toggle);
    row.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            toggle();
        }
    });
}

function timelineRow(req, seg, x) {
    const row = el("div", "tl-row");
    row.dataset.req = req.id;
    row.append(timelineWho(req));

    const track = el("span", "tl-track");
    track.append(el("i", "lane-line"));
    for (let i = 0; i < seg.segments.length; i++) {
        const s = seg.segments[i];
        const left = x(s.t0);
        const right = x(s.t1);
        if (right <= 0) continue;
        const node = el("i", "tl-seg " + segClass(s));
        node.style.left = (left * 100).toFixed(3) + "%";
        node.style.width = Math.max(0.15, (right - left) * 100).toFixed(3) + "%";
        if (seg.live && i === seg.segments.length - 1) node.classList.add("live-edge");
        attachSegmentHover(node, req, s);
        track.append(node);
    }
    row.append(track);
    row.append(timelineMeta(req, seg.end - seg.start));
    expandOnClick(row, req);
    return row;
}

// "per request" scale: the request's own duration fills the track, so phase
// composition is equally legible for a 2 s completion and a 40 min agent turn.
// Magnitude is not lost — it is the number in the meta column.
function timelineRowFit(req, bd) {
    const row = el("div", "tl-row");
    row.dataset.req = req.id;
    row.append(timelineWho(req));

    const track = el("span", "tl-track fit");
    for (const p of bd.phases) {
        const node = el("i", "tl-seg " + segClass(p));
        node.style.width = (p.frac * 100).toFixed(3) + "%";
        // a phase worth >=12% of the bar names itself; the rest rely on hover
        if (p.frac >= 0.12) {
            const label = el("em", "", p.kind === "restore" ? (SRC_LABEL[p.src] ?? "setup") : p.kind);
            node.append(label);
        }
        attachSegmentHover(node, req, { kind: p.kind, src: p.src, t0: 0, t1: p.ms });
        track.append(node);
    }
    if (bd.live) track.classList.add("live-edge");
    row.append(track);
    row.append(timelineMeta(req, bd.total));
    expandOnClick(row, req);
    return row;
}

function segmentTooltip(req, s) {
    const dur = formatDuration(s.t1 - s.t0);
    switch (s.kind) {
        case "queued": {
            const why = req.deferred.length > 0
                ? "deferred ×" + req.deferred.length + " (" + [...new Set(req.deferred.map((d) => d.why))].join(", ") + ")"
                : "waiting for dispatch";
            return [["queued", dur], [why, ""]];
        }
        case "restore": {
            const r = req.restore;
            const lines = [["slot setup + cache consult", dur]];
            if (r && r.src !== "miss") {
                lines.push([SRC_LABEL[r.src] ?? r.src,
                    formatTokens(r.tokens) + " tok" + (r.bytes > 0 ? " · " + formatBytes(r.bytes) : "") + " · " + r.ms.toFixed(1) + " ms"]);
                if (r.src === "donor" && r.donorSlot !== null) lines.push(["donor slot " + r.donorSlot, ""]);
            } else {
                lines.push(["cache miss", "nothing reusable"]);
            }
            return lines;
        }
        case "prefill": {
            const ps = req.promptStart;
            const done = req.prefill.length > 0 ? req.prefill[req.prefill.length - 1].done : ps?.nPast ?? 0;
            const total = ps?.nPrompt ?? 0;
            const newTokens = total - (ps?.nPast ?? 0);
            const rate = s.t1 > s.t0 ? (done - (ps?.nPast ?? 0)) / ((s.t1 - s.t0) / 1000) : 0;
            return [["prefill", dur],
                [formatTokens(done) + "/" + formatTokens(total) + " in KV", formatTokens(newTokens) + " to compute · " + req.prefill.length + " chunks"],
                [formatRate(rate) + " tok/s", ""]];
        }
        case "decode": {
            const f = req.finished;
            const last = req.decode[req.decode.length - 1];
            const n = f?.nDecoded ?? last?.n ?? 0;
            const draftN = f?.draftN ?? last?.draftN ?? 0;
            const draftA = f?.draftA ?? last?.draftA ?? 0;
            const lines = [["decode", dur], [formatTokens(n) + " tokens", ""]];
            if (draftN > 0) lines.push(["draft acceptance", formatPercent(draftA / draftN) + " (" + draftA + "/" + draftN + ")"]);
            return lines;
        }
    }
    return [[s.kind, dur]];
}

function attachSegmentHover(node, req, s) {
    const tooltip = $("tooltip");
    node.addEventListener("pointermove", (ev) => {
        const lines = segmentTooltip(req, s);
        tooltip.hidden = false;
        tooltip.replaceChildren(...lines.flatMap(([a, b], i) => {
            const parts = [];
            if (i > 0) parts.push(el("br"));
            parts.push(el("b", "", a));
            if (b) parts.push(document.createTextNode(" · " + b));
            return parts;
        }));
        tooltip.style.left = Math.min(ev.clientX + 12, window.innerWidth - tooltip.offsetWidth - 8) + "px";
        tooltip.style.top = (ev.clientY + 14) + "px";
        ev.stopPropagation();
    });
    node.addEventListener("pointerleave", () => { tooltip.hidden = true; });
}

function timelineDetail(req, seg) {
    const box = el("div", "tl-detail");

    // facts line
    box.append(el("h3", "", "request #" + req.id));
    const facts = el("div", "facts");
    const fact = (label, value) => {
        const span = el("span", "");
        span.append(el("span", "u", label + " "), el("b", "", value));
        return span;
    };
    facts.append(fact("lane", req.lane ?? "—"));
    if (req.slot !== null) facts.append(fact("slot", String(req.slot)));
    if (req.waitMs !== null) facts.append(fact("queue wait", formatDuration(req.waitMs)));
    if (req.select) {
        facts.append(fact("slot pick", req.select.sel + (req.select.sim ? " (" + formatPercent(req.select.sim) + " match)" : "")));
    }
    if (req.admission) {
        const why = req.admission.why;
        facts.append(fact("admission", req.admission.outcome +
            (why ? " · " + (CLEAR_LABEL[why] ?? why) : "")));
    }
    if (req.finished) {
        facts.append(fact("stop", req.finished.stop));
        facts.append(fact("prefill", formatDuration(req.finished.ppMs)));
        facts.append(fact("decode", formatDuration(req.finished.tgMs) +
            " (" + formatRate(req.finished.nDecoded / Math.max(0.001, req.finished.tgMs / 1000)) + " t/s)"));
        if (req.finished.draftN > 0) {
            facts.append(fact("draft accept", formatPercent(req.finished.draftA / req.finished.draftN)));
        }
    } else if (req.terminal) {
        facts.append(fact("terminal", req.terminal.state + " · " + req.terminal.reason));
    }
    box.append(facts);

    // work the server paid for and then discarded
    const wasted = wastedRestore(req);
    if (wasted) {
        const warn = el("p", "warn-line");
        warn.append(el("b", "", "▲ restore discarded"));
        warn.append(document.createTextNode(
            " — the cache restored " + formatTokens(wasted.tokens) + " tokens" +
            (wasted.bytes > 0 ? " (" + formatBytes(wasted.bytes) + ")" : "") +
            (wasted.ms > 0 ? " in " + wasted.ms.toFixed(0) + " ms" : "") +
            ", then the elastic admission cleared the slot" +
            (wasted.why ? " — " + (CLEAR_LABEL[wasted.why] ?? wasted.why) : "") +
            ", so the whole prompt was recomputed."));
        box.append(warn);
    }

    // where the request's own time went, normalized to its duration so the
    // breakdown reads the same at 2 s and at 40 min
    const bd = phaseBreakdown(req, serverNow());
    if (bd) {
        box.append(el("h3", "", "time · " + formatDuration(bd.total) + " total — where it went"));
        const bar = el("div", "phase-bar");
        for (const p of bd.phases) {
            const i = el("i", segClass(p));
            i.style.width = (p.frac * 100).toFixed(2) + "%";
            i.title = (p.kind === "restore" ? (SRC_LABEL[p.src] ?? "slot setup") : p.kind) +
                " · " + formatDuration(p.ms) + " · " + formatPercent(p.frac);
            bar.append(i);
        }
        box.append(bar);
        const legend = el("div", "phase-list");
        for (const p of bd.phases) {
            const line = el("span", "");
            const sw = el("i", "swatch " + segClass(p));
            line.append(sw, document.createTextNode(" "));
            line.append(el("b", "", p.kind === "restore" ? (SRC_LABEL[p.src] ?? "slot setup") : p.kind));
            line.append(document.createTextNode(" " + formatDuration(p.ms) + " (" + formatPercent(p.frac) + ")"));
            legend.append(line);
        }
        box.append(legend);
    }

    // prompt composition: restored vs recomputed token spans
    const spans = tokenSpans(req);
    if (spans.length > 0) {
        const ps = req.promptStart;
        box.append(el("h3", "", "prompt · " + formatTokens(ps.nPrompt) + " tokens — where they came from"));
        const bar = el("div", "span-bar");
        for (const s of spans) {
            const i = el("i", s.kind === "restored" ? (SRC_CLASS[s.src] ?? "c1") : "rc");
            i.style.width = ((s.tokens / ps.nPrompt) * 100).toFixed(2) + "%";
            i.title = spanText(s);
            bar.append(i);
        }
        box.append(bar);
        const list = el("div", "span-list");
        for (const s of spans) {
            const line = el("div", "");
            const sw = el("i", "swatch");
            sw.style.background = s.kind === "restored" ? "var(--" + ({ c0: "cache-0", c1: "cache-1", c2: "cache-2" })[SRC_CLASS[s.src] ?? "c1"] + ")" : "var(--pp)";
            line.append(sw, document.createTextNode(" "));
            line.append(el("b", "", formatTokens(s.tokens) + " tok"));
            line.append(document.createTextNode(" [" + s.start + ", " + s.end + ") — " + spanText(s)));
            list.append(line);
        }
        // restore cost detail
        if (req.restore && req.restore.src !== "miss" && req.restore.ms > 0.05) {
            const line = el("div", "");
            line.append(document.createTextNode("restore cost: " + req.restore.ms.toFixed(1) + " ms" +
                (req.restore.bytes > 0 ? " · " + formatBytes(req.restore.bytes) + " state" : "") +
                (req.restore.src === "disk" ? " (includes SSD load)" : "")));
            list.append(line);
        }
        box.append(list);
    }

    // prefill progress curve. A sparkline carries shape, not values, so the
    // heading states the endpoints (the hover layer gives any single point).
    if (req.prefill.length > 1) {
        const last = req.prefill[req.prefill.length - 1];
        box.append(el("h3", "", "prefill progress · " + req.prefill.length + " ubatches · " +
            "0 → " + formatTokens(last.done) + " tok over " + formatDuration(last.ms)));
        box.append(miniChart(req.prefill.map((p) => ({ t: p.ms, v: p.done })), "pp",
            (p) => formatTokens(p.v) + " tok @ " + formatDuration(p.t)));
    }

    // what the request actually said and what came back. Hidden until asked
    // for; see contentSection().
    box.append(el("h3", "", "content · prompt in / generation out"));
    box.append(contentSection(req));

    // decode rate curve
    const decodePoints = req.decode.filter((d) => d.tps !== null);
    if (decodePoints.length > 1) {
        const rates = decodePoints.map((d) => d.tps);
        const lo = Math.min(...rates);
        const hi = Math.max(...rates);
        box.append(el("h3", "", "decode rate · " + formatRate(lo) + "–" + formatRate(hi) + " tok/s"));
        box.append(miniChart(decodePoints.map((d) => ({ t: d.ms, v: d.tps })), "gen",
            (p) => formatRate(p.v) + " tok/s @ " + formatDuration(p.t)));
    }

    box.addEventListener("click", (ev) => ev.stopPropagation());
    return box;
}

function spanText(s) {
    if (s.kind === "restored") return "restored from " + (SRC_LABEL[s.src] ?? s.src);
    return "recomputed — " + (WHY_LABEL[s.why] ?? s.why);
}

// ---------------------------------------------------------------------------
// Content (v3): prompt in / generation out, revealed only on request
//
// The three rules this section implements:
//   1. nothing renders content until a human presses a button for THAT request;
//   2. whatever is on screen says how much of the real thing it is, so a capped
//      preview can never read as the complete prompt;
//   3. reveals are memory-only, capped, and dropped when the row collapses.
// ---------------------------------------------------------------------------

function contentDisabled() {
    return state.contentEnabled === false;
}

function pruneReveals() {
    while (state.reveal.size > MAX_REVEALS) {
        const oldest = state.reveal.keys().next().value;
        state.reveal.delete(oldest);
        state.contentNodes.delete(oldest);
    }
}

function requestReveal(req) {
    if (state.reveal.has(req.id)) return;
    state.reveal.set(req.id, { status: "loading" });
    pruneReveals();
    repaintContent(req);
    state.transport.content(req.id).then((raw) => {
        const view = parseContent(raw);
        state.reveal.set(req.id, view
            ? { status: "ok", view }
            : { status: "error", error: "unrecognized content response" });
    }).catch((error) => {
        const message = error.status === 501
            ? "content disclosure is disabled on this server"
            : error.status === 401 || error.status === 403
                ? "not authorized to read content"
                : "content unavailable (" + (error.status ?? error.message) + ")";
        if (error.status === 501) state.contentEnabled = false;
        state.reveal.set(req.id, { status: "error", error: message });
    }).finally(() => repaintContent(req));
}

function hideReveal(req) {
    if (state.watch && state.watch.req === req.id) stopWatch();
    state.reveal.delete(req.id);
    repaintContent(req);
}

// A signature of the block's STRUCTURE — deliberately not of the live text.
// Rebuilding an 8 KB <pre> on every arriving token would fight the operator for
// text selection and scroll position, so a token arrival updates the live pane
// in place (updateWatchPane) and only a structural change rebuilds.
function contentRevision(req) {
    const entry = state.reveal.get(req.id);
    const watching = state.watch && state.watch.req === req.id ? state.watch : null;
    return [
        entry?.status ?? "-",
        entry?.view?.found ?? "-",
        watching ? "w" : "-",
        watching?.state?.ended ?? "-",
        state.contentEnabled,
    ].join("|");
}

function repaintContent(req) {
    const node = state.contentNodes.get(req.id);
    if (node) {
        const rev = contentRevision(req);
        if (node.dataset.rev === rev && node.dataset.built === "1") {
            const watching = state.watch && state.watch.req === req.id ? state.watch : null;
            if (watching) updateWatchPane(watching);
        } else {
            node.dataset.rev = rev;
            renderContentBlock(req, node);
        }
    }
    renderStatus();
}

// ---- live token mirror ----

function startWatch(req) {
    stopWatch();
    const entry = { req: req.id, stream: null, state: null, pinned: true };
    entry.stream = new WatchStream({
        req: req.id,
        apiKey: currentApiKey(),
        onUpdate: (s) => {
            entry.state = s;
            repaintContent(req);
        },
        onClose: () => {
            // leave the last frame on screen; only the arming is released
            repaintContent(req);
        },
    });
    state.watch = entry;
    entry.stream.start();
    repaintContent(req);
}

function stopWatch() {
    if (!state.watch) return;
    const open = state.watch;
    state.watch = null;
    open.stream.stop();
}

function watchable(req) {
    return state.transport.mode === "live" && req.terminal === null && !contentDisabled();
}

// ---- rendering ----

// How many lines of a pane are shown before it is clamped. The clamp is
// decided from the TEXT, not from measured layout, so it is deterministic:
// there is always either the whole pane or a visible "show all" control, and
// never a silently-scrolled pane that looks complete.
const PANE_CLAMP_LINES = 14;
const PANE_CLAMP_CHARS = 1100;

function textPane(value, { clamp = true } = {}) {
    const pre = el("pre", "content-pre");
    pre.textContent = value;
    const lines = value.split("\n").length;
    if (!clamp || (lines <= PANE_CLAMP_LINES && value.length <= PANE_CLAMP_CHARS)) {
        return pre;
    }
    const wrap = el("div", "pane-wrap");
    pre.classList.add("clamped");
    const more = el("button", "pane-more", "show all " + lines.toLocaleString("en-US") + " lines");
    more.type = "button";
    more.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const open = pre.classList.toggle("clamped");
        more.textContent = open
            ? "show all " + lines.toLocaleString("en-US") + " lines"
            : "collapse";
    });
    wrap.append(pre, more);
    return wrap;
}

function elisionRule(text) {
    const rule = el("div", "elision");
    rule.append(el("span", "", "⋯ " + text + " ⋯"));
    return rule;
}

function sideBlock(side, { title, cls, extent = null, liveTag = null, actions = null, autoscroll = false }) {
    const box = el("div", "content-side " + cls);
    const head = el("div", "cs-head");
    head.append(el("b", "", title));
    head.append(el("span", "cs-extent", extent ?? extentLabel(side)));
    if (liveTag) head.append(liveTag);
    if (actions) head.append(actions);
    box.append(head);

    if (!side || side.empty) {
        box.append(el("p", "cs-empty", side && side.nTokens === 0
            ? "empty" : "nothing captured for this side"));
        return box;
    }
    const body = el("div", "cs-body" + (autoscroll ? " scroll" : ""));
    body.append(textPane(side.head, { clamp: !autoscroll }));
    if (side.truncated) {
        body.append(elisionRule(side.kind === "input"
            ? side.elidedTokens.toLocaleString("en-US") + " tokens elided"
            : side.elidedBytes.toLocaleString("en-US") + " bytes elided"));
        if (side.tail) body.append(textPane(side.tail));
    }
    box.append(body);
    return box;
}

function renderContentBlock(req, node) {
    const entry = state.reveal.get(req.id);
    const watching = state.watch && state.watch.req === req.id ? state.watch : null;

    if (!entry && !watching) {
        node.replaceChildren(closedContentPrompt(req));
        return;
    }

    const nodes = [];
    const bar = el("div", "content-bar");
    // the flag marks an actual disclosure, so it stays off when the answer was
    // "nothing retained" — otherwise it cries wolf and stops meaning anything
    const showing = (entry?.status === "ok" && entry.view?.found) ||
        (watching !== null && (watching.state?.text?.length ?? 0) > 0);
    bar.append(showing
        ? el("span", "reveal-flag", "content shown")
        : el("span", "dim", "no content shown"));
    const hide = el("button", "", "hide");
    hide.type = "button";
    hide.addEventListener("click", (ev) => { ev.stopPropagation(); hideReveal(req); });
    bar.append(hide);
    if (watchable(req)) {
        const w = el("button", "", watching && !watching.state?.ended ? "stop watching" : "watch live");
        w.type = "button";
        w.addEventListener("click", (ev) => {
            ev.stopPropagation();
            if (watching && !watching.state?.ended) stopWatch(); else startWatch(req);
            repaintContent(req);
        });
        bar.append(w);
    }
    bar.append(el("span", "content-policy dim",
        "fetched now for this request only · not logged, not in the event stream"));
    nodes.push(bar);

    if (entry?.status === "loading") {
        nodes.push(el("p", "cs-empty", "loading…"));
    } else if (entry?.status === "error") {
        nodes.push(el("p", "warn-line", entry.error));
    } else if (entry?.status === "ok") {
        const view = entry.view;
        if (!view.found) {
            nodes.push(el("p", "cs-empty", view.reason === "evicted"
                ? "no longer retained — the content store evicted this request (cap " +
                  (view.caps.maxRequests ?? "?") + " requests / " + formatBytes(view.caps.maxBytes) + ")"
                : "nothing captured for this request — it never reached prefill, or the server restarted"));
        } else {
            nodes.push(sideBlock(view.input, { title: "INPUT · prompt as sent", cls: "in" }));
            if (view.output) {
                nodes.push(sideBlock(view.output, { title: "OUTPUT · generated", cls: "out" }));
            } else if (!watching) {
                const live = el("p", "cs-empty",
                    "still generating — the final text is captured when the request completes.");
                if (watchable(req)) live.append(document.createTextNode(" Use “watch live” to follow it token by token."));
                nodes.push(live);
            }
        }
        nodes.push(el("p", "content-policy dim",
            "server retains at most " + (view.caps.maxRequests ?? "?") + " requests / " +
            formatBytes(view.caps.maxBytes) + " — currently " + view.store.requests + " (" +
            formatBytes(view.store.bytes) + "), " + view.store.evicted + " evicted"));
    }

    if (watching) nodes.push(watchPane(watching));
    node.replaceChildren(...nodes);
    node.dataset.built = "1";
    if (watching) updateWatchPane(watching); // now that the pane is in the document
}

// How much of a live generation is laid out at once. The accumulated text is
// already capped in content.mjs; this is a second, tighter bound so a 45-minute
// turn does not re-layout 256 KiB of <pre> on every frame.
const WATCH_RENDER_BYTES = 24 * 1024;

// Built once per watch, then mutated in place as tokens arrive.
function watchPane(watching) {
    const box = el("div", "content-side out watching");
    const head = el("div", "cs-head");
    head.append(el("b", "", "OUTPUT · live"));
    const extent = el("span", "cs-extent");
    const tag = el("span", "live-tag on", "connecting");
    head.append(extent, tag);
    box.append(head);

    const body = el("div", "cs-body scroll");
    const pre = el("pre", "content-pre");
    const caret = el("i", "caret");
    body.append(pre);
    box.append(body);
    body.addEventListener("scroll", () => {
        watching.pinned = body.scrollHeight - body.scrollTop - body.clientHeight < 40;
    });

    watching.pane = { box, extent, tag, body, pre, caret };
    updateWatchPane(watching);
    return box;
}

function updateWatchPane(watching) {
    const pane = watching?.pane;
    if (!pane) return;
    const s = watching.state;
    const full = s?.text ?? "";
    const shown = full.length > WATCH_RENDER_BYTES ? full.slice(-WATCH_RENDER_BYTES) : full;

    // A live pane must never claim to be complete: it says what it has and,
    // when the stream outran the mirror, that earlier output is simply gone.
    const trimmed = (s?.dropped ?? 0) + (s?.clientDropped ?? 0) + (full.length - shown.length);
    pane.extent.textContent = (s?.nDec ?? 0).toLocaleString("en-US") + " tokens · " +
        formatBytes(s?.cursor ?? 0) + " generated" +
        (trimmed > 0 ? " · newest " + formatBytes(shown.length) + " shown, " +
            formatBytes(trimmed) + " earlier output dropped" : "");
    pane.tag.className = "live-tag " + (s?.ended ? "done" : "on");
    pane.tag.textContent = s?.ended
        ? (WATCH_REASON_LABEL[s.reason] ?? s.reason ?? "closed")
        : (s?.frames ?? 0) > 0 ? "streaming" : "connecting";

    pane.pre.textContent = shown;
    if (!s?.ended) pane.pre.append(pane.caret);

    // stay on the newest token unless the operator scrolled away from it
    if (watching.pinned !== false) pane.body.scrollTop = pane.body.scrollHeight;
}

function closedContentPrompt(req) {
    const p = el("p", "content-closed");
    if (contentDisabled()) {
        p.append(el("span", "dim",
            "content disclosure is disabled on this server (--no-dashboard-content) — metadata only"));
        return p;
    }
    if (state.transport.mode !== "live" && !state.fixtureContent) {
        p.append(el("span", "dim", "no content source in this fixture"));
        return p;
    }
    const btn = el("button", "reveal-btn", "show prompt & output");
    btn.type = "button";
    btn.addEventListener("click", (ev) => { ev.stopPropagation(); requestReveal(req); });
    p.append(btn);
    if (watchable(req)) {
        const w = el("button", "reveal-btn", "watch live");
        w.type = "button";
        w.addEventListener("click", (ev) => { ev.stopPropagation(); startWatch(req); });
        p.append(w);
    }
    p.append(el("span", "dim",
        "prompt and generated text are hidden by default — fetched only when you ask, for this request"));
    return p;
}

function contentSection(req) {
    let node = state.contentNodes.get(req.id);
    if (!node) {
        node = el("div", "content-block");
        node.dataset.rev = "";
        state.contentNodes.set(req.id, node);
    }
    const rev = contentRevision(req);
    if (node.dataset.rev !== rev || node.dataset.built !== "1") {
        node.dataset.rev = rev;
        renderContentBlock(req, node);
    }
    return node;
}

// tiny line chart over {t, v} points with hover; x spans [0, max t].
// `domainMax` pins the y-domain: a ratio series must be drawn against its true
// [0,1] range, otherwise a steady 67 % auto-scales to a full-height block and
// reads as "100 %".
function miniChart(points, cls, describe, { domainMax = null } = {}) {
    const W = 600, H = 40, PAD = 2;
    const svg = miniSvg(null, { width: W, height: H });
    svg.setAttribute("class", "mini " + cls);
    const maxT = points[points.length - 1].t || 1;
    const maxV = domainMax ?? (Math.max(...points.map((p) => p.v)) || 1);
    const px = (p) => PAD + (p.t / maxT) * (W - 2 * PAD);
    const py = (p) => H - PAD - (p.v / maxV) * (H - 2 * PAD);
    let d = "";
    for (let i = 0; i < points.length; i++) {
        d += (i === 0 ? "M" : "L") + px(points[i]).toFixed(1) + " " + py(points[i]).toFixed(1);
    }
    const area = document.createElementNS("http://www.w3.org/2000/svg", "path");
    area.setAttribute("d", d + "L" + px(points[points.length - 1]).toFixed(1) + " " + (H - PAD) + "L" + PAD + " " + (H - PAD) + "Z");
    area.setAttribute("class", "spark-area");
    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    line.setAttribute("d", d);
    line.setAttribute("class", "spark-line");
    svg.append(area, line);

    const tooltip = $("tooltip");
    svg.addEventListener("pointermove", (ev) => {
        const rect = svg.getBoundingClientRect();
        const tx = ((ev.clientX - rect.left) / rect.width) * W;
        let best = points[0];
        for (const p of points) if (Math.abs(px(p) - tx) < Math.abs(px(best) - tx)) best = p;
        tooltip.hidden = false;
        tooltip.replaceChildren(el("b", "", describe(best)));
        tooltip.style.left = Math.min(ev.clientX + 12, window.innerWidth - tooltip.offsetWidth - 8) + "px";
        tooltip.style.top = (ev.clientY + 14) + "px";
    });
    svg.addEventListener("pointerleave", () => { tooltip.hidden = true; });
    return svg;
}

// periodic timeline repaint while visible: advances the "now" edge for live
// rows and folds any batched SSE arrivals (500 ms; cheap DOM rebuild of ≤30 rows)
function timelineTick() {
    clearTimeout(state.tlTimer);
    if (state.view === "timeline" && !document.hidden) {
        renderTimeline();
        state.tlDirty = false;
    }
    state.tlTimer = setTimeout(timelineTick, 500);
}

// ---------------------------------------------------------------------------
// Cache view
// ---------------------------------------------------------------------------

function scheduleCachePoll(delay = CACHE_POLL_MS) {
    clearTimeout(state.cacheTimer);
    state.cacheTimer = setTimeout(pollCacheState, delay);
}

async function pollCacheState() {
    if (state.view !== "cache") return;
    try {
        const raw = await state.transport.cacheState();
        state.cacheState = cacheView(raw);
    } catch (error) {
        if (error.status === 401) { showKeyForm("this server requires an API key"); return; }
        state.cacheState = state.cacheState ?? null;
    }
    renderCache();
    scheduleCachePoll();
}

function renderCache() {
    const tiles = $("cache-tiles");
    const view = state.cacheState;
    if (!view) {
        tiles.replaceChildren(tile("", "Prompt cache", "—", "", "no cache-state yet — endpoint unreachable or cache disabled"));
        $("cache-body").replaceChildren();
        return;
    }
    const nodes = [];

    const ramTile = tile("", "RAM tier", formatBytes(view.ram.used),
        view.ram.limit > 0 ? " / " + formatBytes(view.ram.limit) : "",
        formatTokens(view.ram.tokens) + " cached tokens" +
        (view.ram.fraction !== null ? " · " + formatPercent(view.ram.fraction) + " full" : ""));
    const ramBar = el("div", "tier-bar ram");
    const ramFill = el("i");
    ramFill.style.width = ((view.ram.fraction ?? 0) * 100).toFixed(1) + "%";
    ramBar.append(ramFill);
    ramTile.append(ramBar);
    nodes.push(ramTile);

    const diskTile = tile("", "SSD tier", formatBytes(view.disk.used),
        view.disk.limit > 0 ? " / " + formatBytes(view.disk.limit) : "",
        view.entries.filter((e) => e.tier === "disk").length + " spilled entries" +
        (view.disk.fraction !== null ? " · " + formatPercent(view.disk.fraction) + " full" : ""));
    const diskBar = el("div", "tier-bar disk");
    const diskFill = el("i");
    diskFill.style.width = ((view.disk.fraction ?? 0) * 100).toFixed(1) + "%";
    diskBar.append(diskFill);
    diskTile.append(diskBar);
    nodes.push(diskTile);

    // hit rate with rolling sparkline from the event stream's consults
    const rateTile = tile("cache-rate", "Hit rate",
        view.hitRate !== null ? formatPercent(view.hitRate) : "—", "",
        (view.counters.lookups ?? 0) + " consults · " + (view.counters.hits_entry ?? 0) + " entry hits · " +
        (view.counters.hits_resident ?? 0) + " resident");
    const series = hitRateSeries(state.events.lookups);
    if (series.length > 1) {
        // drawn against a fixed 0–100 % box so the line's height is the rate
        const frame = el("div", "rate-frame");
        frame.append(el("span", "cap", "100%"));
        frame.append(miniChart(series.map((p, i) => ({ t: i, v: p.v })), "cache-rate",
            (p) => formatPercent(p.v) + " rolling hit rate (20-consult window)",
            { domainMax: 1 }));
        rateTile.append(frame);
    }
    nodes.push(rateTile);

    nodes.push(tile("", "Tier traffic",
        String((view.counters.spills ?? 0) + (view.counters.disk_loads ?? 0)), " ops",
        "spills " + (view.counters.spills ?? 0) + " (" + formatBytes(view.counters.bytes_spilled ?? 0) + ")" +
        " · loads " + (view.counters.disk_loads ?? 0) + " (" + formatBytes(view.counters.bytes_disk_load ?? 0) + ")" +
        " · drops " + (view.counters.drops ?? 0)));

    tiles.replaceChildren(...nodes);

    // entries table
    $("cache-note").textContent = "· " + view.entries.length + " entries, snapshot " +
        (view.ageMs < 2000 ? "fresh" : formatDuration(view.ageMs) + " old");
    const body = $("cache-body");
    const rows = [];
    for (const e of [...view.entries].sort(
            (a, b) => (a.tier === b.tier ? a.ageS - b.ageS : a.tier === "ram" ? -1 : 1))) {
        const tr = el("tr", state.cacheExpanded === e.id ? "expanded" : "");
        tr.append(el("td", "num", String(e.id)));
        const tierTd = el("td", "");
        tierTd.append(el("span", "tier-tag " + e.tier, e.tier));
        tr.append(tierTd);
        tr.append(el("td", "num", formatTokens(e.tokens)));
        tr.append(el("td", "num", formatBytes(e.bytes)));
        tr.append(el("td", "num", formatDuration(e.ageS * 1000)));
        tr.append(el("td", "num", e.lastHitS !== null ? formatDuration(e.lastHitS * 1000) + " ago" : "never"));
        tr.append(el("td", "num" + (e.hits > 0 ? " strong" : ""), String(e.hits)));
        tr.append(el("td", "num", e.req !== null ? "#" + e.req : "—"));
        const fileTd = el("td", "");
        if (e.file) fileTd.append(el("code", "", e.file));
        else fileTd.textContent = "—";
        tr.append(fileTd);
        if (e.preview) {
            // rows that hold previewable text are real controls, not decoration
            tr.classList.add("clickable");
            tr.tabIndex = 0;
            tr.setAttribute("role", "button");
            tr.setAttribute("aria-expanded", state.cacheExpanded === e.id ? "true" : "false");
            tr.setAttribute("aria-label", "cache entry " + e.id + " contents");
            const toggle = () => {
                state.cacheExpanded = state.cacheExpanded === e.id ? null : e.id;
                renderCache();
            };
            tr.addEventListener("click", toggle);
            tr.addEventListener("keydown", (ev) => {
                if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); toggle(); }
            });
        }
        rows.push(tr);
        if (state.cacheExpanded === e.id) rows.push(cacheDetailRow(e));
    }
    body.replaceChildren(...rows);
    if (rows.length === 0) {
        const tr = el("tr");
        const td = el("td", "empty", view.enabled ? "cache is empty" : "prompt cache is disabled on this server");
        td.colSpan = 9;
        tr.append(td);
        body.append(tr);
    }
    renderContentFlag();
}

// ---- cache entry preview ----

function requestCachePreview(id) {
    if (state.cachePreview.has(id)) return;
    state.cachePreview.set(id, { status: "loading" });
    while (state.cachePreview.size > MAX_CACHE_PREVIEWS) {
        state.cachePreview.delete(state.cachePreview.keys().next().value);
    }
    renderCache();
    state.transport.cachePreview(id).then((raw) => {
        const view = parseCachePreview(raw);
        state.cachePreview.set(id, view
            ? { status: "ok", view }
            : { status: "error", error: "unrecognized preview response" });
    }).catch((error) => {
        if (error.status === 501) state.contentEnabled = false;
        state.cachePreview.set(id, { status: "error", error: error.status === 501
            ? "content disclosure is disabled on this server"
            : "preview unavailable (" + (error.status ?? error.message) + ")" });
    }).finally(() => renderCache());
}

function cacheDetailRow(entry) {
    const tr = el("tr", "cache-detail-row");
    const td = el("td", "");
    td.colSpan = 9;
    const box = el("div", "cache-detail");

    const head = el("div", "cs-head");
    head.append(el("b", "", "entry " + entry.id));
    head.append(el("span", "cs-extent",
        formatTokens(entry.tokens) + " tokens · " + entry.tier + " tier · " + formatBytes(entry.bytes) +
        (entry.req !== null ? " · saved by request #" + entry.req : " · origin unknown (rebuilt from disk)")));
    box.append(head);

    const state_ = state.cachePreview.get(entry.id);
    if (!state_) {
        const p = el("p", "content-closed");
        if (contentDisabled()) {
            p.append(el("span", "dim",
                "content disclosure is disabled on this server (--no-dashboard-content)"));
        } else {
            const btn = el("button", "reveal-btn", "show cached text");
            btn.type = "button";
            btn.addEventListener("click", (ev) => { ev.stopPropagation(); requestCachePreview(entry.id); });
            p.append(btn);
            p.append(el("span", "dim",
                "the entry's own tokens, decoded on demand — first " +
                "and last few hundred characters only"));
        }
        box.append(p);
    } else if (state_.status === "loading") {
        box.append(el("p", "cs-empty", "loading…"));
    } else if (state_.status === "error") {
        box.append(el("p", "warn-line", state_.error));
    } else if (!state_.view.found) {
        box.append(el("p", "cs-empty", "entry is gone — evicted or spilled since the snapshot"));
    } else {
        const p = state_.view;
        const bar = el("div", "content-bar");
        bar.append(el("span", "reveal-flag", "content shown"));
        const hide = el("button", "", "hide");
        hide.type = "button";
        hide.addEventListener("click", (ev) => {
            ev.stopPropagation();
            state.cachePreview.delete(entry.id);
            renderCache();
        });
        bar.append(hide);
        bar.append(el("span", "content-policy dim", cachePreviewLabel(p)));
        box.append(bar);

        const body = el("div", "cs-body");
        body.append(textPane(p.head));
        if (p.truncated) {
            body.append(elisionRule(p.elidedTokens.toLocaleString("en-US") + " tokens elided"));
            if (p.tail) body.append(textPane(p.tail));
        }
        box.append(body);
    }

    box.addEventListener("click", (ev) => ev.stopPropagation());
    td.append(box);
    tr.append(td);
    return tr;
}

// ---------------------------------------------------------------------------
// View switching
// ---------------------------------------------------------------------------

function setView(view) {
    state.view = view;
    for (const btn of $("tabs").querySelectorAll("button")) {
        btn.classList.toggle("active", btn.dataset.view === view);
    }
    $("view-live").hidden = view !== "live";
    $("view-timeline").hidden = view !== "timeline";
    $("view-cache").hidden = view !== "cache";
    // keep #key= while recording the view in the hash
    const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
    if (view === "live") hash.delete("view");
    else hash.set("view", view);
    const encoded = hash.toString();
    history.replaceState(null, "", encoded ? "#" + encoded : location.pathname + location.search);
    if (view === "cache") void pollCacheState();
    render();
}

// ---------------------------------------------------------------------------
// API key form + boot
// ---------------------------------------------------------------------------

function showKeyForm(reason) {
    const form = $("key-form");
    form.hidden = false;
    $("status").className = "status warn";
    $("status").textContent = reason;
    $("key-input").focus();
}

async function boot() {
    const params = new URLSearchParams(location.search);

    // ?theme=dark|light pins the palette (wall displays, and it makes both
    // modes reproducible in screenshot tests); default follows the OS
    const theme = params.get("theme");
    if (theme === "dark" || theme === "light") {
        document.documentElement.dataset.theme = theme;
    }

    const fixture = params.get("fixture");
    if (fixture !== null && fixture !== "0") {
        const name = fixture === "demo" || fixture === "v2" ? "demo" : "replay";
        const replay = await (await fetch("fixtures/" + name + ".json", { cache: "no-store" })).json();
        // pre-warm: feed all but the last few ticks at their recorded cadence so
        // sparklines and the request tail are populated immediately
        const interval = replay.interval_ms ?? POLL_MS;
        const warmCount = Math.max(0, replay.ticks.length - 5);
        const now = performance.now();
        for (let i = 0; i < warmCount; i++) {
            const tick = replay.ticks[i];
            state.tracker.feed(now - (warmCount - i) * interval, tick.slots);
        }
        state.transport = fixtureTransport(replay, warmCount);
        if (fixture === "v2" || fixture === "1" || fixture === "replay") {
            // event-stream fixture for the timeline + cache views
            try {
                const ev = await (await fetch("fixtures/events-v2.json", { cache: "no-store" })).json();
                state.fixtureEvents = ev;
                // remap recorded times so the newest event just happened
                const events = ev.events ?? [];
                const lastT = events.length > 0 ? events[events.length - 1].t_ms : 0;
                const shift = Date.now() - lastT;
                for (const evt of events) {
                    if (typeof evt.t_ms === "number") evt.t_ms = evt.t_ms + shift;
                    state.events.feed(evt);
                }
            } catch { /* fixture without events: timeline stays empty */ }
            try {
                state.fixtureContent = await (await fetch("fixtures/content-v3.json", { cache: "no-store" })).json();
                state.contentEnabled = true;
            } catch { state.contentEnabled = false; }
        }
    } else {
        state.transport = liveTransport(currentApiKey());
        startStream();
    }

    $("key-form").addEventListener("submit", (ev) => {
        ev.preventDefault();
        state.enteredKey = $("key-input").value;
        $("key-input").value = "";
        $("key-form").hidden = true;
        state.transport = liveTransport(currentApiKey());
        state.failures = 0;
        restartStream();
        void poll();
    });

    $("tabs").addEventListener("click", (ev) => {
        const btn = ev.target.closest("button");
        if (btn) setView(btn.dataset.view);
    });

    $("tl-windows").addEventListener("click", (ev) => {
        const btn = ev.target.closest("button");
        if (!btn) return;
        state.tlWindow = btn.dataset.win;
        for (const b of $("tl-windows").querySelectorAll("button")) {
            b.classList.toggle("active", b === btn);
        }
        renderTimeline({ force: true });
    });

    $("tl-scales").addEventListener("click", (ev) => {
        const btn = ev.target.closest("button");
        if (!btn) return;
        state.tlScale = btn.dataset.scale;
        for (const b of $("tl-scales").querySelectorAll("button")) {
            b.classList.toggle("active", b === btn);
        }
        renderTimeline({ force: true });
    });

    document.addEventListener("visibilitychange", () => {
        if (!document.hidden) void poll(); // immediate refresh + faster cadence
        else schedule();
    });

    const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
    const initialView = hash.get("view");
    if (initialView === "timeline" || initialView === "cache") setView(initialView);

    timelineTick();
    void poll();
}

boot().catch((error) => {
    $("status").className = "status err";
    $("status").textContent = "failed to start: " + (error.message ?? error);
});
