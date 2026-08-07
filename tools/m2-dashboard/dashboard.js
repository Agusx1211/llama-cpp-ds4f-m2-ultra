// M2 llama-server operations dashboard — browser entry point.
//
// Polls /slots + the request-registry snapshot, derives live rates in
// lib/core.mjs, and renders a flat, data-dense view. No build step, no
// external dependencies. `?fixture=1` (or `?fixture=replay|demo`) replays
// recorded payloads for development. The API key comes from `#key=...` in
// the URL hash or the inline form and is kept in memory only.

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

const POLL_MS = 2000;
const HIDDEN_POLL_MS = 15000;
const MAX_BACKOFF_MS = 30000;
const SPARK_WINDOW_MS = 4 * 60 * 1000;
const METRICS_EVERY = 5; // poll /metrics every N ticks when available

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
};

function currentApiKey() {
    const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
    return state.enteredKey ?? hash.get("key") ?? "";
}

// ---------------------------------------------------------------------------
// Poll loop
// ---------------------------------------------------------------------------

async function poll() {
    clearTimeout(state.pollTimer);
    const t = state.transport;
    try {
        if (state.props === null) {
            state.props = await t.props();
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
    const base = document.hidden ? HIDDEN_POLL_MS : POLL_MS;
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
// Rendering
// ---------------------------------------------------------------------------

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
}

function renderHeader() {
    const model = state.props ? shortModelName(state.props.model_alias ?? state.props.model_path) : "";
    $("model").textContent = model;
    $("model").title = state.props?.model_path ?? "";
    $("build").textContent = state.props?.build_info ?? "";
    document.title = "M2 llama-server · " + model;
}

function renderStatus() {
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
    status.className = "status ok";
    status.textContent = "live · " + (document.hidden ? HIDDEN_POLL_MS : POLL_MS) / 1000 + "s poll";
}

function render() {
    document.querySelector("main").classList.remove("stale");
    renderStatus();
    renderTiles();
    renderWork();
    renderTail();
    renderLanes();
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

function sparkTile(className, label, rateKey, current) {
    const node = tile(className, label, formatRate(current), " tok/s");
    const geo = sparkPath(state.tracker.history, rateKey, {
        width: 200, height: 34, windowMs: SPARK_WINDOW_MS, now: performance.now(),
    });
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 200 34");
    svg.setAttribute("preserveAspectRatio", "none");
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
    $("foot").textContent = bits.join("  ·  ");
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
    const fixture = params.get("fixture");
    if (fixture !== null && fixture !== "0") {
        const name = fixture === "demo" ? "demo" : "replay";
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
    } else {
        state.transport = liveTransport(currentApiKey());
    }

    $("key-form").addEventListener("submit", (ev) => {
        ev.preventDefault();
        state.enteredKey = $("key-input").value;
        $("key-input").value = "";
        $("key-form").hidden = true;
        state.transport = liveTransport(currentApiKey());
        state.failures = 0;
        void poll();
    });

    document.addEventListener("visibilitychange", () => {
        if (!document.hidden) void poll(); // immediate refresh + faster cadence
        else schedule();
    });

    void poll();
}

boot().catch((error) => {
    $("status").className = "status err";
    $("status").textContent = "failed to start: " + (error.message ?? error);
});
