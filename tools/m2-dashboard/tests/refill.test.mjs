import assert from "node:assert/strict";
import test from "node:test";

import { fastRefillView, renderFastRefill } from "../lib/refill.mjs";
import { clone, loadSnapshot } from "./helpers.mjs";

class FakeElement {
    constructor(tag) {
        this.tag = tag;
        this.className = "";
        this.children = [];
        this.textContent = "";
    }

    append(...children) {
        this.children.push(...children);
    }

    replaceChildren(...children) {
        this.children = children;
    }

    set innerHTML(_value) {
        throw new Error("refill rendering must not parse HTML");
    }
}

const fakeDocument = {
    createElement: (tag) => new FakeElement(tag),
};

function textContent(element) {
    return [element.textContent, ...element.children.flatMap((child) => textContent(child))].join(" ");
}

test("fast-refill view distinguishes disabled, eligible, quota, and exact expiry", async () => {
    const source = (await loadSnapshot()).fast_refill;
    assert.equal(fastRefillView(source).state, "width_full");

    const eligible = clone(source);
    eligible.refill.one_member_eligible_now = true;
    assert.equal(fastRefillView(eligible).state, "eligible");
    assert.equal(fastRefillView(eligible).oneMemberEligibleNow, true);

    const exhausted = clone(source);
    exhausted.refill.fast_members_used = 4;
    exhausted.refill.fast_members_remaining = 0;
    exhausted.refill.window_open = false;
    assert.equal(fastRefillView(exhausted).state, "quota_exhausted");

    const disabled = clone(source);
    disabled.configuration = { enabled: false, max_members: 0, window_ms: 0 };
    disabled.cohort = { active: false, selection_open: false, dominant_lane: null, limit: 0 };
    disabled.refill = {
        fast_members_used: 0,
        fast_members_remaining: 0,
        deadline_at: null,
        remaining_ms: 0,
        deadline_expired: false,
        window_open: false,
        one_member_eligible_now: false,
    };
    assert.equal(fastRefillView(disabled).state, "disabled");

    const expired = fastRefillView(eligible, eligible.refill.remaining_ms);
    assert.equal(expired.state, "expired");
    assert.equal(expired.remainingMs, 0);
    assert.equal(expired.windowOpen, false);
    assert.equal(expired.oneMemberEligibleNow, false);
});

test("fast-refill DOM rendering closes the sampled window at its deadline", async () => {
    const source = clone((await loadSnapshot()).fast_refill);
    source.refill.one_member_eligible_now = true;
    const target = new FakeElement("div");

    const before = renderFastRefill(target, source, 0, fakeDocument);
    assert.equal(before.oneMemberEligibleNow, true);
    assert.match(textContent(target), /eligible at sample/);
    assert.match(textContent(target), /2 used · 2 remaining/);
    assert.match(textContent(target), /Initial selection closed/);

    const expired = renderFastRefill(target, source, source.refill.remaining_ms, fakeDocument);
    const rendered = textContent(target);
    assert.equal(expired.windowOpen, false);
    assert.match(rendered, /Refill state expired/);
    assert.match(rendered, /closed · 0 ms left/);
    assert.match(rendered, /One member not eligible at sample/);
});
