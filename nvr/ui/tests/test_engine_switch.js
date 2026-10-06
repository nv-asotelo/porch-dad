"use strict";
// Synthetic browser-contract tests; no device, backend or model calls.
const test = require("node:test");
const assert = require("node:assert/strict");
const {EngineRequestScope, EnginePolicySettings, enginePolicy, engineChoices, renderEngineChoices, engineGroups,
  tabBar, needsDemoMode, demoWarning, runEngineSwitch, SSEParser, readCompletionEvent} = require("../web/app.js");

const policy = {prompt: "Synthetic fixed identification prompt.", max_tokens: 64,
  temperature: 0, image_tokens: 512, stream: false};
const data = () => ({configured: true, switching: false,
  active: {id: "cosmos", name: "Cosmos3-Edge", model_id: "nvidia/Cosmos3-Edge", request_policy: null},
  engines: [{id: "cosmos", name: "Cosmos3-Edge", available: true},
    {id: "alt", name: "Alt model", available: true, request_policy: policy}]});

test("switch invalidates pending inference and health responses", () => {
  const scope = new EngineRequestScope();
  const inference = scope.begin(), health = scope.begin();
  scope.invalidate();
  assert.equal(scope.current(inference), false); assert.equal(scope.current(health), false);
  assert.equal(inference.controller.signal.aborted, true);
  const next = scope.begin();
  scope.release(inference); // Old cleanup cannot remove the new owner.
  assert.equal(scope.current(next), true); assert.equal(scope.requests.size, 1);
});

for (const failure of [false, true]) {
  test(`switch refreshes model/runtime on ${failure ? "rollback" : "success"}`, async () => {
    const scope = new EngineRequestScope(), old = scope.begin(), order = [];
    const action = runEngineSwitch(scope, {
      before() { assert.equal(scope.current(old), false); order.push("paused"); },
      async request() { order.push("request"); if (failure) throw new Error("Synthetic rollback"); },
      after() { order.push("unblocked"); },
      async refresh() { order.push("fresh health/models/runtime/engines"); },
    });
    if (failure) await assert.rejects(action, /rollback/); else await action;
    assert.deepEqual(order, ["paused", "request", "unblocked", "fresh health/models/runtime/engines"]);
  });
}

test("refresh failure is visible and original switch failure is preserved", async () => {
  for (const failure of [false, true]) {
    await assert.rejects(runEngineSwitch(new EngineRequestScope(), {
      before() {}, after() {}, request: async () => { if (failure) throw Error("original switch failure"); },
      refresh: async () => { throw Error("refresh failed"); },
    }), failure ? /original switch failure/ : /refresh failed/);
  }
});

test("switch retains no implicit readiness while the request is pending", async () => {
  let complete; const pending = new Promise(resolve => { complete = resolve; });
  const calls = [], scope = new EngineRequestScope();
  const action = runEngineSwitch(scope, {before: () => calls.push("paused"), request: () => pending,
    after: () => calls.push("done"), refresh: async () => calls.push("refresh")});
  await Promise.resolve(); assert.deepEqual(calls, ["paused"]);
  complete(); await action; assert.deepEqual(calls, ["paused", "done", "refresh"]);
});

test("a fixed policy, if present, must be internally consistent", () => {
  assert.equal(enginePolicy({id: "cosmos", request_policy: null}), null);
  assert.equal(enginePolicy({id: "cosmos"}), null);
  assert.deepEqual(enginePolicy({id: "alt", request_policy: policy}), policy);
  for (const [key, value] of Object.entries({prompt: "", max_tokens: "64", temperature: "0", image_tokens: "512", stream: true})) {
    assert.throws(() => enginePolicy({id: "alt", request_policy: {...policy, [key]: value}}));
  }
});

test("engine settings survive repeated policy refreshes and switch back", () => {
  const settings = new EnginePolicySettings();
  const original = {prompt: "My own scene prompt", maxTokens: "123", imageTokenPreset: "custom", customImageTokens: "444", topP: ".87"};
  const fixed = settings.apply(policy, original);
  assert.deepEqual(fixed, {prompt: policy.prompt, maxTokens: "64", imageTokenPreset: "512", customImageTokens: "512", topP: "1"});
  settings.apply(policy, fixed); settings.apply(policy, fixed);
  assert.deepEqual(settings.apply(null, fixed), original);
  assert.equal(settings.apply(null, original), null);
});

test("only the configured engines are offered - no placeholders for unconfigured ones", () => {
  const choices = engineChoices(data());
  assert.deepEqual(choices.map(x => x.id), ["cosmos", "alt"]);
  const duplicated = data();
  duplicated.engines.push(duplicated.engines[0]);
  assert.throws(() => engineChoices(duplicated), /Invalid/);
});

// Every model card, across the groups the panel renders.
function cards(container) {
  return container.children.filter(c => c.className === "model-group").flatMap(g => g.children[1].children);
}

function element(tag, document) {
  return {tag, ownerDocument: document, children: [], dataset: {}, attributes: {}, events: {},
    setAttribute(k, v) { this.attributes[k] = v; }, addEventListener(k, v) { this.events[k] = v; },
    append(...children) { this.children.push(...children); }, replaceChildren(...children) { this.children = children; },
    set innerHTML(_) { throw Error("Unsafe innerHTML must never be used"); }};
}

test("buttons use safe text, visible reasons, active state and availability", () => {
  const document = {createElement: tag => element(tag, document)};
  const container = element("div", document), calls = [], engines = data();
  engines.engines[1].name = '<img src=x onerror="bad()">';
  engines.engines[1].available = false; engines.engines[1].reason = "Target qualification pending";
  renderEngineChoices(container, engines, false, id => calls.push(id));
  const [active, unavailable] = cards(container).map(c => c.children[0]);
  assert.equal(active.disabled, true); assert.equal(active.attributes["aria-pressed"], "true");
  assert.equal(unavailable.textContent, engines.engines[1].name);
  assert.equal(cards(container)[1].children[1].textContent, "Target qualification pending");
  unavailable.events.click(); assert.deepEqual(calls, []);
  engines.engines[1].available = true;
  renderEngineChoices(container, engines, false, id => calls.push(id));
  assert.equal(cards(container)[1].children[1].textContent, "");   // no "Available" filler
  cards(container)[1].children[0].events.click(); assert.deepEqual(calls, ["alt"]);
  renderEngineChoices(container, engines, true, id => calls.push(id));
  assert.ok(cards(container).every(c => c.children[0].disabled));
});

test("a model that only fits in demo mode says so, and asks before the NVR stops", () => {
  const document = {createElement: tag => element(tag, document)};
  const container = element("div", document), calls = [], engines = data();
  Object.assign(engines.engines[1], {needs_demo_mode: true, demo_reason: "3.6 GB at first inference"});
  engines.demo = {configured: true, on: false};
  assert.equal(needsDemoMode(engines.engines[1], engines), true);
  assert.equal(needsDemoMode(engines.engines[0], engines), false);
  renderEngineChoices(container, engines, false, id => calls.push(id));
  const [, card] = cards(container);
  assert.equal(card.children[1].textContent, "Needs demo mode");
  assert.match(card.children[0].title, /Needs demo mode: 3\.6 GB/);
  card.children[0].events.click(); assert.deepEqual(calls, ["alt"]);   // the page then asks first
  assert.match(demoWarning(engines.engines[1]), /does not fit beside the NVR \(3\.6 GB at first inference\)/);
  assert.match(demoWarning(engines.engines[1]), /Frigate's recording and detection/);
  engines.demo.on = true;
  assert.equal(needsDemoMode(engines.engines[1], engines), false);
  renderEngineChoices(container, engines, false, () => {});
  assert.equal(cards(container)[1].children[1].textContent, "");
});

test("models are grouped: a tab per kind, the registry's groups inside, prefixes dropped", () => {
  const document = {createElement: tag => element(tag, document)};
  const container = element("div", document), tabs = [];
  const engines = {configured: true, switching: false, active: {id: "v3", name: "Cosmos v3"}, engines: [
    {id: "v3", name: "Cosmos v3", group: "Cosmos3-Edge", available: true},
    {id: "qwen", name: "Qwen3-VL 2B", group: "Other VLMs", available: true},
    {id: "v2", name: "Cosmos v2", group: "Cosmos3-Edge", available: true},
    {id: "pk", name: "Pokémon · Skshmjn ViT-B/16", group: "Pokémon", kind: "classifier", available: true},
    {id: "dog", name: "Dog breeds · Dog-Breed-120", group: "Dog breeds", kind: "classifier", available: true}]};
  assert.deepEqual(engineGroups(engineChoices(engines), "vlm").map(([g, es]) => [g, es.map(e => e.id)]),
    [["Cosmos3-Edge", ["v3", "v2"]], ["Other VLMs", ["qwen"]]]);
  // Opens on the active model's tab; the tab bar counts each kind and marks where the active one is.
  assert.equal(renderEngineChoices(container, engines, false, () => {}, null, key => tabs.push(key)), "vlm");
  const [bar, ...groups] = container.children;
  assert.deepEqual(bar.children.map(t => t.textContent), ["Vision-language models (3) · active", "Classifiers (2)"]);
  assert.deepEqual(groups.map(g => g.children[0].textContent), ["Cosmos3-Edge", "Other VLMs"]);
  bar.children[1].events.click(); assert.deepEqual(tabs, ["classifier"]);   // the view only: no switch
  assert.equal(renderEngineChoices(container, engines, false, () => {}, "classifier"), "classifier");
  assert.deepEqual(cards(container).map(c => c.children[0].textContent), ["Skshmjn ViT-B/16", "Dog-Breed-120"]);
  // One kind only: no tab bar at all.
  engines.engines = engines.engines.filter(e => e.kind !== "classifier");
  renderEngineChoices(container, engines, false, () => {});
  assert.equal(container.children[0].className, "model-group");
});

test("tabs are selectable: a click, the arrow keys, Home and End; the focus follows a key", () => {
  const document = {createElement: tag => element(tag, document)};
  const calls = [];
  const bar = tabBar(document, [["a", "Alpha (3)"], ["b", "Beta (2)"], ["c", "Gamma (1)"]], "a",
    (key, focus) => calls.push([key, focus]), "Sample sets");
  const [a, b, c] = bar.children;
  assert.equal(bar.attributes.role, "tablist"); assert.equal(bar.attributes["aria-label"], "Sample sets");
  assert.deepEqual([a, b, c].map(t => [t.textContent, t.attributes.role, t.attributes["aria-selected"], t.tabIndex]),
    [["Alpha (3)", "tab", "true", 0], ["Beta (2)", "tab", "false", -1], ["Gamma (1)", "tab", "false", -1]]);
  a.events.click(); assert.deepEqual(calls, []);          // already selected
  b.events.click(); assert.deepEqual(calls.pop(), ["b", false]);
  const key = (target, name) => { let stopped = false;
    bar.events.keydown({key: name, target, preventDefault() { stopped = true; }}); return stopped; };
  assert.equal(key(a, "ArrowRight"), true); assert.deepEqual(calls.pop(), ["b", true]);
  key(a, "ArrowLeft"); assert.deepEqual(calls.pop(), ["c", true]);     // wraps around
  key(a, "End"); assert.deepEqual(calls.pop(), ["c", true]);
  assert.equal(key(a, "Enter"), false); assert.deepEqual(calls, []); // not a tab key: left alone
  // While a run is going the bar is shown but cannot change.
  const locked = tabBar(document, [["a", "A"], ["b", "B"]], "a", (k, f) => calls.push([k, f]), "", true);
  assert.equal(locked.children[1].disabled, true);
  locked.events.keydown({key: "ArrowRight", target: locked.children[0], preventDefault() {}});
  assert.deepEqual(calls, []);
});

test("complete-answer SSE is compatible without fabricating token metrics", () => {
  const events = [], parser = new SSEParser(event => events.push(readCompletionEvent(event)));
  parser.feed('data: {"model":"nvidia/Cosmos3-Edge","choices":[{"delta":{"content":"A scene."},"finish_reason":"stop"}],"live_vision":{"streaming":false,"token_timing_available":false}}\n\ndata: [DONE]\n\n');
  assert.deepEqual(events, [{text: "A scene.", finishReason: "stop"}, {done: true}]);
  assert.ok(events.every(e => !Object.hasOwn(e, "metrics")));
});
