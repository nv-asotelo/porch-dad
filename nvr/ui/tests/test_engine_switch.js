"use strict";
// Synthetic browser-contract tests; no device, backend or model calls.
const test = require("node:test");
const assert = require("node:assert/strict");
const {EngineRequestScope, EnginePolicySettings, enginePolicy, engineChoices,
  renderEngineChoices, runEngineSwitch, SSEParser, readCompletionEvent} = require("../web/app.js");

const policy = {prompt: "Synthetic fixed identification prompt.", max_tokens: 64,
  temperature: 0, image_tokens: 512, stream: false};
const data = () => ({configured: true, switching: false,
  active: {id: "cosmos", name: "Cosmos3-Edge", model_id: "nvidia/Cosmos3-Edge", request_policy: null},
  engines: [{id: "cosmos", name: "Cosmos3-Edge", available: true},
    {id: "brockone", name: "brockone", available: true, request_policy: policy}]});

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

test("fixed policy requires every actual brockone constraint", () => {
  assert.equal(enginePolicy({id: "cosmos", request_policy: null}), null);
  assert.deepEqual(enginePolicy({model_id: "brockone", request_policy: policy}), policy);
  assert.throws(() => enginePolicy({id: "brockone"}));
  for (const [key, value] of Object.entries({prompt: "", max_tokens: 512, temperature: 0.7, image_tokens: 320, stream: true})) {
    assert.throws(() => enginePolicy({id: "brockone", request_policy: {...policy, [key]: value}}));
  }
});

test("Cosmos settings survive repeated policy refreshes and switch back", () => {
  const settings = new EnginePolicySettings();
  const original = {prompt: "My own scene prompt", maxTokens: "123", imageTokenPreset: "custom", customImageTokens: "444", topP: ".87"};
  const fixed = settings.apply(policy, original);
  assert.deepEqual(fixed, {prompt: policy.prompt, maxTokens: "64", imageTokenPreset: "512", customImageTokens: "512", topP: "1"});
  settings.apply(policy, fixed); settings.apply(policy, fixed);
  assert.deepEqual(settings.apply(null, fixed), original);
  assert.equal(settings.apply(null, original), null);
});

test("one configured brockone entry and only absent models get placeholders", () => {
  const choices = engineChoices(data());
  assert.deepEqual(choices.map(x => x.id), ["cosmos", "brockone", "brocktwo"]);
  assert.equal(choices[2].available, false);
  const configured = data(); configured.engines.push({id: "brocktwo", name: "brocktwo", available: false, reason: "Not qualified"});
  assert.equal(engineChoices(configured).length, 3);
  configured.engines.push(configured.engines[0]);
  assert.throws(() => engineChoices(configured), /Invalid/);
});

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
  const [active, unavailable] = container.children.map(c => c.children[0]);
  assert.equal(active.disabled, true); assert.equal(active.attributes["aria-pressed"], "true");
  assert.equal(unavailable.textContent, engines.engines[1].name);
  assert.equal(container.children[1].children[1].textContent, "Target qualification pending");
  unavailable.events.click(); assert.deepEqual(calls, []);
  engines.engines[1].available = true;
  renderEngineChoices(container, engines, false, id => calls.push(id));
  container.children[1].children[0].events.click(); assert.deepEqual(calls, ["brockone"]);
  renderEngineChoices(container, engines, true, id => calls.push(id));
  assert.ok(container.children.every(c => c.children[0].disabled));
});

test("complete-answer SSE is compatible without fabricating token metrics", () => {
  const events = [], parser = new SSEParser(event => events.push(readCompletionEvent(event)));
  parser.feed('data: {"model":"brockone","choices":[{"delta":{"content":"This is Pikachu."},"finish_reason":"stop"}],"live_vision":{"streaming":false,"token_timing_available":false}}\n\ndata: [DONE]\n\n');
  assert.deepEqual(events, [{text: "This is Pikachu.", finishReason: "stop"}, {done: true}]);
  assert.ok(events.every(e => !Object.hasOwn(e, "metrics")));
});
