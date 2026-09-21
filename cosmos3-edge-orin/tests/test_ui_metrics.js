"use strict";
// Deterministic UI accounting tests; no model, device, or browser transport is used.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const {LatencySummary, appendTelemetrySample, telemetrySegments} = require("../web/app.js");

{
  const summary = new LatencySummary();
  assert.equal(summary.count, 0);
  assert.equal(summary.lastMs, null);
  assert.equal(summary.averageMs, null);
  const latencies = [10000.125, ...Array(99).fill(0.125)];
  for (const latency of latencies) assert.equal(summary.add(latency), true);
  assert.equal(summary.count, 100);
  assert.equal(summary.lastMs, 0.125);
  assert.equal(summary.averageMs, 100.125, "Average includes the full session, not a moving window");
  const before = {count: summary.count, lastMs: summary.lastMs, averageMs: summary.averageMs};
  for (const invalid of [-1, NaN, Infinity, -Infinity, null, undefined, "12"])
    assert.equal(summary.add(invalid), false);
  assert.deepEqual({count: summary.count, lastMs: summary.lastMs, averageMs: summary.averageMs}, before);
  summary.reset();
  assert.deepEqual([summary.count, summary.totalMs, summary.lastMs, summary.averageMs], [0, 0, null, null]);
  for (const value of [100.49, 100.49, 100.99]) summary.add(value);
  assert.equal(summary.lastMs, 100.99);
  assert.equal(Math.round(summary.averageMs), 101, "Rounding each observation before averaging would give 100");
  assert.ok(Math.abs(summary.averageMs - (100.49 + 100.49 + 100.99) / 3) < 1e-12);
  summary.reset();
  assert.equal(summary.add(0), true, "A finite measured zero is valid");
  assert.equal(summary.averageMs, 0);
}

{
  const original = [];
  let history = appendTelemetrySample(original, {at: 1000, cpu: 0, gpu: null, memory: 75});
  assert.deepEqual(original, [], "Appending history does not mutate the previous snapshot");
  assert.equal(history[0].cpu, 0, "Real zero usage must not become missing data");
  assert.equal(history[0].gpu, null);
  const duplicate = appendTelemetrySample(history, {at: 1000, cpu: 99, gpu: 99, memory: 99});
  assert.deepEqual(duplicate, history, "A cached timestamp cannot invent another sample or overwrite its values");
  assert.deepEqual(appendTelemetrySample(history, {at: NaN, cpu: 1}), history);
  history = appendTelemetrySample(history, {at: 2000, cpu: 10, gpu: 25, memory: 76});
  history = appendTelemetrySample(history, {at: 3000, cpu: null, gpu: 30, memory: 77});
  history = appendTelemetrySample(history, {at: 4000, cpu: 20, gpu: 35, memory: 78});
  history = appendTelemetrySample(history, {at: 6000, cpu: 30, gpu: 40, memory: 79});
  history = appendTelemetrySample(history, {at: 7000, cpu: 40, gpu: 45, memory: 80, breakBefore: true});
  assert.deepEqual(telemetrySegments(history, "cpu", 7000), [
    [{at: 1000, value: 0}, {at: 2000, value: 10}],
    [{at: 4000, value: 20}], [{at: 6000, value: 30}], [{at: 7000, value: 40}]
  ], "Missing values, timestamp gaps, and explicit outage boundaries must not be connected");
  assert.deepEqual(telemetrySegments(history, "gpu", 3000), [
    [{at: 2000, value: 25}, {at: 3000, value: 30}]
  ], "One metric's missing value does not erase other metrics, and future points are excluded");
  const badValues = appendTelemetrySample([], {at: 8000, cpu: "0", gpu: Infinity, memory: 101});
  assert.deepEqual([badValues[0].cpu, badValues[0].gpu, badValues[0].memory], [null, null, null]);
  assert.deepEqual(telemetrySegments(badValues, "cpu", 8000), []);
  const restarted = appendTelemetrySample(history, {at: 500, cpu: 5, gpu: 6, memory: 7});
  assert.equal(restarted.length, 1, "Device timestamp rollback starts a fresh chart history");
  assert.equal(restarted[0].breakBefore, true);
  assert.deepEqual(telemetrySegments(history, "cpu", 67001), [], "Old values expire from the visible minute");
  let dense = [];
  for (let at = 0; at < 130; at++) dense = appendTelemetrySample(dense, {at, cpu: at % 100});
  assert.equal(dense.length, 120, "History is bounded even with unexpected rapid samples");
  const expired = appendTelemetrySample(history, {at: 67001, cpu: 5, gpu: 6, memory: 7});
  assert.equal(expired.length, 1, "Appending a fresh device sample evicts points older than a minute");
}
console.log("PASS: Cumulative unrounded latency accounting/reset; bounded telemetry history preserves zero, gaps and device timestamps.");

// Run the real application handlers with a controlled clock. Failed/truncated/
// canceled requests must not be added merely because they produced partial text.
(async () => {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      value: "", textContent: "", hidden: false, disabled: false, dataset: {}, style: {},
      handlers: {}, children: [], parentNode: null, naturalWidth: 512, naturalHeight: 512,
      classList: {add() {}, remove() {}},
      addEventListener(type, callback) { this.handlers[type] = callback; },
      setAttribute() {},
      append(node) {
        if (node.parentNode) {
          const children = node.parentNode.children;
          children.splice(children.indexOf(node), 1);
        }
        this.children.push(node); node.parentNode = this;
      },
      async decode() {}
    });
    return elements.get(id);
  }
  const radios = ["above", "below", "side"].map(value => Object.assign(element(`caption-${value}`), {value}));
  element("prompt").value = "Describe this fixture.";
  element("maxTokens").value = "512";
  element("captionBelow").append(element("answer"));
  const fixtures = [];
  let clockMs = 0, posts = 0;
  const canvas = {getContext: () => ({fillRect() {}, drawImage() {}}),
    toDataURL() { clockMs += fixtures[0].encodeMs; return "data:image/jpeg;base64,AA=="; }};
  const token = 'data: {"choices":[{"delta":{"content":"Fixture answer"}}]}\n\n';
  const finish = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';
  function chunk(text) { return {value: new TextEncoder().encode(text), done: false}; }
  let created = 0;
  const context = {
    document: {
      getElementById: id => id === "cpuUsage" ? null : element(id), // Resource polling is tested separately above.
      createElement: tag => tag === "canvas" ? canvas : element(`created-${tag}-${++created}`),
      querySelectorAll: () => radios, addEventListener() {}, visibilityState: "visible"
    },
    window: {isSecureContext: true, addEventListener() {}}, navigator: {},
    localStorage: {getItem: () => null, setItem() {}},
    URL: {createObjectURL: () => "blob:metrics-fixture", revokeObjectURL() {}},
    AbortController, AbortSignal, TextDecoder, setTimeout, clearTimeout, clearInterval, setInterval() {},
    performance: {now: () => clockMs},
    async fetch(url, options) {
      if (url === "/health/ready") return {ok: true, json: async () => ({status: "ready"})};
      if (url === "/v1/models") return {ok: true, json: async () => ({data: [{id: "fixture-only"}]})};
      assert.equal(url, "/v1/chat/completions");
      const fixture = fixtures.shift(); posts += 1;
      assert.ok(fixture, "Every model request must have a test fixture");
      clockMs += fixture.networkMs;
      if (fixture.kind === "http-error") return {ok: false, status: 500, json: async () => ({error: {message: "fixture failure"}})};
      let reads = 0;
      const reader = {
        async read() {
          reads += 1;
          if (reads === 1) { clockMs += fixture.firstMs; return chunk(token); }
          if (fixture.kind === "canceled") return new Promise((_, reject) => {
            if (options.signal.aborted) reject(options.signal.reason);
            else options.signal.addEventListener("abort", () => reject(options.signal.reason), {once: true});
          });
          clockMs += fixture.tailMs;
          if (fixture.kind === "truncated") return {done: true};
          if (fixture.kind === "stream-error") return chunk('data: {"error":{"message":"fixture stream failed"}}\n\n');
          return chunk(finish);
        },
        cancel: async () => {}
      };
      return {ok: true, headers: {get: () => "text/event-stream"}, body: {getReader: () => reader}};
    }
  };
  vm.runInNewContext(fs.readFileSync(require.resolve("../web/app.js"), "utf8"), context);
  const turn = () => new Promise(resolve => setImmediate(resolve));
  await turn();
  await element("imageInput").handlers.change({target: {files: [{type: "image/jpeg", size: 1}]}});
  function display() {
    return [element("latencyValue").textContent, element("avgLatencyValue").textContent, element("countValue").textContent];
  }
  async function run(fixture) {
    fixtures.push(fixture);
    await element("analyzeButton").handlers.click();
  }
  await run({kind: "complete", encodeMs: 13.49, networkMs: 100, firstMs: 100, tailMs: 147});
  assert.deepEqual(display(), ["360", "360", "1"]);
  assert.equal(element("totalTime").textContent, "347 ms", "Existing request timing excludes JPEG encoding");
  assert.equal(element("ttft").textContent, "200 ms", "Latency summary is not the first-token duration");
  const completedDisplay = display();
  for (const kind of ["http-error", "stream-error", "truncated"]) {
    await run({kind, encodeMs: 5, networkMs: 100, firstMs: 100, tailMs: 100});
    assert.deepEqual(display(), completedDisplay, `${kind} must leave the last successful latency and average intact`);
    assert.equal(element("runStatus").textContent, "Request failed");
  }
  fixtures.push({kind: "canceled", encodeMs: 50, networkMs: 100, firstMs: 100});
  const pending = element("analyzeButton").handlers.click();
  await turn();
  assert.deepEqual(display(), completedDisplay, "Pending partial text does not count as completed inference");
  clockMs += 5000;
  element("stopButton").handlers.click();
  await pending;
  assert.deepEqual(display(), completedDisplay, "Cancellation and Stop do not add or reset observations");
  element("liveToggleButton").handlers.click();
  assert.deepEqual(display(), completedDisplay, "Live pause does not reset the session average");
  await run({kind: "complete", encodeMs: 5.49, networkMs: 200, firstMs: 50, tailMs: 105});
  await run({kind: "complete", encodeMs: 10.99, networkMs: 200, firstMs: 50, tailMs: 100});
  assert.deepEqual(display(), ["361", "361", "3"], "Display rounds the cumulative raw mean, not individual samples");
  assert.equal(posts, 7);
  element("lightweightPreset").checked = true;
  element("lightweightPreset").handlers.change();
  assert.deepEqual(display(), ["—", "—", "0"], "Capture preset reset starts a fresh latency session");
  await run({kind: "complete", encodeMs: 5, networkMs: 25, firstMs: 25, tailMs: 25});
  assert.deepEqual(display(), ["80", "80", "1"], "The new session does not retain pre-reset measurements");
  console.log("PASS: Actual UI completion accounting includes JPEG capture, excludes failed/truncated/canceled requests, preserves results on Stop/pause, and resets on capture preset changes.");
})().catch(error => { console.error(error); process.exitCode = 1; });
