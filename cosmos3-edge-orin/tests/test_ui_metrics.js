"use strict";
// Deterministic UI accounting tests; no model, device, or browser transport is used.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const {LatencySummary, appendTelemetrySample, telemetrySegments, ADVANCED_DEFAULTS,
  validateAdvancedSettings, readServerInferenceMs, readServerFirstTextMs} = require("../web/app.js");

const metrics = (milliseconds = 100.49, firstTextMs = 30.49) => ({request_id: "fixture-request", native_inference_ms: milliseconds,
  server_elapsed_ms: milliseconds + 55, completion_tokens: 12,
  server_first_text_ms: firstTextMs, first_text_timing_boundary: "native_start_to_server_text",
  timing_boundary: "native_inference", timing_source: "server_monotonic"});
{
  assert.deepEqual(ADVANCED_DEFAULTS, {imageTokens: 512, imageTokenLimit: 512, topP: 1});
  for (const budget of [4, 320, 512]) assert.deepEqual(validateAdvancedSettings(budget, 0.95), {imageTokens: budget, topP: 0.95});
  for (const budget of [3, 513, 320.5, NaN, Infinity, "320", null])
    assert.throws(() => validateAdvancedSettings(budget, 0.95), /image token budget/);
  assert.throws(() => validateAdvancedSettings(320, 0.95, 256), /4 to 256/, "Loaded capacity bounds the custom field");
  for (const topP of [0, -1, 1.01, NaN, Infinity, "0.95", null])
    assert.throws(() => validateAdvancedSettings(320, topP), /top-p/);
  for (const topP of [0.0001, 0.95, 1]) assert.equal(validateAdvancedSettings(320, topP).topP, topP);
  assert.equal(readServerInferenceMs(metrics(0)), 0);
  assert.equal(readServerInferenceMs(metrics(100.49)), 100.49);
  for (const invalid of [null, undefined, {}, {server_elapsed_ms: 20},
    {...metrics(), timing_boundary: "request"}, {...metrics(), timing_source: "browser"},
    {...metrics(), request_id: ""}, {...metrics(), completion_tokens: -1},
    {...metrics(), completion_tokens: "12"}, {...metrics(), native_inference_ms: "123"},
    {...metrics(), native_inference_ms: NaN}, {...metrics(), native_inference_ms: -1},
    {...metrics(), native_inference_ms: Infinity}]) assert.equal(readServerInferenceMs(invalid), null);
  assert.equal(readServerFirstTextMs(metrics(0, 0)), 0);
  assert.equal(readServerFirstTextMs(metrics(100, 250.49)), 250.49,
    "Server consumer scheduling can put first-text observation after native completion; do not clamp it");
  for (const invalid of [null, undefined, {}, {server_elapsed_ms: 20},
    {...metrics(), first_text_timing_boundary: undefined}, {...metrics(), first_text_timing_boundary: "browser_request_to_text"},
    {...metrics(), timing_boundary: "request"}, {...metrics(), timing_source: "browser"},
    {...metrics(), request_id: ""}, {...metrics(), completion_tokens: -1},
    {...metrics(), completion_tokens: "12"}, {...metrics(), server_first_text_ms: "123"},
    {...metrics(), server_first_text_ms: null}, {...metrics(), server_first_text_ms: undefined},
    {...metrics(), server_first_text_ms: true}, {...metrics(), server_first_text_ms: NaN},
    {...metrics(), server_first_text_ms: -1}, {...metrics(), server_first_text_ms: Infinity}]) {
    assert.equal(readServerFirstTextMs(invalid), null, "Only the explicit valid server first-text field is usable");
  }
}

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
  const html = fs.readFileSync(require.resolve("../web/index.html"), "utf8");
  element("prompt").value = html.match(/<textarea id="prompt"[^>]*>([^<]*)<\/textarea>/)[1];
  element("maxTokens").value = html.match(/id="maxTokens"[^>]*value="(\d+)"/)[1];
  element("captionBelow").append(element("answer"));
  const fixtures = [];
  let clockMs = 0, posts = 0;
  const requestBodies = [], pollers = new Map();
  let runtime = {engine_id: "fixture-engine-1", max_image_tokens_per_image: 512,
    max_image_tokens_per_image_limit: 512, top_p: 1, static_clocks: false, encoder_cache_bytes: 0};
  const canvas = {getContext: () => ({fillRect() {}, drawImage() {}}),
    toDataURL() { clockMs += fixtures[0].encodeMs; return "data:image/jpeg;base64,AA=="; }};
  const token = 'data: {"choices":[{"delta":{"content":"Fixture answer"}}]}\n\n';
  function finish(fixture) {
    const timedEvent = fixture.metrics === undefined ? "" : `data: ${JSON.stringify({choices: [], cosmos_metrics: fixture.metrics})}\n\n`;
    return 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n' + timedEvent + 'data: [DONE]\n\n';
  }
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
    AbortController, AbortSignal, TextDecoder, setTimeout, clearTimeout, clearInterval,
    setInterval(callback, delay) { pollers.set(delay, callback); },
    performance: {now: () => clockMs},
    async fetch(url, options) {
      if (url === "/health/ready") return {ok: true, json: async () => ({status: "ready"})};
      if (url === "/v1/models") return {ok: true, json: async () => ({data: [{id: "fixture-only"}]})};
      if (url === "/api/runtime") return {ok: true, json: async () => runtime};
      assert.equal(url, "/v1/chat/completions");
      const fixture = fixtures.shift(); posts += 1;
      requestBodies.push(JSON.parse(options.body));
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
          if (fixture.gate) await fixture.gate;
          clockMs += fixture.tailMs;
          if (fixture.kind === "truncated") return {done: true};
          if (fixture.kind === "stream-error") return chunk('data: {"error":{"message":"fixture stream failed"}}\n\n');
          return chunk(finish(fixture));
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
  assert.equal(element("imageTokenPreset").value, "512");
  assert.equal(element("topP").value, "1");
  assert.equal(element("staticClocksValue").textContent, "Disabled (0)");
  assert.equal(element("encoderCacheValue").textContent, "0 MiB");
  assert.equal(element("lightweightPreset").checked, true);
  assert.equal(element("liveVlmPreset").checked, false);
  assert.equal(element("interval").value, "1000");
  assert.equal(element("resolution").value, "512");
  assert.equal(element("interval").disabled, false);
  assert.equal(element("resolution").disabled, false);
  assert.equal(element("serverTtftValue").textContent, "—");
  await run({kind: "complete", encodeMs: 13.49, networkMs: 100, firstMs: 100, tailMs: 147, metrics: metrics(100.49)});
  assert.deepEqual(display(), ["100", "100", "1"]);
  assert.equal(element("serverTtftValue").textContent, "30");
  assert.equal(requestBodies[0].max_image_tokens_per_image, 512);
  assert.equal(requestBodies[0].max_tokens, 64, "Input image budget never replaces the output cap");
  assert.equal(requestBodies[0].top_p, 1);
  assert.equal(requestBodies[0].temperature, 0, "Initial Lightweight requests use greedy decoding");
  assert.equal(requestBodies[0].messages[0].content[0].text,
    "Describe the visible scene in one concise sentence. Focus on objects and actions.");
  assert.equal(element("totalTime").textContent, "347 ms", "Existing request timing excludes JPEG encoding");
  assert.equal(element("ttft").textContent, "200 ms", "Latency summary is not the first-token duration");
  const completedDisplay = display();
  for (const kind of ["http-error", "stream-error", "truncated"]) {
    await run({kind, encodeMs: 5, networkMs: 100, firstMs: 100, tailMs: 100});
    assert.deepEqual(display(), completedDisplay, `${kind} must leave the last successful latency and average intact`);
    assert.equal(element("serverTtftValue").textContent, "30", "Failed requests keep the last successful TTFT");
    assert.equal(element("runStatus").textContent, "Request failed");
  }
  fixtures.push({kind: "canceled", encodeMs: 50, networkMs: 100, firstMs: 100});
  const pending = element("analyzeButton").handlers.click();
  await turn();
  assert.deepEqual(display(), completedDisplay, "Pending partial text does not count as completed inference");
  assert.equal(element("serverTtftValue").textContent, "30", "Browser first-token arrivals cannot overwrite server TTFT");
  clockMs += 5000;
  element("stopButton").handlers.click();
  await pending;
  assert.deepEqual(display(), completedDisplay, "Cancellation and Stop do not add or reset observations");
  assert.equal(element("serverTtftValue").textContent, "30");
  element("liveToggleButton").handlers.click();
  assert.deepEqual(display(), completedDisplay, "Live pause does not reset the session average");
  await run({kind: "complete", encodeMs: 8000, networkMs: 9000, firstMs: 50, tailMs: 105, metrics: metrics(100.49)});
  assert.deepEqual(display(), ["100", "100", "2"], "JPEG encoding and network delays cannot change a server measurement");
  assert.equal(element("serverTtftValue").textContent, "30", "JPEG encoding and network delays cannot enter server TTFT");
  await run({kind: "complete", encodeMs: 10.99, networkMs: 200, firstMs: 50, tailMs: 100, metrics: metrics(100.99)});
  assert.deepEqual(display(), ["101", "101", "3"], "Display rounds the cumulative raw mean, not individual samples");
  assert.equal(posts, 7);
  for (const missing of [undefined, null, {...metrics(), timing_boundary: "request"}, {...metrics(), native_inference_ms: "123"}]) {
    await run({kind: "complete", encodeMs: 5, networkMs: 25, firstMs: 25, tailMs: 25, metrics: missing});
    assert.deepEqual(display(), ["—", "101", "3"], "Missing or invalid server timing clears the latest value and never falls back to browser time");
    assert.match(element("timingStatus").textContent, /no valid server inference timing/);
  }
  element("prompt").value = "Keep this edited prompt.";
  element("maxTokens").value = "73";
  element("liveVlmPreset").checked = true;
  element("liveVlmPreset").handlers.change();
  assert.equal(element("prompt").value, "Keep this edited prompt.");
  assert.equal(element("maxTokens").value, "73", "Capture preset toggles preserve edited output cap");
  assert.deepEqual(display(), ["—", "—", "0"], "Capture preset reset starts a fresh latency session");
  assert.equal(element("serverTtftValue").textContent, "—", "Capture changes reset server TTFT");
  const good = {kind: "complete", encodeMs: 5, networkMs: 25, firstMs: 25, tailMs: 25, metrics: metrics(42, 12.49)};
  await run(good);
  assert.deepEqual(display(), ["42", "42", "1"], "The new session does not retain pre-reset measurements");
  assert.equal(element("serverTtftValue").textContent, "12");
  assert.equal(requestBodies.at(-1).temperature, 0.7);
  assert.equal(requestBodies.at(-1).top_p, 1, "Capture preset preserves explicit advanced sampling settings");
  element("lightweightPreset").checked = true; element("lightweightPreset").handlers.change();
  assert.equal(element("prompt").value, "Keep this edited prompt.");
  assert.equal(element("maxTokens").value, "73");
  await run(good);
  assert.equal(requestBodies.at(-1).temperature, 0);
  assert.equal(requestBodies.at(-1).top_p, 1, "Greedy Lightweight explicitly sends configured top_p");
  element("imageTokenPreset").value = "320"; element("imageTokenPreset").handlers.change();
  assert.deepEqual(display(), ["—", "—", "0"]);
  assert.equal(element("serverTtftValue").textContent, "—", "Image budget changes reset server TTFT");
  await run(good);
  assert.equal(requestBodies.at(-1).max_image_tokens_per_image, 320);
  assert.equal(requestBodies.at(-1).max_tokens, 73);
  element("imageTokenPreset").value = "custom"; element("imageTokenPreset").handlers.change();
  assert.equal(element("customImageTokensLabel").hidden, false);
  element("customImageTokens").value = "513"; element("customImageTokens").handlers.input();
  assert.equal(element("analyzeButton").disabled, true);
  const beforeInvalid = posts;
  await element("analyzeButton").handlers.click();
  assert.equal(posts, beforeInvalid, "Invalid custom values cannot send requests even via a direct handler call");
  element("customImageTokens").value = "256"; element("customImageTokens").handlers.input();
  assert.equal(element("analyzeButton").disabled, false);
  element("topP").value = "0"; element("topP").handlers.input();
  assert.equal(element("analyzeButton").disabled, true);
  element("topP").value = "0.8"; element("topP").handlers.input();
  await run(good);
  assert.equal(requestBodies.at(-1).max_image_tokens_per_image, 256);
  assert.equal(requestBodies.at(-1).top_p, 0.8);
  await pollers.get(5000)();
  assert.equal(element("imageTokenPreset").value, "custom", "Routine runtime polling retains choices for the same engine");
  assert.equal(element("topP").value, "0.8");
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  fixtures.push({...good, gate});
  const inFlight = element("analyzeButton").handlers.click();
  await turn();
  element("imageTokenPreset").value = "512"; element("imageTokenPreset").handlers.change();
  assert.deepEqual(display(), ["—", "—", "0"]);
  assert.equal(element("serverTtftValue").textContent, "—");
  release(); await inFlight;
  assert.deepEqual(display(), ["—", "—", "0"], "Old in-flight settings cannot contaminate the newly selected timing group");
  assert.equal(element("serverTtftValue").textContent, "—", "Stale in-flight first-text timing cannot overwrite a new group");
  assert.equal(requestBodies.at(-1).max_image_tokens_per_image, 256, "Each request snapshots the settings before capture");
  runtime = {...runtime, engine_id: "fixture-engine-2"};
  await pollers.get(5000)();
  assert.equal(element("serverTtftValue").textContent, "—", "New engines reset server TTFT");
  assert.equal(element("imageTokenPreset").value, "512", "A new engine load restores its first-demo 512-token default");
  assert.equal(element("topP").value, "1");
  assert.equal(element("customImageTokensLabel").hidden, true);
  await run(good);
  assert.deepEqual(display(), ["42", "42", "1"]);
  assert.equal(element("serverTtftValue").textContent, "12");
  for (const badFirstText of [undefined, null, "12", -1, Infinity]) {
    await run({...good, metrics: {...good.metrics, server_first_text_ms: badFirstText}});
    assert.equal(element("serverTtftValue").textContent, "—", "Missing or invalid first-text timing clears TTFT without using browser or total time");
    assert.equal(element("latencyValue").textContent, "42", "Valid total timing remains independently visible");
    assert.equal(element("ttft").textContent, "50 ms", "Browser diagnostic remains separate and available");
  }
  await run({...good, metrics: {...good.metrics, first_text_timing_boundary: "http_to_first_text"}});
  assert.equal(element("serverTtftValue").textContent, "—", "A broader timing boundary never substitutes for server TTFT");
  runtime = {...runtime, engine_id: "smaller-fixture", max_image_tokens_per_image: 256, max_image_tokens_per_image_limit: 256};
  await pollers.get(5000)();
  assert.equal(element("customImageTokens").max, "256");
  assert.equal(element("customImageTokens").value, "256");
  assert.equal(element("imageTokenPreset").value, "custom");
  console.log("PASS: Server-only latency ignores JPEG/network delay, excludes invalid/missing/failed/canceled timing, and resets safely on capture, advanced-settings and engine changes.");
  console.log("PASS: Independent input/output budgets, top-p validation, loaded capacity limits and per-request setting snapshots.");
})().catch(error => { console.error(error); process.exitCode = 1; });
