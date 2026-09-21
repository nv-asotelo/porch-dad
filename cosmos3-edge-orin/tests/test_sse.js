"use strict";
// Protocol and browser capture tests. These do not validate model inference.
const assert = require("node:assert/strict");
const {SSEParser, readCompletionEvent} = require("../web/app.js");
const source = ': keepalive\r\nevent: token\r\ndata: {"text":"café 🌍"}\r\n\r\ndata: line 1\ndata: line 2\n\ndata: [DONE]\r\r';
const bytes = new TextEncoder().encode(source);
const expected = [{type: "token", data: '{"text":"café 🌍"}'},
  {type: "message", data: "line 1\nline 2"}, {type: "message", data: "[DONE]"}];
for (let cut = 0; cut <= bytes.length; cut++) {
  const events = [], parser = new SSEParser(event => events.push(event)), decoder = new TextDecoder();
  parser.feed(decoder.decode(bytes.slice(0, cut), {stream: true}));
  parser.feed(decoder.decode(bytes.slice(cut), {stream: true}));
  parser.feed(decoder.decode(), true);
  assert.deepEqual(events, expected, `network boundary ${cut}`);
}
{
  const events = [], parser = new SSEParser(event => events.push(event)), decoder = new TextDecoder();
  for (const byte of bytes) parser.feed(decoder.decode(Uint8Array.of(byte), {stream: true}));
  parser.feed(decoder.decode(), true);
  assert.deepEqual(events, expected, "byte-at-a-time UTF-8");
}
for (const truncated of ["data: [DONE]", "data: [DONE]\n", "data: [DONE]\r", "data: [DONE]\r\n"]) {
  const events = [], parser = new SSEParser(event => events.push(event));
  parser.feed(truncated); parser.feed("", true);
  assert.deepEqual(events, [], "incomplete final event must be discarded");
}
for (const ending of ["\n\n", "\r\n\r\n", "\r\r"]) {
  const events = [], parser = new SSEParser(event => events.push(event));
  parser.feed("data: [DONE]" + ending); parser.feed("", true);
  assert.deepEqual(events, [{type: "message", data: "[DONE]"}]);
}
assert.throws(() => new SSEParser(() => {}).feed("x".repeat(1048577)), /size limit/);
for (const reason of ["error", "cancelled", "canceled"]) {
  assert.throws(() => readCompletionEvent({data: JSON.stringify({choices: [{finish_reason: reason}]})}), /Backend ended/);
}
for (const reason of ["stop", "length"]) {
  assert.equal(readCompletionEvent({data: JSON.stringify({choices: [{finish_reason: reason}]})}).finishReason, reason);
}
assert.equal(readCompletionEvent({data: "[DONE]"}).done, true);
assert.throws(() => readCompletionEvent({data: JSON.stringify({error: {message: "transport fixture failure"}})}), /fixture failure/);
console.log("PASS: SSE boundaries, UTF-8, CRLF/CR/LF, multiline data, comments, event limit, truncated EOF and failed completion status.");

// Exercise the actual DOM request lifecycle with cancellation promises that
// never settle. This reproduces the real browser's Stop/restart failure without
// making a model request or depending on a particular browser transport.
(async () => {
  const fs = require("node:fs");
  const vm = require("node:vm");
  const elements = new Map();
  function detach(node) {
    if (node.parentNode) {
      const siblings = node.parentNode.children;
      siblings.splice(siblings.indexOf(node), 1);
      node.parentNode = null;
    }
  }
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      value: "", textContent: "", disabled: false, hidden: false, handlers: {}, style: {}, dataset: {},
      children: [], parentNode: null, attributes: {},
      naturalWidth: 512, naturalHeight: 512,
      classList: {add() {}, remove() {}},
      addEventListener(type, callback) { this.handlers[type] = callback; },
      setAttribute(name, value) { this.attributes[name] = String(value); },
      getAttribute(name) { return this.attributes[name] ?? null; },
      append(...nodes) {
        for (const node of nodes) { detach(node); this.children.push(node); node.parentNode = this; }
      },
      before(node) {
        detach(node);
        this.parentNode.children.splice(this.parentNode.children.indexOf(this), 0, node);
        node.parentNode = this.parentNode;
      },
      after(node) {
        detach(node);
        this.parentNode.children.splice(this.parentNode.children.indexOf(this) + 1, 0, node);
        node.parentNode = this.parentNode;
      },
      async decode() {}
    });
    return elements.get(id);
  }
  element("workspace").append(element("cameraPanel"), element("captionPanel"));
  const captionRadios = ["side", "above", "below"].map(value => {
    const radio = element(`caption-${value}`); radio.value = value; return radio;
  });
  const storedPreferences = new Map();
  const frameCallbacks = new Map();
  let frameCallbackId = 0, tracksStopped = 0;
  const track = {stop() { tracksStopped += 1; }, getSettings: () => ({frameRate: 30})};
  const media = {getTracks: () => [track], getVideoTracks: () => [track]};
  const video = element("video");
  Object.assign(video, {readyState: 0, videoWidth: 1280, videoHeight: 720,
    async play() { this.readyState = 2; },
    requestVideoFrameCallback(callback) { const id = ++frameCallbackId; frameCallbacks.set(id, callback); return id; },
    cancelVideoFrameCallback(id) { frameCallbacks.delete(id); }
  });
  function emitFrame(presentedFrames) {
    assert.equal(frameCallbacks.size, 1, "A single camera callback owns the frame cadence");
    const [id, callback] = frameCallbacks.entries().next().value;
    frameCallbacks.delete(id);
    callback(performance.now(), {presentedFrames});
  }
  element("prompt").value = "Describe the uploaded image.";
  element("maxTokens").value = "64";
  element("resolution").value = "512";
  element("requestCount").textContent = "0 completed";
  function deferred() {
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return {promise, resolve, reject};
  }
  function stream(text, complete = false, throwOnCancel = false) {
    const cancellation = deferred();
    const pendingRead = deferred();
    const delta = `data: ${JSON.stringify({choices: [{delta: {content: text}}]})}\n\n`;
    const ending = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';
    return {cancellation, signal: null, cancelCalls: 0,
      finish() { pendingRead.resolve({value: new TextEncoder().encode(ending), done: false}); },
      response(signal) {
        this.signal = signal;
        let first = true;
        const owner = this;
        // Register rejection only when a read is actually pending.
        const reader = {
          read() {
            if (first) {
              first = false;
              return Promise.resolve({value: new TextEncoder().encode(delta + (complete ? ending : "")), done: false});
            }
            signal.addEventListener("abort", () => pendingRead.reject(signal.reason), {once: true});
            return pendingRead.promise;
          },
          cancel() {
            owner.cancelCalls += 1;
            if (throwOnCancel) throw new Error("fixture synchronous cancellation failure");
            return cancellation.promise;
          }
        };
        return {ok: true, headers: {get: () => "text/event-stream"}, body: {getReader: () => reader}};
      }
    };
  }
  const completed = stream("Completed fixture", true);
  const stopped = stream("Stopped fixture");
  const restarted = stream("Current request");
  const final = stream("Restart completed", true, true);
  const manualCamera = stream("Manual camera request");
  const automaticCamera = stream("Automatic camera request");
  const pausedCamera = stream("Manual paused camera", true);
  const responses = [completed, stopped, restarted, final, manualCamera, automaticCamera, pausedCamera];
  let posts = 0;
  const captures = [];
  const canvasContext = {
    fillRect(...bounds) { this.fill = {color: this.fillStyle, bounds}; },
    drawImage(source, ...bounds) { this.image = {source, bounds}; }
  };
  const canvas = {
    getContext: () => canvasContext,
    toDataURL() {
      captures.push({width: this.width, height: this.height,
        fill: canvasContext.fill, image: canvasContext.image});
      return "data:image/jpeg;base64,AA==";
    }
  };
  function checkCapture(index, width, height, bounds) {
    const capture = captures[index];
    assert.deepEqual([capture.width, capture.height], [width, height]);
    assert.deepEqual(capture.fill, {color: "#000", bounds: [0, 0, width, height]});
    assert.equal(capture.image.source, element("uploadedImage"));
    assert.deepEqual(capture.image.bounds, bounds, "Full source image is centered without crop or upscaling");
  }
  let createdElements = 0;
  const context = {
    document: {getElementById: element,
      createElement: tag => tag === "canvas" ? canvas : element(`created-${tag}-${++createdElements}`),
      querySelectorAll: selector => {
        assert.equal(selector, 'input[name="captionPosition"]'); return captionRadios;
      },
      visibilityState: "visible", addEventListener() {}},
    window: {isSecureContext: true, addEventListener() {}},
    navigator: {mediaDevices: {getUserMedia: async () => media}},
    localStorage: {getItem: key => storedPreferences.get(key) ?? null,
      setItem: (key, value) => storedPreferences.set(key, value)},
    URL: {createObjectURL: () => "blob:fixture", revokeObjectURL() {}},
    AbortController, AbortSignal, TextDecoder, performance, setTimeout, clearTimeout, clearInterval, setInterval() {},
    async fetch(url, options) {
      if (url === "/health/ready") return {ok: true, json: async () => ({status: "ready"})};
      if (url === "/v1/models") return {ok: true, json: async () => ({data: [{id: "fixture-only"}]})};
      if (url === "/api/metrics") return {ok: true, json: async () => ({
        sampled_at: new Date().toISOString(), age_ms: 0, status: "ok",
        cpu: {utilization_percent: 25}, gpu: {utilization_percent: 75},
        memory: {used_bytes: 6 * 2**30, total_bytes: 8 * 2**30, utilization_percent: 75, shared: true, measurement: "MemTotal - MemAvailable"}
      })};
      assert.equal(url, "/v1/chat/completions");
      return responses[posts++].response(options.signal);
    }
  };
  vm.runInNewContext(fs.readFileSync(require.resolve("../web/app.js"), "utf8"), context);
  const turn = () => new Promise(resolve => setImmediate(resolve));
  async function settles(promise, message) {
    let timer;
    try {
      await Promise.race([promise, new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), 1000);
      })]);
    } finally { clearTimeout(timer); }
  }
  await turn();
  // Legacy capture/cancellation behavior remains available in Lightweight.
  element("lightweightPreset").checked = true;
  element("lightweightPreset").handlers.change();
  await element("imageInput").handlers.change({target: {files: [{type: "image/jpeg", size: 1}]}});
  element("uploadedImage").naturalHeight = 384;
  assert.equal(element("analyzeButton").disabled, false);
  await settles(element("analyzeButton").handlers.click(), "Completion cleanup waited for an unresolved cancellation");
  checkCapture(0, 512, 384, [0, 0, 512, 384]); // Common 512px 4:3 stays unchanged.
  assert.equal(element("requestCount").textContent, "1 completed");
  assert.equal(element("analyzeButton").disabled, false);
  assert.equal(completed.cancelCalls, 1);

  element("uploadedImage").naturalWidth = 1920;
  element("uploadedImage").naturalHeight = 1080;
  element("resolution").value = "384";
  const stopRequest = element("analyzeButton").handlers.click();
  await turn();
  checkCapture(1, 384, 256, [0, 20, 384, 216]); // 16:9 at 384px receives top/bottom bars.
  assert.equal(element("answer").textContent, "Stopped fixture");
  element("stopButton").handlers.click();
  await settles(stopRequest, "Stop cleanup waited for an unresolved cancellation");
  assert.equal(element("runStatus").textContent, "Stopped · partial answer");
  assert.equal(element("requestCount").textContent, "1 completed");
  assert.equal(element("analyzeButton").disabled, false);
  assert.equal(stopped.signal.aborted, true);

  element("uploadedImage").naturalWidth = 128;
  element("uploadedImage").naturalHeight = 96;
  element("resolution").value = "512";
  const newRequest = element("analyzeButton").handlers.click();
  await turn();
  checkCapture(2, 256, 256, [64, 80, 128, 96]); // Small image keeps original content pixels.
  stopped.cancellation.resolve(); // An old transport finally reports cleanup.
  completed.cancellation.resolve();
  await turn();
  assert.equal(element("answer").textContent, "Current request");
  assert.equal(element("runStatus").textContent, "Writing answer…");
  assert.equal(element("analyzeButton").disabled, true);
  assert.equal(restarted.signal.aborted, false);
  element("stopButton").handlers.click();
  await settles(newRequest, "Restart lost ownership of its abort controller");
  assert.equal(restarted.signal.aborted, true);
  assert.equal(element("requestCount").textContent, "1 completed");
  restarted.cancellation.reject(new Error("late fixture cancellation rejection"));
  element("uploadedImage").naturalWidth = 256;
  element("uploadedImage").naturalHeight = 2048;
  element("resolution").value = "768";
  await settles(element("analyzeButton").handlers.click(), "Synchronous cancellation failure blocked controls");
  checkCapture(3, 256, 768, [80, 0, 96, 768]); // Tall image receives left/right bars.
  assert.equal(posts, 4);
  assert.equal(element("requestCount").textContent, "2 completed");
  assert.equal(element("analyzeButton").disabled, false);
  await turn();
  console.log("PASS: 512px 4:3 capture unchanged; 384px widescreen, small and 768px portrait captures preserve the whole image inside centered black padding.");
  console.log("PASS: pending/rejected/synchronous stream cancellation cannot block completion or Stop/restart, and late cleanup preserves the current request.");

  element("liveVlmPreset").checked = true;
  element("liveVlmPreset").handlers.change();
  element("liveToggleButton").handlers.click(); // Live off before opening the preview.
  assert.equal(element("liveToggleButton").getAttribute("aria-pressed"), "false");
  await element("startButton").handlers.click();
  assert.equal(video.srcObject, media);
  assert.equal(tracksStopped, 0);
  assert.equal(element("analyzeButton").disabled, false);
  emitFrame(1); emitFrame(30);
  await turn();
  assert.equal(posts, 4, "Live off must not submit automatic frames");

  const manualRequest = element("analyzeButton").handlers.click();
  await turn();
  assert.equal(posts, 5);
  assert.equal(captures[4].image.source, video, "Manual inference captures the running camera");
  assert.deepEqual([captures[4].width, captures[4].height], [1280, 720]);
  element("liveToggleButton").handlers.click(); // Live on during manual streaming.
  emitFrame(60);
  await element("analyzeButton").handlers.click(); // Exercise the busy guard independently of button disabling.
  assert.equal(posts, 5, "Automatic cadence and manual clicks cannot steal an active request");
  const above = captionRadios.find(radio => radio.value === "above");
  above.checked = true; above.handlers.change();
  assert.equal(element("workspace").dataset.captionPosition, "above");
  assert.deepEqual(element("workspace").children, [element("captionPanel"), element("cameraPanel")]);
  assert.equal(video.srcObject, media);
  assert.equal(element("answer").textContent, "Manual camera request");
  assert.equal(manualCamera.signal.aborted, false, "Caption movement preserves the manual stream");
  element("liveToggleButton").handlers.click(); // Live off must not cancel a manual owner.
  assert.equal(manualCamera.signal.aborted, false);
  manualCamera.finish();
  await settles(manualRequest, "Manual camera completion failed after toggling Live off");
  emitFrame(90); emitFrame(120);
  await turn();
  assert.equal(posts, 5);
  assert.equal(video.srcObject, media);
  assert.equal(tracksStopped, 0);

  element("liveToggleButton").handlers.click();
  emitFrame(150);
  await turn();
  assert.equal(posts, 6);
  assert.equal(element("answer").textContent, "Automatic camera request");
  const side = captionRadios.find(radio => radio.value === "side");
  side.checked = true; side.handlers.change();
  assert.deepEqual(element("workspace").children, [element("cameraPanel"), element("captionPanel")]);
  assert.equal(storedPreferences.get("cosmos3-edge:caption-position"), "side");
  assert.equal(automaticCamera.signal.aborted, false, "Caption movement preserves the automatic stream");
  element("liveToggleButton").handlers.click();
  await turn();
  assert.equal(automaticCamera.signal.aborted, true, "Live off cancels the automatic owner");
  assert.equal(video.srcObject, media, "Live off keeps the same preview stream");
  assert.equal(tracksStopped, 0);
  assert.equal(element("analyzeButton").disabled, false);
  emitFrame(180);
  await turn();
  assert.equal(posts, 6);
  await settles(element("analyzeButton").handlers.click(), "Manual inference after pausing Live did not complete");
  assert.equal(posts, 7);
  assert.equal(captures[6].image.source, video);
  assert.equal(element("answer").textContent, "Manual paused camera");
  assert.equal(video.srcObject, media);
  assert.equal(tracksStopped, 0);
  element("stopButton").handlers.click();
  assert.equal(tracksStopped, 1, "Stop, unlike Live off, releases the camera");
  assert.equal(frameCallbacks.size, 0);
  assert.equal(video.srcObject, null);
  console.log("PASS: Live off preserves camera and manual requests, cancels automatic requests, and blocks new automatic captures; manual/cadence requests share one owner.");
  console.log("PASS: Caption reordering and persistence preserve camera and active manual/automatic streams.");
})().catch(error => { console.error(error); process.exitCode = 1; });
