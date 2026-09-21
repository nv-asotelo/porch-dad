"use strict";

const CAPTURE_PRESETS = {
  "live-vlm": {cameraConstraints: {width: {ideal: 1280}, height: {ideal: 720}},
    temperature: 0.7, jpegQuality: 0.75, maxSide: null},
  lightweight: {cameraConstraints: {width: {ideal: 640}, height: {ideal: 480}, frameRate: {ideal: 15, max: 30}},
    temperature: 0, jpegQuality: 0.8, maxSide: 512}
};

// Count video frames, not wall-clock seconds. Delayed browser callbacks can
// cross several sampling boundaries; only the latest frame is eligible.
class FrameCadence {
  constructor(every = 30) { this.every = every; this.previous = null; this.count = 0; }
  observe(presentedFrames) {
    if (!Number.isInteger(presentedFrames) || presentedFrames < 1) return false;
    if (this.previous === null || presentedFrames < this.previous) {
      this.previous = presentedFrames; this.count = 1; return this.every === 1;
    }
    const before = Math.floor(this.count / this.every);
    this.count += presentedFrames - this.previous;
    this.previous = presentedFrames;
    return Math.floor(this.count / this.every) > before;
  }
}

// Handles arbitrary UTF-8/network chunk boundaries, LF/CRLF/CR, comments and
// multiple data lines. The caller supplies a streaming TextDecoder.
class SSEParser {
  constructor(onEvent) { this.onEvent = onEvent; this.buffer = ""; this.data = []; this.event = "message"; }
  feed(text, final = false) {
    this.buffer += text;
    if (this.buffer.length > 1048576) throw new Error("SSE event exceeded the size limit.");
    while (true) {
      const match = /[\r\n]/.exec(this.buffer);
      if (!match) break;
      const position = match.index;
      if (this.buffer[position] === "\r" && position === this.buffer.length - 1 && !final) break;
      const size = this.buffer[position] === "\r" && this.buffer[position + 1] === "\n" ? 2 : 1;
      const line = this.buffer.slice(0, position);
      this.buffer = this.buffer.slice(position + size);
      this.line(line);
    }
    // EOF does not dispatch an event: a real blank-line boundary is required.
    // In particular, a truncated final [DONE] must not count as completion.
    if (final) { this.buffer = ""; this.data = []; this.event = "message"; }
  }
  line(line) {
    if (!line) {
      if (this.data.length) this.onEvent({type: this.event, data: this.data.join("\n")});
      this.data = []; this.event = "message";
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") this.data.push(value);
    if (field === "event") this.event = value;
  }
}

function readCompletionEvent(event) {
  if (event.data === "[DONE]") return {done: true};
  const data = JSON.parse(event.data);
  if (data.error) throw new Error(data.error.message || "Backend stream failed.");
  const choice = data.choices?.[0];
  const finishReason = choice?.finish_reason;
  if (["error", "cancelled", "canceled"].includes(finishReason)) {
    throw new Error(`Backend ended generation with status: ${finishReason}.`);
  }
  return {text: choice?.delta?.content, finishReason};
}

// Device telemetry is independent of inference and camera request ownership.
// A failed or stale sample clears the values instead of leaving them looking live.
function startDeviceTelemetry() {
  const nodes = Object.fromEntries(["cpuUsage", "gpuUsage", "memoryAmount", "memoryPercent",
    "cpuMeter", "gpuMeter", "memoryMeter", "telemetryStatus"].map(id => [id, document.getElementById(id)]));
  if (Object.values(nodes).some(node => !node)) return;
  let active = null, interval = null, lastSample = null;
  const percent = value => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100 ? value : null;
  const bytes = value => typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
  function status(message, kind) {
    nodes.telemetryStatus.textContent = message;
    nodes.telemetryStatus.className = `telemetry-status ${kind}`;
  }
  function clear(message, kind = "unavailable") {
    lastSample = null;
    nodes.cpuUsage.textContent = "—"; nodes.gpuUsage.textContent = "—";
    nodes.memoryAmount.textContent = "—"; nodes.memoryPercent.textContent = "Unavailable";
    for (const name of ["cpuMeter", "gpuMeter", "memoryMeter"]) nodes[name].style.width = "0%";
    status(message, kind);
  }
  async function poll() {
    if (active || document.visibilityState !== "visible") return;
    const controller = new AbortController();
    active = controller;
    const started = performance.now();
    const timeout = setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch("/api/metrics", {cache: "no-store", signal: controller.signal});
      if (!response.ok) throw new Error("Metrics unavailable");
      const sample = await response.json();
      if (controller.signal.aborted || active !== controller || document.visibilityState !== "visible") return;
      if (!sample || !["ok", "partial"].includes(sample.status) ||
          !Number.isFinite(Date.parse(sample.sampled_at)) ||
          typeof sample.age_ms !== "number" || !Number.isFinite(sample.age_ms) || sample.age_ms < 0) {
        throw new Error("Invalid or unavailable metrics");
      }
      // Server age avoids relying on synchronized browser and device clocks.
      // Including request duration is a conservative bound on transit age.
      const age = sample.age_ms + performance.now() - started;
      if (age > 5000) { clear("Stale · waiting for fresh metrics", "stale"); return; }
      const cpu = percent(sample.cpu?.utilization_percent);
      const gpu = percent(sample.gpu?.utilization_percent);
      const memory = sample.memory;
      const used = bytes(memory?.used_bytes), total = bytes(memory?.total_bytes);
      const validMemory = used !== null && total !== null && total > 0 && used <= total &&
        memory.shared === true && memory.measurement === "MemTotal - MemAvailable";
      const memoryPercent = validMemory ? percent(memory.utilization_percent) : null;
      nodes.cpuUsage.textContent = cpu === null ? "—" : `${cpu.toFixed(1)}%`;
      nodes.gpuUsage.textContent = gpu === null ? "—" : `${gpu.toFixed(1)}%`;
      nodes.memoryAmount.textContent = validMemory ? `${(used / 2**30).toFixed(2)} / ${(total / 2**30).toFixed(2)} GiB` : "—";
      nodes.memoryPercent.textContent = memoryPercent === null ? "Unavailable" : `${memoryPercent.toFixed(1)}%`;
      nodes.cpuMeter.style.width = `${cpu ?? 0}%`;
      nodes.gpuMeter.style.width = `${gpu ?? 0}%`;
      nodes.memoryMeter.style.width = `${memoryPercent ?? 0}%`;
      if (cpu === null && gpu === null && !validMemory) { clear("Device metrics unavailable"); return; }
      lastSample = {age, receivedAt: performance.now()};
      const partial = sample.status === "partial" || cpu === null || gpu === null || memoryPercent === null;
      status(partial ? "Partial metrics · some readings unavailable" : "Live · updates every second", partial ? "partial" : "live");
    } catch (_) {
      if (active === controller && document.visibilityState === "visible") clear("Device metrics unavailable");
    } finally {
      clearTimeout(timeout);
      if (active === controller) active = null;
    }
  }
  function tick() {
    if (lastSample && lastSample.age + performance.now() - lastSample.receivedAt > 5000) {
      clear("Stale · waiting for fresh metrics", "stale");
    }
    void poll();
  }
  function pause() {
    clearInterval(interval); interval = null;
    active?.abort();
    clear("Paused · page not visible", "paused");
  }
  function resume() {
    clearInterval(interval); interval = null;
    if (document.visibilityState !== "visible") { pause(); return; }
    clear("Refreshing device metrics…");
    tick(); interval = setInterval(tick, 1000);
  }
  document.addEventListener("visibilitychange", resume);
  window.addEventListener("pagehide", pause);
  window.addEventListener("pageshow", resume);
  resume();
}

if (typeof module !== "undefined") module.exports = {SSEParser, readCompletionEvent, CAPTURE_PRESETS, FrameCadence};

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const state = {ready: false, model: "", running: false, busy: false, media: null,
    abort: null, captureAt: null, completed: 0, imageURL: null, checking: false, cameraGeneration: 0,
    preset: "live-vlm", frameCallback: null, sampled: 0, skipped: 0};
  const canvas = document.createElement("canvas");
  const duration = ms => ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
  function error(message = "") { $("error").textContent = message; $("error").hidden = !message; }
  function controls() {
    $("startButton").disabled = state.running || state.busy;
    $("stopButton").disabled = !state.running && !state.busy;
    $("analyzeButton").disabled = !state.imageURL || !state.ready || state.busy || state.running;
  }
  async function checkBackend() {
    if (state.checking) return;
    state.checking = true;
    try {
      const [health, models] = await Promise.all([
        fetch("/health/ready", {signal: AbortSignal.timeout(5000)}),
        fetch("/v1/models", {signal: AbortSignal.timeout(5000)})
      ]);
      if (!health.ok || !models.ok) throw new Error("unavailable");
      const healthData = await health.json();
      const modelsData = await models.json();
      const model = modelsData.data?.[0]?.id;
      if (healthData.status !== "ready" || !model) throw new Error("not ready");
      state.ready = true; state.model = model;
      $("backendStatus").textContent = "Local backend ready";
      $("backendStatus").className = "badge ready";
      $("modelName").textContent = model;
    } catch (_) {
      state.ready = false;
      $("backendStatus").textContent = "Backend not ready";
      $("backendStatus").className = "badge unavailable";
      $("modelName").textContent = "Waiting for local TensorRT-Edge-LLM";
      if (!state.busy) $("runStatus").textContent = "Waiting for local backend";
    } finally { state.checking = false; controls(); }
  }
  function capture(source) {
    const width = source.videoWidth || source.naturalWidth;
    const height = source.videoHeight || source.naturalHeight;
    if (!width || !height) throw new Error("The image is not ready yet.");
    const preset = CAPTURE_PRESETS[state.preset];
    const limit = state.preset === "lightweight" ? Number($("resolution").value) : null;
    const ratio = limit === null ? 1 : Math.min(1, limit / Math.max(width, height));
    const imageWidth = Math.max(1, Math.round(width * ratio));
    const imageHeight = Math.max(1, Math.round(height * ratio));
    // Preserve the whole image without upscaling while keeping both canvas
    // axes in the supported visual position grid (at least 16 patches).
    canvas.width = Math.max(256, imageWidth);
    canvas.height = Math.max(256, imageHeight);
    if (state.preset === "live-vlm" && (canvas.width > 4096 || canvas.height > 4096 ||
        Math.max(canvas.width, canvas.height) / Math.min(canvas.width, canvas.height) > 8)) {
      throw new Error("This full-size image is too large or too wide for the selected model profile. Choose a source up to 4096 pixels per side and an aspect ratio within 8:1, or use Lightweight.");
    }
    const context = canvas.getContext("2d", {alpha: false});
    context.fillStyle = "#000";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(source, Math.floor((canvas.width - imageWidth) / 2),
      Math.floor((canvas.height - imageHeight) / 2), imageWidth, imageHeight);
    const capturedAt = performance.now();
    const url = canvas.toDataURL("image/jpeg", preset.jpegQuality);
    if (url.length > 1900000) throw new Error("This full-size frame exceeds the image request limit. Choose a smaller source or the Lightweight preset.");
    $("captureStatus").textContent = `Sent ${canvas.width}×${canvas.height} · ${state.preset === "live-vlm" ? "Live VLM WebUI" : "Lightweight"}`;
    return {url, capturedAt};
  }
  async function analyze(source) {
    if (state.busy || !state.ready) return;
    const prompt = $("prompt").value.trim();
    const maxTokens = Number($("maxTokens").value);
    if (!prompt || !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 512) {
      error("Enter a prompt and an output token limit from 1 to 512.");
      stop(); return;
    }
    const controller = new AbortController();
    state.busy = true; state.abort = controller; controls(); error();
    $("ttft").textContent = "—"; $("totalTime").textContent = "—";
    $("runStatus").textContent = "Reading frame…"; $("answer").textContent = "";
    let reader;
    try {
      const image = capture(source);
      state.captureAt = image.capturedAt;
      if (state.running && source === $("video")) state.sampled += 1;
      const started = performance.now();
      let firstToken = null, done = false, output = "", finishReason = null;
      const response = await fetch("/v1/chat/completions", {
        method: "POST", headers: {"Content-Type": "application/json"}, signal: controller.signal,
        body: JSON.stringify({model: state.model, stream: true, temperature: CAPTURE_PRESETS[state.preset].temperature,
          ...(state.preset === "lightweight" ? {top_p: 1} : {}),
          stream_options: {include_usage: true},
          max_tokens: maxTokens, messages: [{role: "user", content: [
            {type: "text", text: prompt}, {type: "image_url", image_url: {url: image.url}}
          ]}]})
      });
      controller.signal.throwIfAborted();
      if (state.abort !== controller) return;
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error?.message || `Backend returned HTTP ${response.status}.`);
      }
      if (!response.headers.get("content-type")?.includes("text/event-stream")) throw new Error("Backend did not return a token stream.");
      if (!response.body) throw new Error("Streaming responses are not supported in this browser.");
      const decoder = new TextDecoder();
      const parser = new SSEParser(event => {
        const completion = readCompletionEvent(event);
        if (completion.done) { done = true; return; }
        if (completion.finishReason) finishReason = completion.finishReason;
        const text = completion.text;
        if (typeof text === "string" && text.length) {
          if (firstToken === null) { firstToken = performance.now(); $("ttft").textContent = duration(firstToken - started); }
          output += text;
          if (output.length > 65536) throw new Error("Model output exceeded the display limit.");
          $("answer").textContent = output;
          $("answer").classList.add("streaming");
          $("runStatus").textContent = "Writing answer…";
        }
      });
      reader = response.body.getReader();
      while (!done) {
        const chunk = await reader.read();
        controller.signal.throwIfAborted();
        if (state.abort !== controller) return;
        if (chunk.done) { parser.feed(decoder.decode(), true); break; }
        parser.feed(decoder.decode(chunk.value, {stream: true}));
      }
      if (!done) throw new Error("Connection ended before the model completed its answer.");
      if (!["stop", "length"].includes(finishReason)) throw new Error("Backend stream did not confirm a successful completion.");
      if (!output.trim()) throw new Error("The backend completed without visible answer text. Try a larger output token limit.");
      $("totalTime").textContent = duration(performance.now() - started);
      $("runStatus").textContent = finishReason === "length" ? "Output token limit reached" : "Answer complete";
      state.completed += 1; $("requestCount").textContent = `${state.completed} completed`;
    } catch (err) {
      if (state.abort !== controller) return;
      if (err.name === "AbortError" || controller.signal.aborted) $("runStatus").textContent = "Stopped · partial answer";
      else { error(err.message); $("runStatus").textContent = "Request failed"; state.running = false; }
    } finally {
      // Browser cancellation can remain pending after a fetch abort. It must
      // never hold the controls busy or clean up a later request's state.
      try { if (reader) reader.cancel().catch(() => {}); } catch (_) {}
      if (state.abort === controller) {
        state.abort = null; state.busy = false;
        $("answer").classList.remove("streaming");
        if (!state.running) releaseCamera();
        controls();
      }
    }
  }
  async function cameraLoop(generation) {
    while (state.running && generation === state.cameraGeneration) {
      const start = performance.now();
      if (state.ready && $("video").readyState >= 2) await analyze($("video"));
      const delay = Math.max(100, Number($("interval").value) - (performance.now() - start));
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  function frameCameraLoop(generation) {
    const cadence = new FrameCadence(30);
    const video = $("video");
    function frame(now, metadata) {
      if (!state.running || generation !== state.cameraGeneration) return;
      state.frameCallback = null;
      if (cadence.observe(metadata.presentedFrames)) {
        if (state.busy) state.skipped += 1;
        else if (state.ready && video.readyState >= 2) {
          void analyze(video);
        }
        $("samplingStatus").textContent = `${state.sampled} frames sent · ${state.skipped} skipped while busy`;
      }
      if (state.running && generation === state.cameraGeneration) state.frameCallback = video.requestVideoFrameCallback(frame);
    }
    state.frameCallback = video.requestVideoFrameCallback(frame);
  }
  function releaseCamera() {
    if (state.frameCallback !== null) $("video").cancelVideoFrameCallback?.(state.frameCallback);
    state.frameCallback = null;
    if (state.media) state.media.getTracks().forEach(track => track.stop());
    state.media = null; $("video").srcObject = null; $("liveIndicator").hidden = true;
    $("sourceStatus").textContent = state.imageURL ? "Selected image" : "Camera stopped";
  }
  function stop() { state.running = false; state.cameraGeneration += 1; state.abort?.abort(); releaseCamera(); controls(); }
  function applyPreset(name, reset = true) {
    if (!CAPTURE_PRESETS[name]) return;
    if (reset) {
      stop();
      // Retire the aborted owner before clearing its result. Its late stream
      // cleanup must never overwrite the new preset or a restarted request.
      state.abort = null; state.busy = false;
      state.completed = 0; state.captureAt = null;
      $("answer").textContent = "Preset changed. Start the camera or analyze your selected image.";
      $("answer").classList.remove("streaming");
      $("runStatus").textContent = "Waiting for input";
      $("requestCount").textContent = "0 completed";
      for (const id of ["ttft", "totalTime", "frameAge"]) $(id).textContent = "—";
      $("captureStatus").textContent = "No frame sent"; error();
    }
    state.preset = name; state.sampled = 0; state.skipped = 0;
    const live = name === "live-vlm";
    $("liveVlmPreset").checked = live; $("lightweightPreset").checked = !live;
    $("interval").value = live ? "frames30" : "1000";
    $("resolution").value = live ? "native" : "512";
    $("interval").disabled = live; $("resolution").disabled = live;
    $("presetDescription").textContent = live
      ? "Requests a 1280×720 camera with no FPS cap. Sends full-size frames every 30 video frames and skips samples while busy. Temperature 0.7."
      : "Requests a 640×480 camera at 15 FPS. Defaults to 512-pixel frames and a 1-second minimum interval, adjustable below. Captures after each answer. Temperature 0.";
    $("samplingStatus").textContent = live ? "Every 30 frames · waiting for camera" : "Capture after each answer";
    controls();
  }
  $("liveVlmPreset").addEventListener("change", () => { if ($("liveVlmPreset").checked) applyPreset("live-vlm"); });
  $("lightweightPreset").addEventListener("change", () => { if ($("lightweightPreset").checked) applyPreset("lightweight"); });
  $("startButton").addEventListener("click", async () => {
    error();
    if (!navigator.mediaDevices?.getUserMedia) { error("Camera access needs HTTPS or http://localhost. You can choose an image instead."); return; }
    if (state.preset === "live-vlm" && !$("video").requestVideoFrameCallback) {
      error("This browser cannot count video frames. Use a current browser or choose Lightweight."); return;
    }
    state.running = true; const generation = ++state.cameraGeneration; controls();
    try {
      const media = await navigator.mediaDevices.getUserMedia({audio: false,
        video: CAPTURE_PRESETS[state.preset].cameraConstraints});
      if (!state.running || generation !== state.cameraGeneration) { media.getTracks().forEach(track => track.stop()); return; }
      state.media = media; $("video").srcObject = media; await $("video").play();
      $("video").hidden = false; $("uploadedImage").hidden = true; $("placeholder").hidden = true;
      $("liveIndicator").hidden = false;
      const actual = media.getVideoTracks()[0].getSettings();
      $("sourceStatus").textContent = `Camera · ${$("video").videoWidth}×${$("video").videoHeight}${actual.frameRate ? ` · ${actual.frameRate.toFixed(1)} FPS` : ""}`;
      state.sampled = 0; state.skipped = 0;
      $("samplingStatus").textContent = state.preset === "live-vlm" ? "Every 30 frames · waiting for first sample" : "Capture after each answer";
      if (state.preset === "live-vlm") frameCameraLoop(generation); else cameraLoop(generation);
    } catch (err) { if (generation === state.cameraGeneration) { stop(); error(`Camera unavailable: ${err.message}. You can choose an image instead.`); } }
  });
  $("stopButton").addEventListener("click", stop);
  $("analyzeButton").addEventListener("click", () => analyze($("uploadedImage")));
  $("imageInput").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    stop(); error();
    if (!new Set(["image/jpeg", "image/png", "image/webp"]).has(file.type) || file.size > 12 * 1024 * 1024) { error("Choose a JPEG, PNG or WebP image smaller than 12 MiB."); return; }
    if (state.imageURL) URL.revokeObjectURL(state.imageURL);
    state.imageURL = URL.createObjectURL(file);
    $("uploadedImage").src = state.imageURL;
    try {
      await $("uploadedImage").decode();
      $("uploadedImage").hidden = false; $("video").hidden = true; $("placeholder").hidden = true;
      $("sourceStatus").textContent = "Selected image"; controls();
    } catch (_) { error("This image could not be decoded."); URL.revokeObjectURL(state.imageURL); state.imageURL = null; controls(); }
  });
  window.addEventListener("pagehide", stop);
  setInterval(() => { if (state.captureAt !== null) $("frameAge").textContent = duration(performance.now() - state.captureAt); }, 100);
  applyPreset("live-vlm", false);
  setInterval(checkBackend, 5000); checkBackend(); controls();
  if (!window.isSecureContext) {
    $("cameraHelp").textContent = "Image upload works here. Camera access requires HTTPS.";
    fetch("/api/access", {signal: AbortSignal.timeout(3000)}).then(r => r.json()).then(access => {
      if (!Number.isInteger(access.https_port) || access.https_port < 1 || access.https_port > 65535) return;
      const url = new URL(window.location.href);
      url.protocol = "https:"; url.port = String(access.https_port); url.pathname = "/"; url.search = ""; url.hash = "";
      const link = document.createElement("a");
      link.href = url.href; link.textContent = "Open HTTPS for camera access";
      $("cameraHelp").append(" ", link, ". This device uses a local certificate.");
    }).catch(() => {});
  }
  startDeviceTelemetry();
}
