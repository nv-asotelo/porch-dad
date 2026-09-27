"""Allowlisted, single-resident model services for the Live Vision proxy.

The registry is local administrator configuration, never request-supplied URLs or
commands. Selecting a model drains inference, stops the previous service, then
checks the new service's identity. A failed start restores the previous service.
"""

import hashlib
import http.client
import json
import re
import subprocess
import threading
import time
from pathlib import Path

BROCKONE_PROMPT = (
    "Identify the Pokémon shown in the image. Reply with one short sentence: "
    "This is <Pokémon name>. If no Pokémon is recognizable, reply: "
    "I cannot identify a Pokémon."
)
BROCKONE_POLICY = {"prompt": BROCKONE_PROMPT, "max_tokens": 64,
                   "temperature": 0, "image_tokens": 512, "stream": False}
ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")
UNIT = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.@-]{0,127}(?:\.service)?\Z")


def validate_registry(entries):
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Expected a nonempty model service registry")
    ports, units = set(), set()
    for key, entry in entries.items():
        if not isinstance(key, str) or not ID.fullmatch(key) or not isinstance(entry, dict):
            raise ValueError("Invalid engine identifier")
        if entry.get("kind") != "service" or entry.get("protocol") not in {"cosmos", "brockone"}:
            raise ValueError("Service engines need an explicit cosmos or brockone protocol")
        if not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise ValueError("Engine name is required")
        if not isinstance(entry.get("model_id"), str) or not entry["model_id"].strip():
            raise ValueError("Engine model_id is required")
        port = entry.get("backend_port")
        if type(port) is not int or not 1 <= port <= 65535 or port in ports:
            raise ValueError("Engines need distinct valid loopback ports")
        service = entry.get("service")
        if not isinstance(service, str) or not UNIT.fullmatch(service):
            raise ValueError("Invalid allowlisted systemd service")
        service = service.removesuffix(".service") + ".service"
        if service in units:
            raise ValueError("Each engine requires its own service")
        path = entry.get("path")
        if not isinstance(path, str) or not Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Engine path must be absolute")
        if entry["protocol"] == "brockone":
            if entry["model_id"] != "brockone":
                raise ValueError("The brockone protocol requires model_id brockone")
            mode = entry.get("validation_mode", "validated")
            if mode not in {"validated", "live_trial"}:
                raise ValueError("Unknown brockone validation mode")
            proof, sha = entry.get("readiness_receipt"), entry.get("readiness_sha256")
            if (not isinstance(proof, str) or not Path(proof).is_absolute()
                    or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
                raise ValueError("brockone requires a pinned target readiness receipt")
        ports.add(port)
        units.add(service)
    return entries


class ServiceEngineSwitcher:
    managed = True

    def __init__(self, entries, default, generation_lock, timeout=120):
        self.engines = validate_registry(entries)
        if default not in entries:
            raise ValueError("Default engine is not registered")
        self.selected = default
        self.generation_lock = generation_lock
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.switching = False
        self.timeout = timeout
        # UI restarts must not claim the configured default if another service is
        # actually resident. Reading model identities neither starts nor stops one.
        ready = [key for key in entries if self.availability(key)[0] and self.healthy(key)]
        if len(ready) == 1:
            self.selected = ready[0]

    def availability(self, key):
        entry = self.engines[key]
        if entry.get("enabled", True) is not True:
            return False, entry.get("unavailable_reason", "Not installed yet")
        if not (Path(entry["path"]) / "llm.engine").is_file():
            return False, "Engine has not been built on this device"
        if entry["protocol"] == "brockone":
            try:
                proof = Path(entry["readiness_receipt"]).read_bytes()
                if hashlib.sha256(proof).hexdigest() != entry["readiness_sha256"]:
                    return False, "Target validation receipt differs from the installed configuration"
                value = json.loads(proof)
                if not isinstance(value, dict):
                    return False, "Target admission receipt must be an object"
                if entry.get("validation_mode", "validated") == "live_trial":
                    if (value.get("state") != "orin_engine_live_trial"
                            or value.get("model_id") != "brockone"
                            or value.get("engine_root") != entry["path"]
                            or value.get("user_authorized_validation_bypass") is not True
                            or value.get("validation_performed") is not False
                            or value.get("validation_passed") is not False):
                        return False, "Unvalidated live trial authorization is incomplete"
                    return True, ""
                if (value.get("state") != "orin_engine_validated" or value.get("model_id") != "brockone"
                        or value.get("engine_root") != entry["path"]
                        or value.get("engine_load_passed") is not True
                        or value.get("finite_predictions_passed") is not True
                        or value.get("prepared_pixels_passed") is not True):
                    return False, "Target engine validation is incomplete"
            except (OSError, ValueError, TypeError):
                return False, "Target engine validation is pending"
        return True, ""

    def descriptor(self, key):
        entry = self.engines[key]
        available, reason = self.availability(key)
        trial = entry["protocol"] == "brockone" and entry.get("validation_mode") == "live_trial"
        name = entry["name"] + " (unvalidated trial)" if trial else entry["name"]
        profile = entry.get("profile", "")
        if trial:
            profile = "Unvalidated live trial" + (" · " + profile if profile else "")
        return {"id": key, "name": name, "model_id": entry["model_id"],
                "profile": profile, "available": available, "reason": reason,
                "validation_status": "unvalidated_live_trial" if trial else "standard",
                "request_policy": BROCKONE_POLICY.copy() if entry["protocol"] == "brockone" else None}

    def active(self):
        with self.state_lock:
            key = self.selected
        return self.descriptor(key)

    def status(self):
        return {"configured": True, "active": self.active(), "switching": self.switching,
                "engines": [self.descriptor(key) for key in self.engines]}

    def backend(self):
        with self.state_lock:
            return dict(self.engines[self.selected])

    def healthy(self, key):
        entry = self.engines[key]
        connection = http.client.HTTPConnection("127.0.0.1", entry["backend_port"], timeout=2)
        try:
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            raw = response.read(65537)
            if response.status != 200 or len(raw) > 65536:
                return False
            models = json.loads(raw)
            if entry["model_id"] not in [m.get("id") for m in models.get("data", []) if isinstance(m, dict)]:
                return False
            response.close()
            connection.close()
            connection = http.client.HTTPConnection("127.0.0.1", entry["backend_port"], timeout=2)
            connection.request("GET", "/health" if entry["protocol"] == "brockone" else "/health/ready")
            response = connection.getresponse()
            raw = response.read(65537)
            return response.status == 200 and len(raw) <= 65536 and json.loads(raw).get("status") == "ready"
        except (OSError, http.client.HTTPException, ValueError, AttributeError, TypeError):
            return False
        finally:
            connection.close()

    def command(self, action, key):
        # No shell and no command or unit name from the HTTP request.
        try:
            result = subprocess.run(["sudo", "-n", "systemctl", action, self.engines[key]["service"]],
                                    capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def wait_ready(self, key):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.healthy(key):
                return True
            time.sleep(0.5)
        return False

    def switch(self, key):
        if key not in self.engines:
            return False, "Unknown engine"
        available, reason = self.availability(key)
        if not available:
            return False, reason
        if not self.lock.acquire(blocking=False):
            return False, "Another engine switch is in progress"
        acquired = False
        try:
            self.switching = True
            acquired = self.generation_lock.acquire(timeout=5)
            if not acquired:
                return False, "Inference is still running; stop Live Vision and retry"
            previous = self.selected
            # Stop every other registered model, including an accidentally started
            # second service, before allocating the new model's shared GPU memory.
            for other in self.engines:
                if other != key and not self.command("stop", other):
                    return False, "Could not stop the previous model; no new model was started"
            if key == previous and self.healthy(key):
                return True, f"Already using {self.engines[key]['name']}"
            if self.command("start", key) and self.wait_ready(key):
                with self.state_lock:
                    self.selected = key
                return True, f"Switched to {self.engines[key]['name']}"
            stopped = self.command("stop", key)
            restored = stopped and self.command("start", previous) and self.wait_ready(previous)
            if restored:
                return False, f"New model failed to start; restored {self.engines[previous]['name']}"
            return False, "New model failed to start and recovery needs attention; inference is unavailable"
        finally:
            if acquired:
                self.generation_lock.release()
            self.switching = False
            self.lock.release()


def brockone_request(payload):
    """Fail on stale browser settings, rather than silently changing a user's prompt."""
    if payload.get("model") != "brockone":
        raise ValueError("The model changed; refresh Live Vision before submitting another image")
    content = payload["messages"][0]["content"]
    texts = [part.get("text") for part in content if part.get("type") == "text"]
    if (texts != [BROCKONE_PROMPT] or payload.get("max_tokens") != 64
            or payload.get("temperature") != 0 or payload.get("max_image_tokens_per_image") != 512):
        raise ValueError("brockone uses its fixed Pokémon prompt, 64 output tokens and 512 image tokens")
    return {"model": "brockone", "messages": payload["messages"], "max_tokens": 64,
            "temperature": 0, "image_tokens": 512, "stream": False}


def brockone_sse(value):
    """Frame one completed caption for the UI, without inventing token timings."""
    if not isinstance(value, dict) or value.get("model") != "brockone":
        raise ValueError("Backend returned the wrong model")
    choices = value.get("choices")
    if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
            or not isinstance(choices[0].get("message"), dict)):
        raise ValueError("Invalid brockone response")
    text = choices[0].get("message", {}).get("content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("brockone returned no caption")
    chunk = {"id": value.get("id"), "model": "brockone", "object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": {"content": text},
                          "finish_reason": choices[0].get("finish_reason", "stop")}],
             "usage": value.get("usage", {}),
             "live_vision": {"streaming": False, "token_timing_available": False}}
    return ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode()
