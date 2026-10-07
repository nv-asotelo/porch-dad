#!/usr/bin/env python3
"""Fixture-only tests; never import the feed's device configuration or contact a robot."""
from __future__ import annotations

import __future__
import ast
from pathlib import Path
import secrets
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

FEED = Path(__file__).resolve().parents[1] / "feed/porch_feed.py"
SOURCE = FEED.read_text()


class HTTPError(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail
        super().__init__(detail)


def response(content, status_code=200, **kwargs):
    return SimpleNamespace(content=content, status_code=status_code, **kwargs)


def helpers():
    names = {"require_control", "api_reachy_bridge", "api_reachy_bridge_change",
             "api_reachy", "reachy_latest", "reachy_audio", "lan_link", "reachy_health_summary",
             "service_status"}
    nodes = []
    for node in ast.parse(SOURCE).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    namespace = {
        "HTTPException": HTTPError, "JSONResponse": response, "Response": response,
        "StreamingResponse": response, "secrets": secrets, "CONTROL_TOKEN": "fixture-token",
        "REACHY_CAM": "http://127.0.0.1:8099", "REACHY_WEBUI": "", "REACHY_SESSION": "test",
        "REACHY_LIVE_VISION": "", "LINKS_HOST": "localhost",
        "BRIDGE_CONTROL": Mock(), "requests": SimpleNamespace(get=Mock(), RequestException=OSError),
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(FEED), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


class FeedBridgeRoutes(unittest.TestCase):
    def setUp(self):
        self.ns = helpers()
        self.control = self.ns["BRIDGE_CONTROL"]
        self.released = {"state": "released", "active": False, "released": True,
                         "message": "Bridge released. Resume to reconnect.", "main_pid": 0}
        self.control.status.return_value = self.released
        self.request = SimpleNamespace(headers={"X-Porch-Token": "fixture-token"})

    def test_read_only_status_never_starts_or_stops(self):
        self.assertEqual(self.ns["api_reachy_bridge"](), self.released)
        self.control.change.assert_not_called()

    def test_release_and_resume_return_verified_result(self):
        for action in ("release", "resume"):
            result = dict(self.released, ok=True)
            self.control.change.return_value = result
            out = self.ns["api_reachy_bridge_change"](action, self.request)
            self.assertEqual((out.status_code, out.content), (200, result))
            self.control.change.assert_called_with(action)

    def test_unverified_transition_is_not_http_success(self):
        result = dict(self.released, ok=False, state="transitioning", released=False)
        self.control.change.return_value = result
        out = self.ns["api_reachy_bridge_change"]("release", self.request)
        self.assertEqual((out.status_code, out.content), (503, result))

    def test_actions_require_control_token(self):
        for headers in ({}, {"X-Porch-Token": "wrong"}):
            with self.assertRaises(HTTPError) as caught:
                self.ns["api_reachy_bridge_change"]("release", SimpleNamespace(headers=headers))
            self.assertEqual(caught.exception.status_code, 401)
        self.control.change.assert_not_called()

    def test_unconfigured_control_plane_is_closed(self):
        self.ns["CONTROL_TOKEN"] = ""
        with self.assertRaises(HTTPError) as caught:
            self.ns["api_reachy_bridge_change"]("resume", self.request)
        self.assertEqual(caught.exception.status_code, 503)
        self.control.change.assert_not_called()

    def test_only_two_fixed_actions(self):
        for action in ("restart", "stop", "reboot", "resume; shutdown"):
            with self.assertRaises(HTTPError) as caught:
                self.ns["api_reachy_bridge_change"](action, self.request)
            self.assertEqual(caught.exception.status_code, 404)
        self.control.change.assert_not_called()

    def test_health_reports_release_without_probing_stopped_bridge(self):
        out = self.ns["api_reachy"]()
        self.assertFalse(out["connected"])
        self.assertEqual((out["state"], out["reason"]), ("released", self.released["message"]))
        self.ns["requests"].get.assert_not_called()

    def test_still_and_audio_refuse_released_bridge_without_network_call(self):
        for name in ("reachy_latest", "reachy_audio"):
            with self.assertRaises(HTTPError) as caught:
                self.ns[name]()
            self.assertEqual(caught.exception.status_code, 409)
        self.ns["requests"].get.assert_not_called()

    def test_service_running_is_not_camera_ready(self):
        self.control.status.return_value = dict(self.released, state="active", active=True, released=False)
        self.ns["requests"].get.return_value.json.return_value = {
            "state": "connecting", "live": False, "has_frame": False}
        out = self.ns["api_reachy"]()
        self.assertFalse(out["connected"])
        self.assertEqual(out["state"], "connecting")

    def test_audio_proxy_closes_upstream_when_consumer_closes(self):
        self.control.status.return_value = {"active": True}
        upstream = self.ns["requests"].get.return_value
        upstream.status_code = 200
        upstream.iter_content.return_value = iter([b"one", b"two"])
        body = self.ns["reachy_audio"]().content
        self.assertEqual(next(body), b"one")
        body.close()
        upstream.close.assert_called_once()

    def test_newer_shared_release_overrides_old_services_intent(self):
        prior = {"desired": "running", "at": 100.0, "by": "user", "note": "started"}
        shared = {"desired": "stopped", "at": 101.0, "by": "bridge controls", "note": "released"}
        self.ns.update(SERVICES={"reachy": {"kind": "systemd", "unit": "reachy-mjpeg-bridge"}},
                       unit_state=lambda key: "inactive", port_open=lambda port: None,
                       load_intent=lambda: {"reachy": prior})
        self.control.intent.return_value = shared
        out = self.ns["service_status"]("reachy")
        self.assertEqual((out["intent"], out["intent_at"], out["intent_by"]),
                         ("stopped", 101.0, "bridge controls"))
        self.assertFalse(out["contradicts_intent"])

    def test_later_generic_service_action_wins_over_shared_release(self):
        self.ns.update(SERVICES={"reachy": {"kind": "systemd", "unit": "reachy-mjpeg-bridge.service"}},
                       unit_state=lambda key: "active", port_open=lambda port: None,
                       load_intent=lambda: {"reachy": {"desired": "running", "at": 102.0}})
        self.control.intent.return_value = {"desired": "stopped", "at": 101.0}
        out = self.ns["service_status"]("reachy")
        self.assertEqual((out["intent"], out["intent_at"]), ("running", 102.0))
        self.assertFalse(out["contradicts_intent"])

    def test_absent_shared_intent_preserves_legacy_record(self):
        self.ns.update(SERVICES={"reachy": {"kind": "systemd", "unit": "reachy-mjpeg-bridge.service"}},
                       unit_state=lambda key: "inactive", port_open=lambda port: None,
                       load_intent=lambda: {"reachy": {"desired": "stopped", "at": 100.0}})
        self.control.intent.return_value = {}
        out = self.ns["service_status"]("reachy")
        self.assertEqual((out["intent"], out["intent_at"]), ("stopped", 100.0))

    def test_shared_intent_does_not_change_another_service(self):
        self.ns.update(SERVICES={"model": {"kind": "systemd", "unit": "model.service"}},
                       unit_state=lambda key: "active", port_open=lambda port: None,
                       load_intent=lambda: {"model": {"desired": "running", "at": 100.0}})
        out = self.ns["service_status"]("model")
        self.assertEqual(out["intent"], "running")
        self.control.intent.assert_not_called()


class FeedBridgeBrowser(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js needed for fixture DOM checks")
    def test_release_resume_and_external_status_changes(self):
        code = SOURCE.split("let reachyBridgeSnapshot =", 1)[1].split("// Reflects /api/reachy/alert", 1)[0]
        code = "let reachyBridgeSnapshot =" + code
        harness = r"""
const assert = require('node:assert/strict');
const elements = new Map();
const document = {getElementById(id){
  if(!elements.has(id)) elements.set(id, {style:{}, src:'', checked:false, disabled:false,
    removeAttribute(name){delete this[name]}, pause(){this.paused=true}, load(){},
    play(){return Promise.resolve()}});
  return elements.get(id);
}};
const messages = [], requests = [];
const say = (message, ok) => messages.push({message,ok});
const tokenReady = async () => 'fixture-token';
let state = {active:false,released:true,state:'released',message:'Released. Resume to reconnect.'};
let refuse = false;
let fetch = async (url, options={}) => {
  requests.push({url,...options});
  if(options.method==='POST'){
    assert.equal(options.headers['X-Porch-Token'], 'fixture-token');
    if(refuse) return {ok:false,json:async()=>({...state,ok:false,message:'Stop timed out'})};
    state = url.endsWith('/release')
      ? {active:false,released:true,state:'released',message:'Released. Resume to reconnect.'}
      : {active:true,released:false,state:'active',message:'Bridge running; verify a fresh camera frame.'};
    return {ok:true,json:async()=>({...state,ok:true})};
  }
  return {ok:true,json:async()=>({...state})};
};
"""
        checks = r"""
(async()=>{
  await refreshReachyBridge();
  assert.ok(requests.every(r=>r.method!=='POST'), 'page load is read only');
  assert.equal(reachyBridgeBlocked(), true);
  assert.equal(document.getElementById('reachyBridgeResume').disabled, false);
  assert.equal(document.getElementById('reachyListenBtn').disabled, true);
  await reachyBridgeChange('resume');
  assert.equal(reachyBridgeBlocked(), false);
  assert.equal(document.getElementById('reachyBridgeResume').disabled, true);
  assert.ok(!document.getElementById('reachyImg').src, 'service start is not a fresh frame');
  document.getElementById('reachyImg').src='fixture.jpg';
  document.getElementById('reachyAudioEl').src='fixture.mp3';
  document.getElementById('reachyListenBtn').checked=true;
  const release=reachyBridgeChange('release');
  assert.ok(!document.getElementById('reachyImg').src, 'release immediately clears preview');
  assert.ok(!document.getElementById('reachyAudioEl').src, 'release immediately closes audio');
  await release;
  assert.equal(document.getElementById('reachyListenBtn').checked,false);
  state={active:true,released:false,state:'active',message:'Bridge running; verify frames.'};
  await refreshReachyBridge();
  assert.equal(reachyBridgeBlocked(),false,'detect resume in another UI');
  state={active:false,released:true,state:'released',message:'Released elsewhere.'};
  await refreshReachyBridge();
  assert.equal(reachyBridgeBlocked(),true,'detect release in another UI');
  for(const name of ['transitioning','unavailable']){
    state={active:false,released:false,state:name,message:'Not ready for another action.'};
    await refreshReachyBridge();
    assert.equal(document.getElementById('reachyBridgeResume').disabled,true,
      `resume remains disabled while ${name}`);
  }
  for(const name of ['released','failed']){
    state={active:false,released:name==='released',state:name,message:'An explicit resume is available.'};
    await refreshReachyBridge();
    assert.equal(document.getElementById('reachyBridgeResume').disabled,false,
      `resume is available while ${name}`);
  }
  const before=requests.length;
  await reachyBridgeChange('reboot');
  assert.equal(requests.length,before,'unknown action does nothing');
  refuse=true;
  await reachyBridgeChange('resume');
  assert.deepEqual(messages.at(-1),{message:'Stop timed out',ok:false});
  assert.equal(reachyBridgeBlocked(),true,'a failed action is not reported resumed');
  refuse=false;
  const originalFetch=fetch;
  let resolveOld;
  fetch=(url,options)=>new Promise(resolve=>{resolveOld=resolve});
  const oldPoll=refreshReachyBridge();
  fetch=originalFetch;
  await reachyBridgeChange('release');
  resolveOld({ok:true,json:async()=>({active:true,released:false,state:'active'})});
  await oldPoll;
  assert.equal(reachyBridgeBlocked(),true,'stale poll cannot undo release');
})().catch(error=>{console.error(error);process.exitCode=1});
"""
        result = subprocess.run([shutil.which("node"), "-"], input=harness + code + checks,
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js needed for page syntax check")
    def test_full_page_javascript_syntax(self):
        source = SOURCE.rsplit("<script>", 1)[1].split("</script>", 1)[0]
        result = subprocess.run([shutil.which("node"), "--check", "-"], input=source,
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
