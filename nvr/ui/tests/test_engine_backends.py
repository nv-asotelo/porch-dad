"""Engine lifecycle and real HTTP proxy contracts; no model, systemd or GPU work."""
import copy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import engine_backends as engines
spec = importlib.util.spec_from_file_location('engine_test_ui', SCRIPTS / 'serve_ui.py')
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


def payload(model='nvidia/Cosmos3-Edge'):
    return {'model': model, 'stream': True, 'temperature': 0, 'max_tokens': 64,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'Describe the scene.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/2Q=='}}]}]}


class RegistryTests(unittest.TestCase):
    """Switch mechanics between two registered engines - "alt" stands in for any second
    service, not a specific model; nothing here depends on which one it is."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for name in ('cosmos', 'alt'):
            (root / name).mkdir()
            (root / name / 'llm.engine').write_bytes(b'synthetic engine fixture')
        self.entries = {
            'cosmos': {'kind': 'service', 'protocol': 'cosmos', 'name': 'Cosmos3-Edge',
                       'model_id': 'nvidia/Cosmos3-Edge', 'service': 'cosmos.service',
                       'backend_port': 8001, 'path': str(root / 'cosmos')},
            'alt': {'kind': 'service', 'protocol': 'cosmos', 'name': 'Alt',
                    'model_id': 'alt-model', 'service': 'alt.service',
                    'backend_port': 8002, 'path': str(root / 'alt')}}
        with mock.patch.object(engines.ServiceEngineSwitcher, 'healthy', return_value=False):
            self.switcher = engines.ServiceEngineSwitcher(self.entries, 'cosmos', threading.Lock(), timeout=0.01)

    def test_config_rejects_commands_shared_ports_and_bad_paths(self):
        for key, value in [('service', 'cosmos; touch /tmp/bad'), ('backend_port', 8001),
                           ('model_id', ''), ('path', '/a/../b'), ('protocol', 'other')]:
            entries = copy.deepcopy(self.entries)
            entries['alt'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                engines.validate_registry(entries)

    def test_unbuilt_engine_is_unavailable_and_never_stops_existing(self):
        (Path(self.entries['alt']['path']) / 'llm.engine').unlink()
        with mock.patch.object(self.switcher, 'command') as command:
            ok, reason = self.switcher.switch('alt')
        self.assertFalse(ok)
        self.assertIn('not been built', reason)
        command.assert_not_called()
        self.assertEqual(self.switcher.selected, 'cosmos')

    def test_one_resident_order_and_new_identity(self):
        with mock.patch.object(self.switcher, 'command', return_value=True) as command, \
             mock.patch.object(self.switcher, 'wait_ready', return_value=True):
            self.assertTrue(self.switcher.switch('alt')[0])
        self.assertEqual(command.call_args_list, [mock.call('stop', 'cosmos'), mock.call('start', 'alt')])
        self.assertEqual(self.switcher.backend()['model_id'], 'alt-model')
        self.assertFalse(self.switcher.switching)
        self.assertFalse(self.switcher.generation_lock.locked())

    def test_failed_start_rolls_back_before_returning(self):
        with mock.patch.object(self.switcher, 'command', return_value=True) as command, \
             mock.patch.object(self.switcher, 'wait_ready', side_effect=[False, True]):
            ok, message = self.switcher.switch('alt')
        self.assertFalse(ok)
        self.assertIn('restored Cosmos3-Edge', message)
        self.assertEqual(self.switcher.selected, 'cosmos')
        self.assertEqual(command.call_args_list, [mock.call('stop', 'cosmos'), mock.call('start', 'alt'),
                                                mock.call('stop', 'alt'), mock.call('start', 'cosmos')])

    def test_failed_stop_does_not_start_new_service(self):
        with mock.patch.object(self.switcher, 'command', return_value=False) as command:
            self.assertFalse(self.switcher.switch('alt')[0])
        command.assert_called_once_with('stop', 'cosmos')

    def test_inference_and_concurrent_switch_are_excluded(self):
        self.switcher.generation_lock = mock.Mock()
        self.switcher.generation_lock.acquire.return_value = False
        with mock.patch.object(self.switcher, 'command') as command:
            self.assertFalse(self.switcher.switch('alt')[0])
        command.assert_not_called()
        self.switcher.generation_lock.release.assert_not_called()
        self.switcher.lock.acquire()
        try:
            self.assertFalse(self.switcher.switch('alt')[0])
        finally:
            self.switcher.lock.release()

    def test_unknown_or_active_choice_does_not_restart(self):
        with mock.patch.object(self.switcher, 'command') as command, \
             mock.patch.object(self.switcher, 'healthy', return_value=True):
            self.assertFalse(self.switcher.switch('unknown')[0])
            self.assertTrue(self.switcher.switch('cosmos')[0])
        command.assert_called_once_with('stop', 'alt')

    def test_disabled_healthy_service_is_not_admitted_at_startup(self):
        self.entries['alt']['enabled'] = False
        with mock.patch.object(engines.ServiceEngineSwitcher, 'healthy', side_effect=lambda key: key == 'alt'):
            switcher = engines.ServiceEngineSwitcher(self.entries, 'cosmos', threading.Lock())
        self.assertEqual(switcher.selected, 'cosmos')


class LogprobRequestTests(unittest.TestCase):
    """The Name-the-Pokémon presets ask the shim for per-token logprobs; nothing else may get through."""

    def test_logprobs_are_bounded(self):
        ui.validate_request({**payload(), 'logprobs': True, 'top_logprobs': 5})
        ui.validate_request({**payload(), 'logprobs': False})
        for extra in [{'logprobs': 1}, {'top_logprobs': 1}, {'logprobs': True, 'top_logprobs': 6},
                      {'logprobs': True, 'top_logprobs': True}, {'logprobs': True, 'top_logprobs': -1}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                ui.validate_request({**payload(), **extra})


class SymlinkSwitcherTests(unittest.TestCase):
    """The legacy switcher: one shim, an engine symlink, `sudo ln -sfn` and a restart."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for name in ('v3', 'v2', 'fp16'):
            (root / name).mkdir()
            (root / name / 'llm.engine').write_bytes(b'synthetic engine fixture')
        self.link = root / 'default'
        self.link.symlink_to(root / 'v3')
        self.entries = {
            'v3': {'name': 'Slow', 'path': str(root / 'v3')},
            'v2': {'name': 'Fast', 'path': str(root / 'v2')},
            'fp16': {'name': 'FP16', 'path': str(root / 'fp16'), 'enabled': False,
                     'unavailable_reason': 'Does not fit beside the NVR'}}
        self.switcher = ui.EngineSwitcher(self.link, self.entries, 'shim', 1)
        self.calls = []

    def fake_sudo(self, cmd, timeout=180):
        """Stands in for sudo: records each command line and performs the relink."""
        self.calls.append(cmd)
        if cmd[2] == 'ln':
            self.link.unlink()
            self.link.symlink_to(cmd[4])
        return True, ''

    def switch(self, eid, ready):
        with mock.patch.object(self.switcher, '_run', side_effect=self.fake_sudo), \
             mock.patch.object(self.switcher, 'wait_ready', side_effect=ready):
            return self.switcher.switch(eid)

    def test_switch_runs_exactly_the_sudoers_command_lines(self):
        ok, message = self.switch('v2', [True])
        self.assertTrue(ok, message)
        self.assertEqual(self.calls, [['sudo', '-n', 'ln', '-sfn', self.entries['v2']['path'], str(self.link)],
                                      ['sudo', '-n', 'systemctl', 'restart', 'shim']])
        self.assertEqual(self.switcher.active()['id'], 'v2')
        self.assertFalse(self.switcher.switching)

    def test_disabled_or_unbuilt_build_is_refused_without_a_command(self):
        self.assertEqual(self.switcher.availability('fp16'), (False, 'Does not fit beside the NVR'))
        self.assertEqual(self.switch('fp16', [True]), (False, 'Does not fit beside the NVR'))
        (Path(self.entries['v2']['path']) / 'llm.engine').unlink()
        ok, reason = self.switch('v2', [True])
        self.assertFalse(ok)
        self.assertIn('not been built', reason)
        self.assertEqual(self.calls, [])

    def test_build_that_never_answers_is_swapped_back(self):
        ok, message = self.switch('v2', [False, True])
        self.assertFalse(ok)
        self.assertIn('restored Slow', message)
        self.assertEqual(self.switcher.active()['id'], 'v3')
        self.assertEqual([call[4] for call in self.calls if call[2] == 'ln'],
                         [self.entries['v2']['path'], self.entries['v3']['path']])

    def test_active_answering_build_is_not_restarted(self):
        ok, message = self.switch('v3', [True])
        self.assertTrue(ok)
        self.assertIn('already active', message)
        self.assertEqual(self.calls, [])

    def test_engine_list_reports_refused_builds(self):
        class Quiet(ui.Handler):
            def log_message(self, *args): pass
        class Telemetry:
            def snapshot(self): return {}
            def close(self): pass
        server = ui.Server(('127.0.0.1', 0), Quiet, telemetry=Telemetry())
        server.engine_switcher = self.switcher
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        connection.request('GET', '/api/engines')
        data = json.loads(connection.getresponse().read())
        connection.close()
        self.assertEqual(data['active']['id'], 'v3')
        engines_by_id = {engine['id']: engine for engine in data['engines']}
        self.assertEqual((engines_by_id['v2']['available'], engines_by_id['v2']['reason']), (True, ''))
        self.assertEqual((engines_by_id['fp16']['available'], engines_by_id['fp16']['reason']),
                         (False, 'Does not fit beside the NVR'))


class FakeModel(BaseHTTPRequestHandler):
    """A real OpenAI-shaped SSE backend - proxy() (serve_ui.py) requires an actual
    text/event-stream response for a managed engine, not a plain JSON object."""

    def log_message(self, *args):
        pass

    def answer(self, value, content_type='application/json'):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/v1/models':
            self.answer({'data': [{'id': self.server.model_id}]})
        else:
            self.answer({'status': 'ready', 'configuration': {'image_tokens': 512}})

    def do_POST(self):
        value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.received.append(value)
        if self.server.malformed is not None:
            self.answer(self.server.malformed)
            return
        chunk = {'model': self.server.model_id,
                 'choices': [{'index': 0, 'delta': {'content': 'A scene.'}, 'finish_reason': 'stop'}]}
        data = ('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.model = ThreadingHTTPServer(('127.0.0.1', 0), FakeModel)
        self.model.model_id = 'nvidia/Cosmos3-Edge'; self.model.received = []; self.model.malformed = None
        threading.Thread(target=self.model.serve_forever, daemon=True).start()
        self.switcher = mock.Mock(managed=True, switching=False)
        self.switcher.backend.return_value = {'protocol': 'cosmos', 'model_id': 'nvidia/Cosmos3-Edge',
                                              'backend_port': self.model.server_port}
        self.switcher.active.return_value = {'id': 'cosmos', 'request_policy': None,
                                             'available': True, 'reason': ''}
        class Telemetry:
            def snapshot(self): return {}
            def close(self): pass
        class Quiet(ui.Handler):
            def log_message(self, *args): pass
        self.server = ui.Server(('127.0.0.1', 0), Quiet, telemetry=Telemetry())
        self.server.backend_port = 1
        self.server.engine_switcher = self.switcher
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close()
        self.model.shutdown(); self.model.server_close()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    def test_real_sse_proxy_and_stale_model_rejection(self):
        code, raw = self.request('POST', '/v1/chat/completions', json.dumps(payload()), {'Content-Type': 'application/json'})
        self.assertEqual(code, 200, raw)
        self.assertIn(b'A scene.', raw)
        self.assertEqual(self.model.received[0]['model'], 'nvidia/Cosmos3-Edge')
        stale = payload(model='some-other-model')
        self.assertEqual(self.request('POST', '/v1/chat/completions', json.dumps(stale), {'Content-Type': 'application/json'})[0], 409)
        self.assertEqual(len(self.model.received), 1)

    def test_health_and_switch_unavailable(self):
        self.assertEqual(json.loads(self.request('GET', '/health/ready')[1])['status'], 'ready')
        self.switcher.switching = True
        self.assertEqual(self.request('GET', '/v1/models')[0], 503)

    def test_unqualified_running_backend_is_not_served(self):
        self.switcher.active.return_value['available'] = False
        self.switcher.active.return_value['reason'] = 'Engine has not been built on this device'
        for path in ['/v1/models', '/health/ready']:
            self.assertEqual(self.request('GET', path)[0], 503)
        self.assertEqual(self.request('POST', '/v1/chat/completions', json.dumps(payload()),
                                      {'Content-Type': 'application/json'})[0], 503)
        self.assertEqual(self.model.received, [])

    def test_malformed_success_becomes_bounded_502(self):
        self.model.malformed = {'error': 'not sse'}
        code, raw = self.request('POST', '/v1/chat/completions', json.dumps(payload()),
                                 {'Content-Type': 'application/json'})
        self.assertEqual(code, 502)
        self.assertIn('message', json.loads(raw)['error'])

    def test_switch_requires_same_origin_token(self):
        self.assertEqual(self.request('POST', '/api/engines/cosmos')[0], 403)
        self.switcher.switch.assert_not_called()
        self.switcher.switch.return_value = (True, 'Switched')
        code, raw = self.request('POST', '/api/engines/cosmos', headers={'X-Reachy-Token': ui.RELAY_TOKEN})
        self.assertEqual(code, 200, raw)
        self.switcher.switch.assert_called_once_with('cosmos')

    def test_health_checks_actual_model_identity(self):
        instance = object.__new__(engines.ServiceEngineSwitcher)
        instance.engines = {'c': {'backend_port': self.model.server_port, 'model_id': 'nvidia/Cosmos3-Edge', 'protocol': 'cosmos'}}
        self.assertTrue(instance.healthy('c'))
        self.model.model_id = 'wrong-model'
        self.assertFalse(instance.healthy('c'))


if __name__ == '__main__':
    unittest.main()
