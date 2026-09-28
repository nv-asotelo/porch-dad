"""Engine lifecycle and real HTTP proxy contracts; no model, systemd or GPU work."""
import copy
import hashlib
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


def payload():
    return {'model': 'brockone', 'stream': True, 'temperature': 0, 'max_tokens': 64,
            'max_image_tokens_per_image': 512,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': engines.BROCKONE_PROMPT},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/2Q=='}}]}]}


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for name in ('cosmos', 'brockone'):
            (root / name).mkdir()
            (root / name / 'llm.engine').write_bytes(b'synthetic engine fixture')
        proof = root / 'proof.json'
        proof.write_text(json.dumps({'state': 'orin_engine_validated', 'model_id': 'brockone',
                                     'engine_root': str(root / 'brockone'), 'engine_load_passed': True,
                                     'finite_predictions_passed': True, 'prepared_pixels_passed': True}))
        self.entries = {
            'cosmos': {'kind': 'service', 'protocol': 'cosmos', 'name': 'Cosmos3-Edge',
                       'model_id': 'nvidia/Cosmos3-Edge', 'service': 'cosmos.service',
                       'backend_port': 8001, 'path': str(root / 'cosmos')},
            'brockone': {'kind': 'service', 'protocol': 'brockone', 'name': 'brockone',
                         'model_id': 'brockone', 'service': 'brockone.service',
                         'backend_port': 8092, 'path': str(root / 'brockone'),
                         'readiness_receipt': str(proof), 'readiness_sha256': hashlib.sha256(proof.read_bytes()).hexdigest()}}
        with mock.patch.object(engines.ServiceEngineSwitcher, 'healthy', return_value=False):
            self.switcher = engines.ServiceEngineSwitcher(self.entries, 'cosmos', threading.Lock(), timeout=0.01)

    def test_config_rejects_commands_shared_ports_and_missing_proofs(self):
        for key, value in [('service', 'cosmos; touch /tmp/bad'), ('backend_port', 8001),
                           ('readiness_sha256', 'invented'), ('model_id', 'other'), ('path', '/a/../b')]:
            entries = copy.deepcopy(self.entries)
            entries['brockone'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                engines.validate_registry(entries)

    def test_unbuilt_or_unqualified_never_stops_existing(self):
        Path(self.entries['brockone']['readiness_receipt']).write_text('{}')
        with mock.patch.object(self.switcher, 'command') as command:
            ok, reason = self.switcher.switch('brockone')
        self.assertFalse(ok)
        self.assertIn('receipt', reason)
        command.assert_not_called()
        self.assertEqual(self.switcher.selected, 'cosmos')

    def test_one_resident_order_and_new_identity(self):
        with mock.patch.object(self.switcher, 'command', return_value=True) as command, \
             mock.patch.object(self.switcher, 'wait_ready', return_value=True):
            self.assertTrue(self.switcher.switch('brockone')[0])
        self.assertEqual(command.call_args_list, [mock.call('stop', 'cosmos'), mock.call('start', 'brockone')])
        self.assertEqual(self.switcher.backend()['model_id'], 'brockone')
        self.assertFalse(self.switcher.switching)
        self.assertFalse(self.switcher.generation_lock.locked())

    def test_failed_start_rolls_back_before_returning(self):
        with mock.patch.object(self.switcher, 'command', return_value=True) as command, \
             mock.patch.object(self.switcher, 'wait_ready', side_effect=[False, True]):
            ok, message = self.switcher.switch('brockone')
        self.assertFalse(ok)
        self.assertIn('restored Cosmos3-Edge', message)
        self.assertEqual(self.switcher.selected, 'cosmos')
        self.assertEqual(command.call_args_list, [mock.call('stop', 'cosmos'), mock.call('start', 'brockone'),
                                                mock.call('stop', 'brockone'), mock.call('start', 'cosmos')])

    def test_failed_stop_does_not_start_new_service(self):
        with mock.patch.object(self.switcher, 'command', return_value=False) as command:
            self.assertFalse(self.switcher.switch('brockone')[0])
        command.assert_called_once_with('stop', 'cosmos')

    def test_inference_and_concurrent_switch_are_excluded(self):
        self.switcher.generation_lock = mock.Mock()
        self.switcher.generation_lock.acquire.return_value = False
        with mock.patch.object(self.switcher, 'command') as command:
            self.assertFalse(self.switcher.switch('brockone')[0])
        command.assert_not_called()
        self.switcher.generation_lock.release.assert_not_called()
        self.switcher.lock.acquire()
        try:
            self.assertFalse(self.switcher.switch('brockone')[0])
        finally:
            self.switcher.lock.release()

    def test_unknown_or_active_choice_does_not_restart(self):
        with mock.patch.object(self.switcher, 'command') as command, \
             mock.patch.object(self.switcher, 'healthy', return_value=True):
            self.assertFalse(self.switcher.switch('unknown')[0])
            self.assertTrue(self.switcher.switch('cosmos')[0])
        command.assert_called_once_with('stop', 'brockone')

    def test_disabled_healthy_service_is_not_admitted_at_startup(self):
        self.entries['brockone']['enabled'] = False
        with mock.patch.object(engines.ServiceEngineSwitcher, 'healthy', side_effect=lambda key: key == 'brockone'):
            switcher = engines.ServiceEngineSwitcher(self.entries, 'cosmos', threading.Lock())
        self.assertEqual(switcher.selected, 'cosmos')

    def test_fixed_request_contract_and_no_fabricated_timings(self):
        original = payload()
        converted = engines.brockone_request(original)
        self.assertFalse(converted['stream'])
        self.assertTrue(original['stream'])
        for key, value in [('model', 'nvidia/Cosmos3-Edge'), ('max_tokens', 100),
                           ('temperature', 0.7), ('max_image_tokens_per_image', 320)]:
            bad = payload(); bad[key] = value
            with self.assertRaises(ValueError):
                engines.brockone_request(bad)
        answer = {'model': 'brockone', 'choices': [{'message': {'content': 'This is Pikachu.'}}],
                  'usage': {'completion_tokens': 5}}
        sse = engines.brockone_sse(answer).decode()
        self.assertIn('This is Pikachu.', sse)
        self.assertIn('"token_timing_available": false', sse)
        self.assertIn('data: [DONE]', sse)
        with self.assertRaises(ValueError):
            engines.brockone_sse({**answer, 'model': 'other'})
        for bad in [[], {'model': 'brockone', 'choices': ['bad']},
                    {'model': 'brockone', 'choices': [{'message': 'bad'}]}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                engines.brockone_sse(bad)


class LiveTrialTests(unittest.TestCase):
    """Explicitly synthetic authorization; no real readiness or systemd calls."""

    setUp = RegistryTests.setUp

    def trial(self, **changes):
        entry = self.entries['brockone']
        entry['validation_mode'] = 'live_trial'
        value = {'state': 'orin_engine_live_trial', 'model_id': 'brockone',
                 'engine_root': entry['path'], 'user_authorized_validation_bypass': True,
                 'validation_performed': False, 'validation_passed': False}
        value.update(changes)
        proof = Path(entry['readiness_receipt'])
        proof.write_text(json.dumps(value))
        entry['readiness_sha256'] = hashlib.sha256(proof.read_bytes()).hexdigest()
        return value

    def test_trial_is_explicit_and_visibly_unvalidated(self):
        proof = self.trial()
        self.assertTrue(self.switcher.availability('brockone')[0])
        self.assertNotIn('engine_load_passed', proof)
        self.assertNotIn('prepared_pixels_passed', proof)
        descriptor = self.switcher.descriptor('brockone')
        self.assertIn('unvalidated trial', descriptor['name'])
        self.assertIn('Unvalidated live trial', descriptor['profile'])
        self.assertEqual(descriptor['validation_status'], 'unvalidated_live_trial')
        self.assertEqual(descriptor['model_id'], 'brockone')
        self.assertEqual(descriptor['request_policy'], engines.BROCKONE_POLICY)
        self.assertEqual(self.switcher.selected, 'cosmos')

    def test_trial_receipt_cannot_pass_default_validated_mode(self):
        self.trial()
        self.entries['brockone'].pop('validation_mode')
        self.assertFalse(self.switcher.availability('brockone')[0])

    def test_validated_receipt_cannot_pass_trial_mode(self):
        self.entries['brockone']['validation_mode'] = 'live_trial'
        self.assertFalse(self.switcher.availability('brockone')[0])

    def test_exact_authorization_scope_is_required(self):
        for key, value in [('state', 'orin_engine_validated'), ('model_id', 'other'),
                           ('engine_root', '/wrong/engine'), ('user_authorized_validation_bypass', False),
                           ('user_authorized_validation_bypass', 1), ('validation_performed', True),
                           ('validation_passed', True)]:
            with self.subTest(key=key, value=value):
                self.trial(**{key: value})
                with mock.patch.object(self.switcher, 'command') as command:
                    self.assertFalse(self.switcher.switch('brockone')[0])
                command.assert_not_called()
                self.assertEqual(self.switcher.selected, 'cosmos')

    def test_trial_keeps_hash_and_disabled_engine_gates(self):
        self.trial()
        self.entries['brockone']['readiness_sha256'] = '0' * 64
        self.assertFalse(self.switcher.availability('brockone')[0])
        self.trial()
        self.entries['brockone']['enabled'] = False
        self.assertFalse(self.switcher.availability('brockone')[0])

    def test_trial_switches_both_ways_and_preserves_rollback(self):
        self.trial()
        with mock.patch.object(self.switcher, 'command', return_value=True) as command, \
             mock.patch.object(self.switcher, 'wait_ready', return_value=True):
            self.assertTrue(self.switcher.switch('brockone')[0])
            self.assertTrue(self.switcher.switch('cosmos')[0])
        self.assertEqual(command.call_args_list, [mock.call('stop', 'cosmos'), mock.call('start', 'brockone'),
                                                 mock.call('stop', 'brockone'), mock.call('start', 'cosmos')])
        self.assertEqual(self.switcher.selected, 'cosmos')
        with mock.patch.object(self.switcher, 'command', return_value=True) as command, \
             mock.patch.object(self.switcher, 'wait_ready', side_effect=[False, True]):
            ok, reason = self.switcher.switch('brockone')
        self.assertFalse(ok)
        self.assertIn('restored Cosmos3-Edge', reason)
        self.assertEqual(self.switcher.selected, 'cosmos')
        self.assertEqual(command.call_args_list, [mock.call('stop', 'cosmos'), mock.call('start', 'brockone'),
                                                 mock.call('stop', 'brockone'), mock.call('start', 'cosmos')])

    def test_unknown_mode_rejected(self):
        self.entries['brockone']['validation_mode'] = 'bypass_everything'
        with self.assertRaises(ValueError):
            engines.validate_registry(self.entries)


class FakeModel(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def answer(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
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
        self.answer(self.server.malformed if self.server.malformed is not None else
                    {'model': self.server.model_id, 'choices': [{'message': {'content': 'This is Pikachu.'}}]})


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.model = ThreadingHTTPServer(('127.0.0.1', 0), FakeModel)
        self.model.model_id = 'brockone'; self.model.received = []; self.model.malformed = None
        threading.Thread(target=self.model.serve_forever, daemon=True).start()
        self.switcher = mock.Mock(managed=True, switching=False)
        self.switcher.backend.return_value = {'protocol': 'brockone', 'model_id': 'brockone',
                                              'backend_port': self.model.server_port}
        self.switcher.active.return_value = {'id': 'brockone', 'request_policy': engines.BROCKONE_POLICY,
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

    def test_real_json_to_sse_proxy_and_stale_model_rejection(self):
        code, raw = self.request('POST', '/v1/chat/completions', json.dumps(payload()), {'Content-Type': 'application/json'})
        self.assertEqual(code, 200, raw)
        self.assertIn(b'This is Pikachu.', raw)
        self.assertEqual(self.model.received[0], engines.brockone_request(payload()))
        stale = payload(); stale['model'] = 'nvidia/Cosmos3-Edge'
        self.assertEqual(self.request('POST', '/v1/chat/completions', json.dumps(stale), {'Content-Type': 'application/json'})[0], 409)
        self.assertEqual(len(self.model.received), 1)

    def test_health_runtime_and_switch_unavailable(self):
        self.assertEqual(json.loads(self.request('GET', '/health/ready')[1])['status'], 'ready')
        runtime = json.loads(self.request('GET', '/api/runtime')[1])
        self.assertFalse(runtime['streaming'])
        self.assertNotIn('static_clocks', runtime)
        self.switcher.switching = True
        self.assertEqual(self.request('GET', '/v1/models')[0], 503)

    def test_unqualified_running_backend_is_not_served(self):
        self.switcher.active.return_value['available'] = False
        self.switcher.active.return_value['reason'] = 'Target engine validation is pending'
        for path in ['/v1/models', '/health/ready', '/api/runtime']:
            self.assertEqual(self.request('GET', path)[0], 503)
        self.assertEqual(self.request('POST', '/v1/chat/completions', json.dumps(payload()),
                                      {'Content-Type': 'application/json'})[0], 503)
        self.assertEqual(self.model.received, [])

    def test_malformed_success_becomes_bounded_502(self):
        for value in [[], {'model': 'brockone', 'choices': ['wrong']},
                      {'model': 'brockone', 'choices': [{'message': 'wrong'}]}]:
            self.model.malformed = value
            code, raw = self.request('POST', '/v1/chat/completions', json.dumps(payload()),
                                     {'Content-Type': 'application/json'})
            self.assertEqual(code, 502)
            self.assertIn('message', json.loads(raw)['error'])

    def test_switch_requires_same_origin_token(self):
        self.assertEqual(self.request('POST', '/api/engines/brockone')[0], 403)
        self.switcher.switch.assert_not_called()
        self.switcher.switch.return_value = (True, 'Switched')
        code, raw = self.request('POST', '/api/engines/brockone', headers={'X-Reachy-Token': ui.RELAY_TOKEN})
        self.assertEqual(code, 200, raw)
        self.switcher.switch.assert_called_once_with('brockone')

    def test_health_checks_actual_model_identity(self):
        instance = object.__new__(engines.ServiceEngineSwitcher)
        instance.engines = {'b': {'backend_port': self.model.server_port, 'model_id': 'brockone', 'protocol': 'brockone'}}
        self.assertTrue(instance.healthy('b'))
        self.model.model_id = 'wrong-model'
        self.assertFalse(instance.healthy('b'))


if __name__ == '__main__':
    unittest.main()
