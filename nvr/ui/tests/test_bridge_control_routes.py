"""Shared bridge controls are guarded and cannot select an arbitrary service or robot."""
import http.client
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location('bridge_route_ui', Path(__file__).resolve().parents[1] / 'scripts/serve_ui.py')
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

class Handler(ui.Handler):
    def log_message(self, *args):
        pass

class BridgeRoutesTest(unittest.TestCase):
    def setUp(self):
        self.server = ui.Server(('127.0.0.1', 0), Handler, telemetry=Mock())
        self.server.bridge_control = Mock()
        self.server.bridge_control.status.return_value = {'state': 'released', 'released': True}
        self.server.bridge_control.change.return_value = {'ok': True, 'state': 'released'}
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method='POST', path='/api/reachy/bridge/release', body=None, headers=None):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            res = conn.getresponse()
            return res.status, res.read()
        finally:
            conn.close()

    def token(self, **extra):
        return {'X-Reachy-Token': ui.RELAY_TOKEN, **extra}

    def test_no_token_cannot_change_bridge(self):
        self.assertEqual(self.request()[0], 403)
        self.server.bridge_control.change.assert_not_called()

    def test_cross_origin_cannot_change_bridge_even_with_token(self):
        self.assertEqual(self.request(headers=self.token(Origin='https://untrusted.example'))[0], 403)
        self.server.bridge_control.change.assert_not_called()

    def test_cross_site_cannot_change_bridge(self):
        self.assertEqual(self.request(headers=self.token(**{'Sec-Fetch-Site': 'cross-site'}))[0], 403)
        self.server.bridge_control.change.assert_not_called()

    def test_rebound_host_cannot_change_bridge(self):
        self.assertEqual(self.request(headers=self.token(Host='untrusted.example'))[0], 421)
        self.server.bridge_control.change.assert_not_called()

    def test_only_fixed_actions(self):
        self.assertEqual(self.request(path='/api/reachy/bridge/restart-other-service', headers=self.token())[0], 404)
        self.server.bridge_control.change.assert_not_called()

    def test_cannot_supply_address_or_unit(self):
        self.assertEqual(self.request(body=json.dumps({'unit':'other.service'}), headers=self.token())[0], 400)
        self.server.bridge_control.change.assert_not_called()

    def test_valid_release_and_resume(self):
        for action in ('release', 'resume'):
            self.assertEqual(self.request(path=f'/api/reachy/bridge/{action}', headers=self.token())[0], 200)
            self.server.bridge_control.change.assert_called_with(action)

    def test_status_has_no_mutation(self):
        code, raw = self.request(method='GET', path='/api/reachy/bridge', headers=self.token())
        self.assertEqual(code, 200)
        self.assertTrue(json.loads(raw)['released'])
        self.server.bridge_control.change.assert_not_called()

    def test_failed_stop_is_not_success(self):
        self.server.bridge_control.change.return_value = {'ok':False, 'message':'Still running'}
        self.assertEqual(self.request(headers=self.token())[0], 503)

if __name__ == '__main__':
    unittest.main()
