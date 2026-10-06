"""Classifier entries, /api/classify and the sample sets against a fake classifier service; no model,
systemd or GPU work."""
import base64
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('classifier_test_ui', SCRIPTS / 'serve_ui.py')
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

JPEG = b'\xff\xd8\xff\xe0synthetic jpeg body'


def answer(model='vit', saliency=True):
    return {'model': model, 'species': 'pikachu', 'label': 'Pikachu', 'score': 0.9,
            'topk': [{'species': 'pikachu', 'label': 'Pikachu', 'score': 0.9},
                     {'species': None, 'label': 'MissingNo', 'score': 0.05}],
            'saliency': {'w': 2, 'h': 1, 'cells': [0.0, 1.0], 'method': 'attention rollout'} if saliency else None,
            'boxes': [], 'timing_ms': {'preprocess': 2.0, 'inference': 9.5, 'total': 12.0}}


class FakeClassifiers(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, value):
        body = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, {'models': [{'id': 'vit', 'installed': True, 'species': ['pikachu', 'eevee']},
                                    {'id': 'cnn', 'installed': False, 'species': ['pikachu']}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.calls.append((self.path, body))
        self.reply(*self.server.answers.get(self.path, (200, {'ok': True})))


class Fixture(unittest.TestCase):
    def setUp(self):
        self.fake = ThreadingHTTPServer(('127.0.0.1', 0), FakeClassifiers)
        self.fake.calls, self.fake.answers = [], {'/classify': (200, answer())}
        threading.Thread(target=self.fake.serve_forever, daemon=True).start()
        self.addCleanup(self.fake.server_close)
        self.addCleanup(self.fake.shutdown)
        self.url = f'http://127.0.0.1:{self.fake.server_port}'
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / 'v3').mkdir()
        (self.root / 'v3' / 'llm.engine').write_bytes(b'synthetic engine fixture')
        self.link = self.root / 'default'
        self.link.symlink_to(self.root / 'v3')
        self.entries = {'v3': {'name': 'Slow', 'path': str(self.root / 'v3')},
                        'vit': {'kind': 'classifier', 'name': 'ViT', 'model_id': 'vit', 'path': str(self.root)},
                        'cnn': {'kind': 'classifier', 'name': 'CNN', 'model_id': 'cnn', 'path': str(self.root)}}
        self.switcher = ui.EngineSwitcher(self.link, self.entries, 'shim', 1, self.url)
        self.samples_dir = self.root / 'samples'
        self.samples_dir.mkdir()
        (self.samples_dir / 'pikachu_01.jpg').write_bytes(JPEG)
        (self.samples_dir / 'absol_01.jpg').write_bytes(JPEG)
        (self.samples_dir / 'manifest.json').write_text(json.dumps({'title': 'Pokémon', 'subject': 'pokemon', 'images': [
            {'id': 'pikachu_01', 'file': 'pikachu_01.jpg', 'species': 'pikachu', 'license': 'CC BY 2.0', 'credit': 'Someone'},
            {'id': 'absol_01', 'file': 'absol_01.jpg', 'species': 'absol', 'license': 'CC0'}]}))
        # A spot set: where one target (here an Eevee) is in each picture, if it is there at all.
        self.spot_dir = self.root / 'park'
        self.spot_dir.mkdir()
        (self.spot_dir / 'p-1.jpg').write_bytes(JPEG)
        (self.spot_dir / 'p-2.jpg').write_bytes(JPEG)
        self.spot_manifest = {'title': 'Park', 'subject': 'pokemon', 'task': 'spot', 'target': 'eevee',
                              'target_name': 'our Eevee', 'images': [
            {'id': 'park-p-1', 'file': 'p-1.jpg', 'caption': '12:00', 'boxes': [[0.1, 0.2, 0.3, 0.4]],
             'candidates': [[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.9, 0.9]]},
            {'id': 'park-p-2', 'file': 'p-2.jpg', 'boxes': []}]}
        (self.spot_dir / 'manifest.json').write_text(json.dumps(self.spot_manifest))


class SwitchTests(Fixture):
    def test_classifier_selection_leaves_the_shim_alone(self):
        with mock.patch.object(self.switcher, '_run') as run, mock.patch.object(self.switcher, 'wait_ready', return_value=True):
            self.assertEqual(self.switcher.switch('vit'), (True, 'switched to ViT'))
            run.assert_not_called()
            self.assertEqual(self.switcher.active()['id'], 'vit')
            self.assertEqual(self.switcher.cosmos_active()['id'], 'v3')
            # Back to the engine the shim still holds: no restart, and the classifier is freed.
            self.assertEqual(self.switcher.switch('v3'), (True, 'switched to Slow'))
            run.assert_not_called()
        self.assertEqual([path for path, _ in self.fake.calls], ['/load', '/unload'])
        self.assertEqual(self.fake.calls[0][1], {'model': 'vit'})
        self.assertEqual(self.switcher.active()['id'], 'v3')

    def test_uninstalled_or_unreachable_classifier_is_refused(self):
        self.assertEqual(self.switcher.availability('cnn'), (False, 'Not installed in the classifier service'))
        self.assertFalse(self.switcher.switch('cnn')[0])
        offline = ui.EngineSwitcher(self.link, self.entries, 'shim', 1, 'http://127.0.0.1:9')
        self.assertEqual(offline.availability('vit'), (False, 'The classifier service is not running'))
        self.assertEqual(self.fake.calls, [])

    def test_failed_load_keeps_the_previous_model(self):
        self.fake.answers['/load'] = (500, {'error': 'CUDA error'})
        ok, message = self.switcher.switch('vit')
        self.assertFalse(ok)
        self.assertIn('CUDA error', message)
        self.assertEqual(self.switcher.active()['id'], 'v3')


class ValidationTests(Fixture):
    def test_requests_carry_one_bounded_image_or_a_known_sample(self):
        samples = ui.load_samples(self.samples_dir)
        image = 'data:image/jpeg;base64,' + base64.b64encode(JPEG).decode()
        sent = ui.classify_request({'model': 'vit', 'image': image, 'saliency': True}, samples)
        self.assertEqual((sent['image'], sent['saliency'], sent['topk']), (base64.b64encode(JPEG).decode(), True, 5))
        self.assertEqual(ui.classify_request({'model': 'vit', 'sample': 'absol_01'}, samples)['image'],
                         base64.b64encode(JPEG).decode())
        for bad in [{'model': 'vit'}, {'model': 'vit', 'image': image, 'sample': 'absol_01'},
                    {'model': 'vit', 'sample': '../manifest'}, {'model': 'vit', 'image': 'https://x/y.jpg'},
                    {'model': 'vit', 'image': 'data:image/jpeg;base64,' + base64.b64encode(b'GIF89a').decode()},
                    {'model': 'vit', 'image': image, 'topk': 11}, {'model': 'vit', 'image': image, 'path': '/etc'}]:
            with self.subTest(bad=str(bad)[:60]), self.assertRaises(ValueError):
                ui.classify_request(bad, samples)

    def test_answers_are_bounded_numbers(self):
        self.assertEqual(ui.classifier_result(answer())['saliency']['cells'], [0.0, 1.0])
        for key, value in [('topk', []), ('topk', [{'species': 'x', 'score': 2}]),
                           ('saliency', {'w': 2, 'h': 1, 'cells': [0.0]}),
                           ('saliency', {'w': 2, 'h': 1, 'cells': [0.0, float('nan')]}),
                           ('boxes', [[0, 0, 1]])]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ui.classifier_result({**answer(), key: value})

    def test_manifest_names_only_files_beside_it(self):
        for item in [{'id': 'a', 'file': '../outside.jpg', 'species': 'x'},
                     {'id': 'a', 'file': 'missing.jpg', 'species': 'x'},
                     {'id': 'A b', 'file': 'pikachu_01.jpg', 'species': 'x'},
                     {'id': 'a', 'file': 'manifest.json', 'species': 'x'}]:
            (self.samples_dir / 'manifest.json').write_text(json.dumps({'images': [item]}))
            with self.subTest(item=item), self.assertRaises(ValueError):
                ui.load_samples(self.samples_dir)

    def test_a_spot_set_carries_its_target_and_boxes(self):
        park = ui.load_sample_set(self.spot_dir)
        self.assertEqual((park['id'], park['title'], park['task'], park['target'], park['target_name']),
                         ('park', 'Park', 'spot', 'eevee', 'our Eevee'))
        first, second = park['images']['park-p-1'], park['images']['park-p-2']
        self.assertEqual((first['caption'], first['boxes'], len(first['candidates']), first['species']),
                         ('12:00', [[0.1, 0.2, 0.3, 0.4]], 2, None))
        self.assertEqual((second['caption'], second['boxes'], second['candidates']), ('park-p-2', [], []))
        pokemon = ui.load_sample_set(self.samples_dir)
        self.assertEqual((pokemon['id'], pokemon['task'], pokemon['target'], pokemon['subject']),
                         ('samples', 'species', None, 'pokemon'))

    def test_spot_manifests_are_checked(self):
        image = self.spot_manifest['images'][0]
        for change in [{'target': None}, {'task': 'count'}, {'id': 'Bad id'}, {'title': ''}, {'target_name': 'x' * 41},
                       {'images': [{**image, 'boxes': [[0.3, 0.2, 0.1, 0.4]]}]},
                       {'images': [{**image, 'boxes': [[0.1, 0.2, 0.3, 1.4]]}]},
                       {'images': [{**image, 'boxes': [[0.1, 0.2, 0.3]]}]},
                       {'images': [{**image, 'boxes': None}]},
                       {'images': [{**image, 'candidates': [[0.1, 0.2, 0.3, 0.4]] * 65}]},
                       {'images': [{**image, 'species': 'eevee'}]}]:
            manifest = {**self.spot_manifest, **change}
            if manifest.get('target') is None:
                manifest.pop('target')
            (self.spot_dir / 'manifest.json').write_text(json.dumps(manifest))
            with self.subTest(change=str(change)[:70]), self.assertRaises(ValueError):
                ui.load_sample_set(self.spot_dir)
        # A species set has no target.
        (self.samples_dir / 'manifest.json').write_text(json.dumps({'target': 'pikachu', 'images': [
            {'id': 'a', 'file': 'pikachu_01.jpg', 'species': 'pikachu'}]}))
        with self.assertRaises(ValueError):
            ui.load_sample_set(self.samples_dir)

    def test_a_set_without_an_id_is_named_after_its_directory(self):
        manifest = {k: v for k, v in self.spot_manifest.items() if k not in ('id', 'title', 'target_name')}
        for name, expected in [('park', 'park'), ('Pokemon', 'pokemon'), ('samples.v2', 'samples-v2'),
                               ('_My Set', 'my-set'), ('x' * 40, 'x' * 32)]:
            directory = self.root / 'named' / name
            directory.mkdir(parents=True)
            (directory / 'p-1.jpg').write_bytes(JPEG)
            (directory / 'p-2.jpg').write_bytes(JPEG)
            (directory / 'manifest.json').write_text(json.dumps({**manifest, 'title': None, 'target_name': None}))
            with self.subTest(name=name):
                loaded = ui.load_sample_set(directory)
                # An explicit null is the key left out: the defaults.
                self.assertEqual((loaded['id'], loaded['title'], loaded['target_name']), (expected, expected, 'the target'))

    def test_sets_need_their_own_ids_and_image_ids(self):
        sets, samples = ui.load_sample_sets([self.samples_dir, self.spot_dir])
        self.assertEqual(([s['id'] for s in sets], sorted(samples)),
                         (['samples', 'park'], ['absol_01', 'park-p-1', 'park-p-2', 'pikachu_01']))
        with self.assertRaises(ValueError):
            ui.load_sample_sets([self.samples_dir, self.samples_dir])
        (self.spot_dir / 'manifest.json').write_text(json.dumps({**self.spot_manifest, 'images': [
            {**self.spot_manifest['images'][0], 'id': 'absol_01'}]}))
        with self.assertRaises(ValueError):
            ui.load_sample_sets([self.samples_dir, self.spot_dir])


class HTTPTests(Fixture):
    def setUp(self):
        super().setUp()

        class Quiet(ui.Handler):
            def log_message(self, *args): pass

        class Telemetry:
            def snapshot(self): return {}
            def close(self): pass
        self.server = ui.Server(('127.0.0.1', 0), Quiet, telemetry=Telemetry())
        self.server.engine_switcher = self.switcher
        self.server.sample_sets, self.server.samples = ui.load_sample_sets([self.samples_dir, self.spot_dir])
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        headers = {**({'Content-Type': 'application/json'} if body is not None else {}), **(headers or {})}
        connection.request(method, path, json.dumps(body) if body is not None else None, headers)
        response = connection.getresponse()
        result = response.status, response.getheader('Content-Type'), response.read()
        connection.close()
        return result

    def test_classify_relays_only_for_the_selected_classifier(self):
        request = {'model': 'vit', 'sample': 'pikachu_01', 'saliency': True, 'topk': 2}
        self.assertEqual(self.request('POST', '/api/classify', request)[0], 409)
        self.assertEqual(self.switcher.switch('vit')[0], True)
        code, _, raw = self.request('POST', '/api/classify', request)
        self.assertEqual(code, 200, raw)
        result = json.loads(raw)
        self.assertEqual((result['species'], result['saliency']['w'], result['boxes']), ('pikachu', 2, []))
        self.assertEqual(self.fake.calls[-1], ('/classify', {'model': 'vit', 'image': base64.b64encode(JPEG).decode(),
                                                            'saliency': True, 'topk': 2}))
        self.assertEqual(self.request('POST', '/api/classify', {**request, 'model': 'cnn'})[0], 409)
        self.fake.answers['/classify'] = (200, {**answer(), 'saliency': {'w': 3, 'h': 3, 'cells': []}})
        self.assertEqual(self.request('POST', '/api/classify', request)[0], 502)

    def test_samples_list_coverage_and_serve_only_manifest_files(self):
        code, _, raw = self.request('GET', '/api/samples?model=vit')
        listed = {item['id']: item for item in json.loads(raw)['sets'][0]['images']}
        self.assertEqual((code, listed['pikachu_01']['covered'], listed['absol_01']['covered']), (200, True, False))
        self.assertNotIn('covered', json.loads(self.request('GET', '/api/samples')[2])['sets'][0]['images'][0])
        code, content_type, body = self.request('GET', '/api/samples/pikachu_01')
        self.assertEqual((code, content_type, body), (200, 'image/jpeg', JPEG))
        for path in ['/api/samples/manifest', '/api/samples/..%2Fmanifest.json', '/api/samples/unknown']:
            self.assertEqual(self.request('GET', path)[0], 404, path)

    def test_spot_sets_are_listed_with_their_boxes_and_covered_by_target(self):
        data = json.loads(self.request('GET', '/api/samples?model=vit')[2])
        self.assertEqual([(s['id'], s['task']) for s in data['sets']], [('samples', 'species'), ('park', 'spot')])
        park = data['sets'][1]
        self.assertEqual((park['title'], park['target'], park['target_name'], park['subject']),
                         ('Park', 'eevee', 'our Eevee', 'pokemon'))
        first = park['images'][0]
        self.assertEqual((first['id'], first['species'], first['caption'], first['boxes'], first['covered']),
                         ('park-p-1', None, '12:00', [[0.1, 0.2, 0.3, 0.4]], True))
        self.assertEqual(len(first['candidates']), 2)
        self.assertNotIn('path', first)
        # Its images are served and classified by id, like any other sample.
        self.assertEqual(self.request('GET', '/api/samples/park-p-2')[:2], (200, 'image/jpeg'))
        self.assertEqual(self.switcher.switch('vit')[0], True)
        self.assertEqual(self.request('POST', '/api/classify', {'model': 'vit', 'sample': 'park-p-1'})[0], 200)

    def test_a_rebound_host_name_reads_no_sample(self):
        # DNS rebinding: the attacker's own name, now resolving to this server. Same-origin to the
        # browser, so only Host gives it away - and a spot set can be private frames.
        self.assertEqual(self.switcher.switch('vit')[0], True)
        for host in ('rebind.attacker.example', f'rebind.attacker.example:{self.server.server_port}'):
            for method, path, body in [('GET', '/api/samples', None), ('GET', '/api/samples/park-p-1', None),
                                       ('POST', '/api/classify', {'model': 'vit', 'sample': 'park-p-1'})]:
                code, _, raw = self.request(method, path, body, {'Host': host})
                self.assertEqual(code, 421, (host, path))
                self.assertNotIn(JPEG, raw)
        # The server's own address still works.
        self.assertEqual(self.request('GET', '/api/samples/park-p-1', None, {'Host': f'127.0.0.1:{self.server.server_port}'})[0], 200)

    def test_engine_list_marks_classifiers(self):
        data = json.loads(self.request('GET', '/api/engines')[2])
        engines = {engine['id']: engine for engine in data['engines']}
        self.assertEqual((engines['vit']['kind'], engines['vit']['available']), ('classifier', True))
        self.assertEqual(engines['cnn']['reason'], 'Not installed in the classifier service')
        self.assertNotIn('kind', engines['v3'])


if __name__ == '__main__':
    unittest.main()
