#!/usr/bin/env python3
"""Accept the deployed native controls against a fresh, idle loopback backend.

Run on the Orin before opening the UI or running other clients. This is a
functional acceptance run, not a latency benchmark or a semantic quality score.
The output is created exclusively and includes a failure receipt if a check fails.
No browser/network timing is used as inference latency.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid

from benchmark import iter_sse

ROOT = Path(__file__).resolve().parents[1]
PROMPT = 'Describe what you see in this image in one sentence.'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def read_log(path):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            require(isinstance(row, dict), f'Invalid request log row {number}')
            if row.get('kind') == 'server_request':
                rows.append(row)
    return rows


def json_http(base, path, payload=None, timeout=120):
    request = Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'},
                      method='GET' if payload is None else 'POST')
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.loads(error.read())


def validate_metrics(metrics, usage, row, budget, expected_cache=None, expected_top_p=0.95):
    require(isinstance(metrics, dict) and isinstance(usage, dict), 'Missing native metrics or backend usage')
    require(metrics.get('timing_source') == 'server_monotonic'
            and metrics.get('timing_boundary') == 'native_inference', 'Wrong timing boundary')
    require(positive(metrics.get('native_inference_ms')), 'Native duration must be finite and positive')
    require(metrics['native_inference_ms'] == metrics.get('server_elapsed_ms'), 'Native duration disagreement')
    tokens = metrics.get('completion_tokens')
    require(type(tokens) is int and tokens > 0 and tokens == usage.get('completion_tokens'),
            'Native completion count differs from backend usage')
    require(metrics.get('prompt_tokens') == usage.get('prompt_tokens'), 'Prompt token count disagreement')
    require(row.get('status') == 'completed' and row.get('completion_tokens') == tokens,
            'Request log does not confirm successful native completion')
    require(row.get('server_elapsed_ms') == metrics['server_elapsed_ms'], 'Response and log timings differ')
    require(row.get('controls', {}).get('max_image_tokens_per_image') == budget, 'Requested image budget not applied')
    require(row.get('controls', {}).get('top_p') == expected_top_p, 'Requested top_p not applied')
    require(row.get('cache_state') == metrics.get('cache_state'), 'Response and log cache states differ')
    if expected_cache is not None:
        require(metrics.get('cache_state') == expected_cache,
                f"Expected {expected_cache} cache state, got {metrics.get('cache_state')}")
    if metrics.get('cache_state') in ('miss', 'disabled'):
        observed = metrics.get('observed_image_tokens')
        require(type(observed) is int and 0 < observed <= budget,
                'Uncached inference did not report visual work within its image budget')
    elif metrics.get('cache_state') == 'hit':
        require(metrics.get('observed_image_tokens') is None,
                'Encoder cache hit unexpectedly reported fresh vision work')


def run(args, receipt):
    parsed = urlsplit(args.base_url)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or '').is_loopback
    except ValueError:
        loopback = parsed.hostname == 'localhost'
    require(parsed.scheme == 'http' and loopback and parsed.path in ('', '/')
            and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
            'Acceptance must run against an HTTP loopback backend')
    base = args.base_url.rstrip('/')
    status, runtime = json_http(base, '/api/runtime')
    require(status == 200, f'Runtime settings returned HTTP {status}')
    receipt['runtime'] = runtime
    lightweight = args.policy == 'lightweight'
    default_budget, top_p = (512, 1.0) if lightweight else (320, 0.95)
    cache_bytes, clocks = (0, False) if lightweight else (268435456, True)
    prompt = ('Describe the visible scene in one concise sentence. Focus on objects and actions.'
              if lightweight else PROMPT)
    for key, expected in {'max_image_tokens_per_image': default_budget, 'max_image_tokens_per_image_limit': 512,
                          'max_image_tokens_per_image_min': 4, 'top_p': top_p,
                          'encoder_cache_bytes': cache_bytes, 'static_clocks': clocks,
                          'timing_boundary': 'native_inference', 'timing_source': 'server_monotonic'}.items():
        require(runtime.get(key) == expected, f'Unexpected engine default {key}: {runtime.get(key)!r}')
    require(isinstance(runtime.get('engine_id'), str) and runtime['engine_id'], 'Missing engine identity')
    status, health = json_http(base, '/health/ready')
    require(status == 200 and health.get('active_requests') == 0 and health.get('queued_requests') == 0,
            'Backend must be ready and idle before acceptance')
    status, models = json_http(base, '/v1/models')
    require(status == 200 and len(models.get('data', [])) == 1, 'Expected exactly one served model')
    model = models['data'][0]['id']
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    receipt['fixture_manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    require(len(manifest['fixtures']) == 3, 'Expected the three Live VLM fixtures')
    fixtures = {}
    for fixture in manifest['fixtures']:
        path = args.manifest.parent / fixture['file']
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == fixture['sha256'], 'Fixture hash mismatch: ' + fixture['file'])
        fixtures[fixture['file']] = {'url': 'data:image/jpeg;base64,' + base64.b64encode(data).decode(), **fixture}
    scissors = fixtures['03-scissors.jpg']
    run_id = receipt['run_id']

    def payload(fixture, budget=320, cache_mode='default', stream=False, omit_defaults=False):
        body = {'model': model, 'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': fixture['url']}}]}],
                'max_tokens': 64 if lightweight else 512, 'temperature': 0 if lightweight else 0.7, 'stream': stream,
                'cosmos_benchmark': {'run_id': run_id, 'warmup': True, 'cache_mode': cache_mode}}
        if not omit_defaults:
            body.update(max_image_tokens_per_image=budget, top_p=top_p)
        if stream:
            body['stream_options'] = {'include_usage': True}
        return body

    def logged(request_id):
        rows = [row for row in read_log(args.request_log) if row.get('request_id') == request_id]
        require(len(rows) == 1, f'Expected exactly one native log for request {request_id}')
        row = rows[0]
        require(row.get('engine_id') == runtime['engine_id'] and row.get('run_id') == run_id,
                'Native record belongs to a different engine or run')
        return row

    def infer(label, fixture, budget=320, cache_mode='default', expected_cache=None, omit_defaults=False):
        status, response = json_http(base, '/v1/chat/completions',
                                      payload(fixture, budget, cache_mode, omit_defaults=omit_defaults))
        require(status == 200, f'{label}: HTTP {status}: {response}')
        metrics, usage = response.get('cosmos_metrics'), response.get('usage')
        require(isinstance(metrics, dict) and metrics.get('request_id'), f'{label}: missing native request identity')
        row = logged(metrics['request_id'])
        validate_metrics(metrics, usage, row, budget, expected_cache, top_p)
        text = response['choices'][0]['message'].get('content', '')
        require(isinstance(text, str) and text.strip(), f'{label}: empty caption')
        record = {'label': label, 'fixture': fixture['file'], 'fixture_sha256': fixture['sha256'],
                  'image_budget': budget, 'cache_mode': cache_mode, 'text': text,
                  'usage': usage, 'metrics': metrics, 'native_log': row}
        receipt['requests'].append(record)
        print(f"PASS {label}: {metrics['cache_state']}, {usage['completion_tokens']} output tokens", flush=True)
        return record

    schedule = [(320, 'default', 'miss'), (320, 'default', 'hit'),
                (512, 'default', 'miss'), (512, 'default', 'hit'),
                (320, 'default', 'hit'), (320, 'bypass', 'disabled'),
                (320, 'bypass', 'disabled'), (320, 'default', 'hit')]
    if not cache_bytes:
        schedule = [(budget, mode, 'disabled') for budget, mode, _ in schedule]
    references = {}
    for index, (budget, cache_mode, expected) in enumerate(schedule, 1):
        record = infer(f'cache-sequence-{index}', scissors, budget, cache_mode, expected)
        signature = (record['text'], record['usage']['completion_tokens'])
        if budget in references:
            require(signature == references[budget], f'Caption/token count changed across cache modes at budget {budget}')
        else:
            references[budget] = signature
    receipt['cache_sequence_verified'] = True
    infer('custom-image-budget-384', scissors, 384, expected_cache='miss' if cache_bytes else 'disabled')
    default_record = infer('omitted-image-budget-and-top-p', scissors, budget=default_budget, omit_defaults=True,
                           expected_cache='hit' if cache_bytes else 'disabled')
    receipt['omitted_defaults_verified'] = default_record['native_log']['controls']

    before_invalid = read_log(args.request_log)
    for invalid in (3, 513, True, 320.5):
        status, response = json_http(base, '/v1/chat/completions', payload(scissors, budget=invalid))
        require(status == 400, f'Invalid cap {invalid!r} returned HTTP {status}, expected 400')
        receipt['invalid_inputs'].append({'value': invalid, 'status': status, 'response': response})
    after_invalid = read_log(args.request_log)
    require(before_invalid == after_invalid, 'Invalid image budgets caused native work or concurrent traffic was present')
    receipt['invalid_inputs_started_no_native_requests'] = True

    # Capture captions for human review; no automated semantic pass/fail is inferred.
    for fixture in fixtures.values():
        for budget in (320, 512):
            record = infer(f"caption-review-{fixture['file']}-{budget}", fixture, budget)
            receipt['caption_review'].append({'fixture': fixture['file'], 'image_budget': budget,
                                              'reference_facts': fixture.get('reference_facts', []),
                                              'text': record['text'], 'request_id': record['metrics']['request_id']})

    request = Request(base + '/v1/chat/completions', data=json.dumps(payload(scissors, stream=True)).encode(),
                      headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'}, method='POST')
    metrics = usage = finish = None
    output, done = [], False
    with urlopen(request, timeout=120) as response:
        require('text/event-stream' in response.headers.get('Content-Type', ''), 'Streaming response has wrong type')
        for event in iter_sse(response):
            if event.strip() == '[DONE]':
                done = True
                break
            chunk = json.loads(event)
            require(not chunk.get('error'), f'Stream error: {chunk.get("error")}')
            if chunk.get('cosmos_metrics') is not None:
                require(metrics is None, 'Duplicate native metrics event')
                metrics = chunk['cosmos_metrics']
            if chunk.get('usage') is not None:
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                content = choice.get('delta', {}).get('content')
                if content:
                    output.append(content)
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
    require(done and finish in ('stop', 'length') and output and isinstance(metrics, dict),
            'Stream lacks complete text, finish event, native metrics or [DONE]')
    row = logged(metrics['request_id'])
    validate_metrics(metrics, usage, row, 320, 'hit' if cache_bytes else 'disabled', top_p)
    require((''.join(output), usage['completion_tokens']) == references[320], 'Streaming caption differs from nonstreaming')
    receipt['streaming'] = {'text': ''.join(output), 'usage': usage, 'metrics': metrics,
                            'finish_reason': finish, 'done': done, 'native_log': row}
    status, final_health = json_http(base, '/health/ready')
    require(status == 200 and final_health.get('active_requests') == 0 and final_health.get('queued_requests') == 0,
            'Backend did not return to idle after acceptance')
    receipt['final_health'] = final_health
    rows = [row for row in read_log(args.request_log) if row.get('run_id') == run_id]
    require(len(rows) == len(receipt['requests']) + 1, 'Acceptance request/log count mismatch')
    receipt['native_request_count'] = len(rows)
    receipt['passed'] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=('lightweight', 'experimental320'), default='lightweight',
                        help='Expected engine-load policy; experimental320 reproduces the superseded intermediate settings')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'benchmarks/live-vlm-1280/manifest.json')
    parser.add_argument('--request-log', type=Path, default=ROOT / 'data/logs/native-requests.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/runtime-controls/validation.json')
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'schema_version': 1, 'kind': 'runtime_controls_acceptance', 'passed': False,
               'policy': args.policy, 'created_utc': datetime.now(timezone.utc).isoformat(), 'run_id': 'acceptance-' + uuid.uuid4().hex,
               'requests': [], 'invalid_inputs': [], 'caption_review': [],
               'scope': 'Functional acceptance only. Captions require human review; these are not performance or quality scores.',
               'cancellation_check': 'Not performed by this bounded acceptance script.'}
    with args.output.open('x') as output:
        try:
            run(args, receipt)
            code = 0
        except Exception as error:
            receipt['error'] = f'{type(error).__name__}: {error}'
            print(receipt['error'], file=sys.stderr)
            code = 1
        receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'passed': receipt['passed'], 'receipt': str(args.output)}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
