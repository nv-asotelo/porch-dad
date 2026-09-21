#!/usr/bin/env python3
"""One bounded, serial TTFT run; score server logs and retain loopback diagnostics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import urllib.request

from benchmark import build_payload, percentile, stream_request

PROMPT = 'Describe the shape and color on the left and on the right. Which shape is left of the other?'
IMAGE_SHA = '52074a486ce36764d48e86fec209694ade16137318545524a3b51e47a048ca55'


def distribution(values):
    assert values and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values)
    return dict(count=len(values), mean=statistics.mean(values), p50=percentile(values, .5),
                p95=percentile(values, .95), minimum=min(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend-id', required=True)
    parser.add_argument('--image', type=Path, default=Path('benchmarks/fixtures/jpeg/01-left-right.jpg'))
    parser.add_argument('--request-log', type=Path, default=Path('data/logs/native-requests.jsonl'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    endpoint = 'http://127.0.0.1:8090'
    source = args.image.read_bytes()
    assert hashlib.sha256(source).hexdigest() == IMAGE_SHA
    runtime = json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/runtime', timeout=10))
    assert runtime['encoder_cache_bytes'] == 0 and runtime['static_clocks'] is False
    assert runtime['max_image_tokens_per_image'] == 512 and runtime['top_p'] == 1
    workload = dict(model='Cosmos3-Edge', max_tokens=64, temperature=0, top_p=1,
                    prompt=PROMPT, image_mime_type='image/jpeg')
    payload = build_payload(workload, source)
    payload['max_image_tokens_per_image'] = 512
    started = datetime.now(timezone.utc).isoformat()
    plan = dict(started_utc=started, backend_id=args.backend_id, runtime=runtime, workload=workload,
                endpoint=endpoint, image_path=str(args.image), image_sha256=IMAGE_SHA,
                image_bytes=len(source), warmups=5, measured=30, concurrency=1,
                primary_boundary='native_start_to_server_text',
                primary_excludes=['JPEG encoding', 'JPEG decoding', 'request preparation', 'admission queue', 'network transport'],
                primary_includes=['native image processing', 'vision encoder', 'prefill', 'generation to first text', 'server consumer scheduling'],
                diagnostic_boundary='loopback HTTP request start to first nonempty SSE text; includes proxy/JPEG decode',
                stop_rule='Exactly five warmups plus thirty measured requests. Stop on failure; no optimization loop.')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    records = []
    with (args.output / 'responses.jsonl').open('x') as stream:
        for index in range(35):
            row = stream_request(endpoint + '/v1/chat/completions', payload, 120)
            row.update(index=index, warmup=index < 5)
            stream.write(json.dumps(row) + '\n'); stream.flush()
            assert row['error'] is None, row
            metrics = row['cosmos_metrics']
            assert metrics['first_text_timing_boundary'] == 'native_start_to_server_text', metrics
            assert metrics['timing_source'] == 'server_monotonic' and metrics['timing_boundary'] == 'native_inference'
            assert metrics['server_first_text_ms'] > 0 and metrics['native_inference_ms'] > 0
            assert metrics['cache_state'] == 'disabled'
            assert metrics['completion_tokens'] == row['completion_tokens'] > 0
            records.append(row)
            if (index + 1) % 5 == 0: print(f'{args.backend_id}: {index + 1}/35 requests complete', flush=True)
    ended = datetime.now(timezone.utc).isoformat()
    ids = {r['cosmos_metrics']['request_id'] for r in records}
    all_native = [json.loads(line) for line in args.request_log.read_text().splitlines() if line.strip()]
    native = [r for r in all_native if r.get('request_id') in ids]
    assert len(native) == len(ids) == 35
    by_id = {r['request_id']: r for r in native}
    controls = None
    for row in records:
        metrics = row['cosmos_metrics']; record = by_id[metrics['request_id']]
        assert record['backend_id'] == args.backend_id and record['engine_id'] == runtime['engine_id']
        assert record['status'] == 'completed' and record['cache_state'] == 'disabled'
        assert record['server_first_text_ms'] == metrics['server_first_text_ms']
        assert record['native_inference_ms'] == metrics['native_inference_ms']
        assert record['observed_image_tokens'] > 0
        if controls is None: controls = record['controls']
        assert record['controls'] == controls
    # Native rows deliberately remain untouched. Warmup membership lives in the
    # response records because the LAN proxy rejects benchmark-only annotations.
    (args.output / 'server-requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in native))
    other = [r for r in all_native if r.get('engine_id') == runtime['engine_id']
             and started <= r.get('at_utc', '') <= ended and r.get('request_id') not in ids]
    (args.output / 'other-requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in other))
    samples = [r for r in records if not r['warmup']]
    receipt = dict(status='passed', started_utc=started, ended_utc=ended, backend_id=args.backend_id,
                   runtime=runtime, warmups=5, measured=30, request_ids_matched=True,
                   controls=controls,
                   other_logged_requests_started_during_run=len(other),
                   other_completed_requests_during_run=sum(r.get('status') == 'completed' for r in other),
                   other_request_detection_scope='Logged request-start timestamps inside the run; not all possible overlapping activity.',
                   other_request_ids=[r['request_id'] for r in other],
                   server_ttft_ms=distribution([r['cosmos_metrics']['server_first_text_ms'] for r in samples]),
                   loopback_http_ttft_ms=distribution([r['ttft_ms'] for r in samples]),
                   native_total_ms=distribution([r['cosmos_metrics']['native_inference_ms'] for r in samples]),
                   actual_completion_tokens=sorted({r['completion_tokens'] for r in samples}),
                   observed_image_tokens=sorted({by_id[r['cosmos_metrics']['request_id']]['observed_image_tokens'] for r in samples}),
                   distinct_answers=len({r['output_text'] for r in samples}))
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
