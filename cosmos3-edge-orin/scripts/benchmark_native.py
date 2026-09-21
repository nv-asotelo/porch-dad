#!/usr/bin/env python3
"""Run a bounded fixed-input token-length experiment against the local engine.

Scores are extracted from the backend's native request log, not this client's
clock. Encoder-cache bypass forces vision work without changing input pixels.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--request-log', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000')
    parser.add_argument('--repetitions', type=int, default=8)
    args = parser.parse_args()
    if not 3 <= args.repetitions <= 30:
        parser.error('Use 3–30 repetitions per output-cap/configuration cell.')
    if args.output.exists():
        parser.error('Output directory must be new; previous experiments are preserved.')
    args.output.mkdir(parents=True)
    run_id = 'native-' + uuid.uuid4().hex
    source = args.image.read_bytes()
    image = 'data:image/jpeg;base64,' + base64.b64encode(source).decode()
    prompt = 'Provide a detailed description of the scene for a visually impaired person.'
    rng = random.Random(20260921)
    conditions = [(budget, mode, cap) for _ in range(args.repetitions)
                  for budget in (320, 512) for mode in ('default', 'bypass') for cap in (8, 16, 32, 64)]
    rng.shuffle(conditions)
    runtime = json.load(urllib.request.urlopen(args.endpoint + '/api/runtime', timeout=5))
    plan = dict(run_id=run_id, created_utc=datetime.now(timezone.utc).isoformat(),
                runtime=runtime, image_sha256=hashlib.sha256(source).hexdigest(), prompt=prompt,
                temperature=0.7, top_p=0.95, output_caps=[8, 16, 32, 64],
                image_budgets=[320, 512], repetitions_per_cell=args.repetitions,
                warmups_per_budget_and_cache_mode=3, randomization_seed=20260921,
                measured_requests=len(conditions), timing_boundary='native_inference',
                cache_modes={'default': 'Repeated image; native cache hit must be observed.',
                             'bypass': 'Native encoder runs each time; lookup/storage bypassed without changing pixels.'},
                stop_rule='One fixed experiment. Stop on any failed request; no optimization search.',
                comparison='Fit fixed_ms + marginal_ms_per_token * actual_completion_tokens separately by image budget and cache mode; do not score client durations.')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    records = []

    def request(budget, mode, cap, warmup):
        body = dict(model='Cosmos3-Edge', stream=False, temperature=.7, top_p=.95,
                    max_tokens=cap, max_image_tokens_per_image=budget,
                    cosmos_benchmark=dict(run_id=run_id, warmup=warmup, cache_mode=mode),
                    messages=[dict(role='user', content=[dict(type='text', text=prompt),
                             dict(type='image_url', image_url=dict(url=image))])])
        call = urllib.request.Request(args.endpoint + '/v1/chat/completions',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        result = json.load(urllib.request.urlopen(call, timeout=120))
        metrics = result['cosmos_metrics']
        usage = result['usage']
        assert metrics['timing_boundary'] == 'native_inference'
        assert metrics['completion_tokens'] == usage['completion_tokens'] > 0
        assert metrics['native_inference_ms'] > 0
        if not warmup:
            assert metrics['cache_state'] == ('hit' if mode == 'default' else 'disabled'), metrics
        row = dict(request_id=metrics['request_id'], budget=budget, cache_mode=mode,
                   max_tokens=cap, warmup=warmup, usage=usage, metrics=metrics,
                   finish_reason=result['choices'][0]['finish_reason'],
                   text=result['choices'][0]['message']['content'])
        records.append(row)
        with (args.output / 'responses.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    for budget in (320, 512):
        for mode in ('default', 'bypass'):
            for _ in range(3): request(budget, mode, 64, True)
    for index, condition in enumerate(conditions):
        request(*condition, False)
        if (index + 1) % 16 == 0:
            print(f'{index + 1}/{len(conditions)} measured requests completed', flush=True)
    ids = {row['request_id'] for row in records}
    # Read authoritative server records, preserving their original request order.
    native = [json.loads(line) for line in args.request_log.read_text().splitlines() if line.strip()]
    native = [row for row in native if row.get('run_id') == run_id]
    assert {row['request_id'] for row in native} == ids
    assert len(native) == len(records)
    with (args.output / 'server-requests.jsonl').open('x') as stream:
        for row in native: stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    summary = dict(status='passed', run_id=run_id, warmup_requests=12,
                   measured_requests=len(conditions), request_ids_matched=True,
                   server_timing_source=str(args.request_log), runtime=runtime)
    (args.output / 'receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
