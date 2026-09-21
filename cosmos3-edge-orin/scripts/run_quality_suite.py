#!/usr/bin/env python3
"""Capture the frozen six-image screen; semantic grading remains explicit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from benchmark import build_payload, stream_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8090/v1/chat/completions')
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', type=Path,
                        help='Optional separately labeled smoke manifest; default is the frozen six-image screen')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest.resolve() if args.manifest else root / 'benchmarks/fixtures/jpeg-manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    settings = manifest['request_settings']
    with Path(args.output).open('x') as output:
        for case in manifest['cases']:
            image = (manifest_path.parent / case['filename']).read_bytes()
            if hashlib.sha256(image).hexdigest() != case['sha256']:
                raise ValueError('Fixture SHA256 mismatch: ' + case['id'])
            workload = dict(model=settings['model_alias'], prompt=case['prompt'],
                            max_tokens=settings['max_tokens'], temperature=settings['temperature'],
                            top_p=settings['top_p'], image_mime_type=case['mime_type'])
            record = dict(candidate=args.candidate, case_id=case['id'],
                          observed_utc=datetime.now(timezone.utc).isoformat(),
                          suite_id=manifest['suite_id'],
                          manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                          image_sha256=case['sha256'], workload=workload,
                          expected_facts=case['expected_facts'],
                          semantic_grade='pending',
                          latency_note='Diagnostic sample only; cache state and warmup uncontrolled.')
            record.update(stream_request(args.url, build_payload(workload, image), timeout=120))
            output.write(json.dumps(record) + '\n')
            output.flush()
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
