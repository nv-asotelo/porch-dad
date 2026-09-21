#!/usr/bin/env python3
"""Re-extract six unmodified full frames from the verified user recording."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path, help='New output directory; must not exist')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('manifest.json').read_text())
    checksum = hashlib.sha256()
    with args.source.open('rb') as recording:
        for block in iter(lambda: recording.read(1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != manifest['source']['sha256']:
        parser.error('Source SHA-256 does not match this evidence manifest')
    args.destination.mkdir(parents=True, exist_ok=False)
    indices = [frame['source_frame_index_zero_based'] for frame in manifest['frames']]
    expression = '+'.join(f'eq(n\\,{index})' for index in indices)
    subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(args.source),
                    '-vf', f'select={expression}', '-fps_mode', 'vfr',
                    str(args.destination / 'decoded-%02d.png')], check=True)
    decoded = sorted(args.destination.glob('decoded-*.png'))
    if len(decoded) != len(indices):
        raise RuntimeError(f'Expected {len(indices)} frames, extracted {len(decoded)}')
    for path, frame in zip(decoded, manifest['frames']):
        path.rename(args.destination / frame['file'])
    shutil.copyfile(Path(__file__).with_name('manifest.json'), args.destination / 'manifest.json')
    print(f'Extracted {len(indices)} original-resolution frames into {args.destination}')
    print('PNG encoding can vary across FFmpeg versions; manifest source indices and source hash identify the evidence.')


if __name__ == '__main__':
    main()
