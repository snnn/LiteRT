#!/usr/bin/env python3
# Copyright 2026 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare staged and single-graph decode, retaining unchanged staged prefill."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as stats
import subprocess
import time


def save(p, d):
    p.write_text(json.dumps(d, indent=2) + '\n')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def capture(a, name, joined, fixture, diagnostic=False, memory=False, repeats=3):
    d = a.out / name
    d.mkdir(exist_ok=False)
    argv = ['taskset', '-c', '0,2,4,6', str(a.binary), '--bundle_dir=' + str(a.bundle),
            '--cases_file=' + str(fixture), '--output_dir=' + str(d / 'capture'),
            '--num_threads=4', '--cache_capacity=8448', '--prefill_chunk_rows=128',
            '--kv_alignment=32', '--reuse_runtimes=true', '--share_workspace=true',
            '--preserve_static_int2=true', '--joined_decode_graph=' + str(joined).lower(),
            '--warmup_runs=' + str(0 if diagnostic else 1), '--measured_runs=' + str(repeats),
            '--memory_report=' + str(memory).lower(), '--dump_full_logits=' + str(diagnostic).lower(),
            '--dump_cache=' + str(diagnostic).lower()]
    valid_timing = not diagnostic and not memory
    save(d / 'command.json', {'argv': argv, 'timings_valid_for_benchmark': valid_timing,
                             'binary_sha256': sha(a.binary), 'fixture_sha256': sha(fixture)})
    print('START', name, flush=True)
    start = time.monotonic()
    with (d / 'process.log').open('w') as f:
        result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=900)
    save(d / 'status.json', {'returncode': result.returncode, 'wall_seconds': time.monotonic()-start,
                            'timings_valid_for_benchmark': valid_timing})
    if result.returncode:
        raise RuntimeError((d / 'process.log').read_text()[-5000:])
    runs = [read(p) for p in sorted((d / 'capture').glob('*.run_*.json'))]
    for run in runs:
        assert all(p['decode_runtime_invocations'] == (1 if joined else 107) for p in run['passes'])
    if valid_timing:
        print('DONE', name, 'decode_ms', stats.median(stats.mean(p['elapsed_ms'] for p in r['passes'][1:]) for r in runs), flush=True)
    else:
        print('DONE', name, flush=True)
    return d / 'capture'


def compare(reference, candidate):
    files = sorted(reference.glob('*.f32*'))
    assert files and {p.name for p in files} == {p.name for p in candidate.glob('*.f32*')}
    differences, total, logits, kv = [], 0, 0, 0
    for p in files:
        q = candidate / p.name
        if p.read_bytes() != q.read_bytes():
            differences.append(p.name)
        total += p.stat().st_size
        logits += p.suffix == '.f32'
        kv += p.suffix == '.i8'
    result = {'files': len(files), 'logit_vectors': logits, 'kv_files': kv,
              'bytes_compared': total, 'different_files': differences,
              'reference': str(reference), 'candidate': str(candidate)}
    save(candidate.parent / 'comparison.json', result)
    assert not differences, result
    print('BIT-IDENTICAL', logits, 'logit vectors and', kv, 'KV files', flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--bundle', type=Path, default=Path('/data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle'))
    p.add_argument('--mode', choices=['smoke','correctness','timing','memory'], required=True)
    a = p.parse_args()
    if a.mode in ('smoke', 'correctness'):
        fixture = a.out / (a.mode + '.tsv')
        if a.mode == 'smoke' and not fixture.exists():
            row = Path('tensor/examples/gemma4/native/fixtures/performance_1024_64.tsv').read_text().strip().split('\t')
            fixture.write_text('empty_history\t2\t' + ','.join(row[2].split(',')[:33]) + '\n'
                               + 'p1024\t' + row[1] + '\t' + row[2] + '\n')
        if a.mode == 'correctness':
            first = (a.out / 'smoke.tsv').read_text().splitlines()[0]
            previous = Path('/data/bt/tmp/gemma4-native-next-20260923/correctness.tsv').read_text()
            fixture.write_text(first + '\n' + previous)
        off = capture(a, a.mode + '-staged', False, fixture, diagnostic=True, repeats=1)
        on = capture(a, a.mode + '-joined', True, fixture, diagnostic=True, repeats=1)
        compare(off, on)
    elif a.mode == 'timing':
        fixture = Path('tensor/examples/gemma4/native/fixtures/performance_1024_64.tsv').resolve()
        for block, order in enumerate([(False,True),(True,False)]):
            for joined in order:
                capture(a, f'timing-b{block}-' + ('joined' if joined else 'staged'), joined, fixture)
    else:
        fixture = Path('tensor/examples/gemma4/native/fixtures/performance_1024_64.tsv').resolve()
        for joined in (False,True):
            capture(a, 'memory-' + ('joined' if joined else 'staged'), joined, fixture, memory=True, repeats=1)


if __name__ == '__main__':
    main()
