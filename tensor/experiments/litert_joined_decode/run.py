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
"""Run the same joined graph natively and through the LiteRT interpreter."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture(args, name, mapped, fixture, diagnostic=False, memory=False, repeats=3):
    directory = args.out / (args.label + name)
    directory.mkdir(parents=True, exist_ok=False)
    command = ['taskset', '-c', '0,2,4,6', str(args.binary.resolve()),
               '--bundle_dir=' + str(args.bundle), '--cases_file=' + str(fixture),
               '--output_dir=' + str(directory / 'capture'), '--num_threads=4',
               '--cache_capacity=8448', '--prefill_chunk_rows=128', '--kv_alignment=32',
               '--reuse_runtimes=true', '--share_workspace=true', '--preserve_static_int2=true',
               '--match_delegate_worker_policy=true',
               '--joined_decode_graph=true', '--litert_decode_graph=' + str(mapped).lower(),
               '--warmup_runs=' + str(0 if diagnostic else 1), '--measured_runs=' + str(repeats),
               '--memory_report=' + str(memory).lower(),
               '--dump_full_logits=' + str(diagnostic).lower(), '--dump_cache=' + str(diagnostic).lower()]
    timing_valid = not diagnostic and not memory
    save(directory / 'command.json', {'argv': command, 'binary_sha256': sha(args.binary),
         'fixture_sha256': sha(fixture), 'timings_valid_for_benchmark': timing_valid})
    print('START', name, flush=True)
    start = time.monotonic()
    with (directory / 'process.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    save(directory / 'status.json', {'returncode': result.returncode,
         'wall_seconds': time.monotonic()-start, 'timings_valid_for_benchmark': timing_valid})
    if result.returncode:
        raise RuntimeError((directory / 'process.log').read_text()[-7000:])
    runs = [json.loads(p.read_text()) for p in sorted((directory / 'capture').glob('*.run_*.json'))]
    for run in runs:
        assert run['litert_decode_graph'] == mapped
        assert all(p['decode_runtime_invocations'] == 1 for p in run['passes'])
    if timing_valid:
        print('DONE', name, 'decode_ms', statistics.median(statistics.mean(
            p['elapsed_ms'] for p in run['passes'][1:]) for run in runs), flush=True)
    else:
        print('DONE', name, flush=True)
    return directory / 'capture'


def compare(reference, candidate):
    files = sorted(reference.glob('*.f32*'))
    assert files and {p.name for p in files} == {p.name for p in candidate.glob('*.f32*')}
    differences = [p.name for p in files if p.read_bytes() != (candidate / p.name).read_bytes()]
    result = {'files': len(files), 'logit_vectors': sum(p.suffix == '.f32' for p in files),
              'kv_files': sum(p.suffix == '.i8' for p in files),
              'bytes_compared': sum(p.stat().st_size for p in files), 'different_files': differences,
              'reference': str(reference), 'candidate': str(candidate)}
    save(candidate.parent / 'comparison.json', result)
    if differences:
        raise RuntimeError(f'{len(differences)} files differ; see comparison.json')
    print('BIT-IDENTICAL', result['logit_vectors'], 'logit vectors and', result['kv_files'], 'KV files', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/data/bt/tmp/gemma4-litert-single-graph-20260923'))
    parser.add_argument('--binary', type=Path, default=Path('bazel-bin/tensor/experiments/litert_joined_decode/runner'))
    parser.add_argument('--bundle', type=Path, default=Path('/data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle'))
    parser.add_argument('--mode', choices=['minimal', 'smoke', 'correctness', 'timing', 'memory'], required=True)
    parser.add_argument('--only', choices=['native', 'litert'])
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    perf = Path('tensor/examples/gemma4/native/fixtures/performance_1024_64.tsv').resolve()
    if args.mode in ('minimal', 'smoke', 'correctness'):
        fixture = args.out / (args.mode + '.tsv')
        if args.mode == 'minimal':
            forced = perf.read_text().strip().split('\t')[2].split(',')[:3]
            fixture.write_text('empty_history\t2\t' + ','.join(forced) + '\n')
        else:
            shutil.copyfile(Path('/data/bt/tmp/gemma4-native-single-graph-20260923') / fixture.name, fixture)
        results = {}
        for variant, mapped in [('native', False), ('litert', True)]:
            if args.only and args.only != variant:
                continue
            results[variant] = capture(args, args.mode + '-' + variant, mapped, fixture, diagnostic=True, repeats=1)
        if len(results) == 2:
            compare(results['native'], results['litert'])
    elif args.mode == 'timing':
        for block, order in enumerate([(False, True), (True, False)]):
            for mapped in order:
                variant = 'litert' if mapped else 'native'
                if args.only and args.only != variant:
                    continue
                capture(args, f'timing-b{block}-' + variant, mapped, perf)
    else:
        for mapped in (False, True):
            variant = 'litert' if mapped else 'native'
            if args.only and args.only != variant:
                continue
            capture(args, 'memory-' + variant, mapped, perf, memory=True, repeats=1)


if __name__ == '__main__':
    main()
