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
"""Check and benchmark staged, joined native, and joined LiteRT prefill."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

VARIANTS = {
    'staged': (False, False, False),
    'native': (True, False, False),
    'litert': (True, True, True),
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture(args, name, variant, fixture, diagnostic=False, memory=False):
    directory = args.out / (args.label + name)
    directory.mkdir(parents=True, exist_ok=False)
    joined, mapped, mapped_decode = VARIANTS[variant]
    command = ['taskset', '-c', args.cpus, str(args.binary.resolve()),
               '--bundle_dir=' + str(args.bundle), '--cases_file=' + str(fixture),
               '--output_dir=' + str(directory / 'capture'), '--num_threads=4',
               '--cache_capacity=8448', '--prefill_chunk_rows=' + str(args.chunk_rows),
               '--kv_alignment=32', '--reuse_runtimes=true', '--share_workspace=true',
               '--preserve_static_int2=true', '--match_delegate_worker_policy=true',
               '--joined_decode_graph=true',
               '--joined_prefill_graph=' + str(joined).lower(),
               '--litert_prefill_graph=' + str(mapped).lower(),
               '--litert_decode_graph=' + str(mapped_decode).lower(),
               '--warmup_runs=' + str(0 if diagnostic else 1),
               '--measured_runs=' + str(1 if diagnostic or memory else args.repeats),
               '--memory_report=' + str(memory).lower(),
               '--dump_full_logits=' + str(diagnostic).lower(), '--dump_cache=' + str(diagnostic).lower()]
    valid = not diagnostic and not memory
    save(directory / 'command.json', {'argv': command, 'binary_sha256': sha(args.binary),
         'fixture_sha256': sha(fixture), 'timings_valid_for_benchmark': valid})
    print('START', name, flush=True)
    start = time.monotonic()
    with (directory / 'process.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    save(directory / 'status.json', {'returncode': result.returncode,
         'wall_seconds': time.monotonic()-start, 'timings_valid_for_benchmark': valid})
    if result.returncode:
        raise RuntimeError((directory / 'process.log').read_text()[-7000:])
    runs = [json.loads(p.read_text()) for p in sorted((directory / 'capture').glob('*.run_*.json'))]
    for run in runs:
        assert run['litert_prefill_graph'] == mapped
        assert run['joined_prefill_graph'] == joined
        assert run['litert_decode_graph'] == mapped_decode
        assert all(p['decode_runtime_invocations'] == 1 for p in run['passes'])
        chunks = (len(run['prompt_token_ids']) - 2 + args.chunk_rows) // args.chunk_rows
        assert run['passes'][0]['prefill_runtime_invocations'] == chunks * (1 if joined else 44)
    for case in sorted({r['case_id'] for r in runs}):
        selected = [r['passes'][0] for r in runs if r['case_id'] == case]
        print('DONE', name, case, 'prefix_ms', statistics.median(p['prefill_prefix_ms'] for p in selected), flush=True)
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
    print('COMPARISON', len(differences), 'different files of', len(files), flush=True)
    if differences:
        raise RuntimeError(f'{len(differences)} files differ; see comparison.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/data/bt/tmp/gemma4-litert-chunked-prefill-20260924'))
    parser.add_argument('--binary', type=Path, default=Path('bazel-bin/tensor/experiments/litert_chunked_prefill/runner'))
    parser.add_argument('--bundle', type=Path, default=Path('/data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle'))
    parser.add_argument('--mode', choices=['smoke', 'correctness', 'timing', 'memory'], required=True)
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--label', default='')
    parser.add_argument('--chunk-rows', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--cpus', default='0,2,4,6')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    perf = Path('tensor/examples/gemma4/native/fixtures/performance_1024_64.tsv').resolve()
    if args.fixture:
        fixture = args.fixture.resolve()
    elif args.mode == 'correctness':
        fixture = Path('/data/bt/tmp/gemma4-native-single-graph-20260923/correctness.tsv')
    else:
        fields = perf.read_text().strip().split('\t')
        tokens = fields[1].split(',')
        forced = ','.join(fields[2].split(',')[:2])
        lengths = [1, 2, 8, 128, 129, 130, 1024] if args.mode == 'smoke' else [128, 1024, 4096, 8192]
        fixture = args.out / (args.label + args.mode + '.tsv')
        fixture.write_text(''.join(f'p{n}\t' + ','.join((tokens * ((n+1023)//1024))[:n]) + '\t' + forced + '\n' for n in lengths))
    if args.mode in ('smoke', 'correctness'):
        results = {v: capture(args, args.mode+'-'+v, v, fixture, diagnostic=True) for v in args.variants}
        if 'staged' in results:
            for v in args.variants:
                if v != 'staged':
                    compare(results['staged'], results[v])
    elif args.mode == 'timing':
        for block in range(args.blocks):
            order = args.variants if block % 2 == 0 else list(reversed(args.variants))
            for variant in order:
                capture(args, f'timing-b{block}-'+variant, variant, fixture)
    else:
        for variant in args.variants:
            capture(args, 'memory-'+variant, variant, fixture, memory=True)


if __name__ == '__main__':
    main()
