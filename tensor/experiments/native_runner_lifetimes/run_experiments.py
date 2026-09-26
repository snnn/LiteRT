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
"""Sequential native runner experiments. Dumps/memory runs are never timing data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare_cases(repo, root):
    fixtures = repo / 'tensor/examples/gemma4/native/fixtures'
    perf = {}
    for n in (128, 1024, 4096):
        row = (fixtures / f'performance_{n}_64.tsv').read_text().strip()
        perf[n] = row
    (root / 'timing.tsv').write_text('\n'.join(perf.values()) + '\n')
    _, tokens, forced = perf[4096].split('\t')
    tokens = tokens.split(',')
    long_tokens = tokens + (tokens[1:] * 2)
    forced = forced.split(',')
    rows = []
    # Grow, reset, shrink, and grow again in the same compiled runtime, including
    # partial prefill chunks and local-window/alignment boundaries.
    for n, d in ((8, 33), (128, 3), (1024, 64), (4096, 2), (8192, 2),
                 (127, 3), (129, 3), (511, 3), (512, 3), (513, 3)):
        if n == 1024:
            _, prompt, decode = perf[n].split('\t')
        else:
            prompt, decode = ','.join(long_tokens[:n]), ','.join(forced[:d])
        rows.append(f'check_{n}_{d}\t{prompt}\t{decode}')
    (root / 'correctness.tsv').write_text('\n'.join(rows) + '\n')
    (root / 'smoke.tsv').write_text(rows[0] + '\n')
    (root / 'memory.tsv').write_text(perf[1024] + '\n')


def run(args, variant, suite, suffix='', chunk=128, alignment=32):
    root = args.out.resolve()
    directory = root / f'{suite}-{variant}{suffix}'
    directory.mkdir(exist_ok=False)
    if variant == 'archived':
        binary = (args.repo / '.native-tensor-build/host/bin/gemma4_native_runner').resolve()
    elif variant == 'baseline':
        binary = root / 'baseline/runner'
    else:
        binary = root / 'probe/runner'
    timing = suite == 'timing'
    dumping = suite in ('correctness', 'smoke')
    fixture = root / (suite + '.tsv')
    argv = ['taskset', '-c', '0,2,4,6', str(binary),
            '--bundle_dir=' + str(args.bundle.resolve()),
            '--cases_file=' + str(fixture), '--output_dir=' + str(directory / 'capture'),
            '--num_threads=4', '--cache_capacity=8448', '--reuse_runtimes=true',
            '--warmup_runs=' + ('0' if suite == 'smoke' else '1'),
            '--measured_runs=' + ('3' if timing else '1'),
            '--prefill_chunk_rows=' + str(chunk), '--kv_alignment=' + str(alignment),
            '--share_workspace=true', '--preserve_static_int2=true',
            '--memory_report=' + str(suite == 'memory').lower(),
            '--dump_full_logits=' + str(dumping).lower(),
            '--dump_cache=' + str(dumping).lower()]
    if variant not in ('baseline', 'archived'):
        argv += ['--experiment_release_packed_sources=' + str(variant in ('release', 'combined')).lower(),
                 '--experiment_cache_stage_shapes=' + str(variant in ('reshape', 'combined')).lower()]
    save(directory / 'command.json', {'argv': argv, 'binary_sha256': sha(binary),
         'fixture_sha256': sha(fixture), 'bundle_manifest_sha256': sha(args.bundle / 'manifest.json'),
         'timings_valid_for_benchmark': timing, 'cpu_affinity': [0, 2, 4, 6]})
    (directory / 'processes-before.txt').write_text(subprocess.check_output(
        ['ps', '-eo', 'pid,pcpu,rss,comm', '--sort=-pcpu'], text=True))
    (directory / 'meminfo-before.txt').write_text(Path('/proc/meminfo').read_text())
    start = time.monotonic()
    with (directory / 'process.log').open('w') as log:
        proc = subprocess.Popen(argv, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
        save(directory / 'status.json', {'status': 'running', 'pid': proc.pid})
        pid, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
    result = {'status': 'completed' if proc.returncode == 0 else 'failed',
              'returncode': proc.returncode, 'wall_seconds': time.monotonic() - start,
              'peak_rss_bytes': usage.ru_maxrss * 1024,
              'minor_faults': usage.ru_minflt, 'major_faults': usage.ru_majflt,
              'user_seconds': usage.ru_utime, 'system_seconds': usage.ru_stime,
              'timings_valid_for_benchmark': timing}
    save(directory / 'status.json', result)
    print(directory.name, result, flush=True)
    if proc.returncode:
        print((directory / 'process.log').read_text()[-8000:])
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=['prepare', 'smoke', 'correctness', 'memory', 'timing'])
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=['baseline', 'release', 'reshape', 'combined'])
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--alignment', type=int, default=32)
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()
    if args.suite == 'prepare':
        prepare_cases(args.repo, args.out)
        return
    if args.suite == 'timing':
        # Reverse the order in alternating blocks; each process warms up each
        # prompt length before its three measurements. Processes never overlap.
        for block in range(args.blocks):
            order = args.variants if block % 2 == 0 else list(reversed(args.variants))
            for variant in order:
                run(args, variant, args.suite, f'-b{block}' + args.suffix, args.chunk, args.alignment)
    else:
        for variant in args.variants:
            run(args, variant, args.suite, args.suffix, args.chunk, args.alignment)


if __name__ == '__main__':
    main()
