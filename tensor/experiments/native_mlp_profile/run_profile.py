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
"""Run matched decode profiling, controls, and full-logit validation."""
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


def runner_args(binary, bundle, fixture, output, repeats=3, threads=4):
    return ['taskset', '-c', '0,2,4,6', str(binary),
            '--bundle_dir=' + str(bundle), '--cases_file=' + str(fixture),
            '--output_dir=' + str(output), '--num_threads=' + str(threads),
            '--cache_capacity=8448', '--prefill_chunk_rows=128', '--kv_alignment=32',
            '--reuse_runtimes=true', '--warmup_runs=1', '--measured_runs=' + str(repeats),
            '--share_workspace=true', '--preserve_static_int2=true',
            '--memory_report=false', '--dump_full_logits=false']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'profile', 'off', 'correctness'],
                        default=['baseline', 'profile', 'off', 'correctness'])
    parser.add_argument('--suffix', default='')
    parser.add_argument('--threads', type=int, default=4)
    a = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    fixture = repo / 'tensor/examples/gemma4/native/fixtures/performance_1024_64.tsv'
    for variant in a.variants:
        directory = a.out.resolve() / (variant + a.suffix)
        directory.mkdir(exist_ok=False)
        binary = a.baseline.resolve() if variant == 'baseline' else a.out.resolve() / 'runner'
        argv = runner_args(binary, a.bundle.resolve(), fixture, directory / 'capture',
                           1 if variant == 'correctness' else 3, a.threads)
        if variant != 'baseline':
            argv += ['--profile_post_ops=' + str(variant != 'off').lower()]
        if variant == 'correctness':
            argv[argv.index('--dump_full_logits=false')] = '--dump_full_logits=true'
        save(directory / 'command.json', {'argv': argv, 'binary_sha256': sha(binary),
             'fixture_sha256': sha(fixture), 'bundle_manifest_sha256': sha(a.bundle / 'manifest.json'),
             'timings_valid_for_benchmark': variant in ('baseline', 'off')})
        (directory / 'processes-before.txt').write_text(subprocess.check_output(
            ['ps', '-eo', 'pid,pcpu,rss,comm', '--sort=-pcpu'], text=True))
        start = time.monotonic()
        with (directory / 'process.log').open('w') as log:
            proc = subprocess.Popen(argv, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            save(directory / 'status.json', {'status': 'running', 'pid': proc.pid})
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
        result = {'returncode': proc.returncode, 'wall_seconds': time.monotonic() - start,
                  'peak_rss_bytes': usage.ru_maxrss * 1024}
        save(directory / 'status.json', result)
        print(directory.name, result, flush=True)
        if proc.returncode:
            print((directory / 'process.log').read_text()[-8000:])
            raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
