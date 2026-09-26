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
"""Check 2/4/8 P-core settings on the audited i9-12900K, without code changes."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from run_profile import runner_args, save, sha


def validate(a, root, repo, fixture, hashes):
    previous = json.loads((root / 'analysis.json').read_text())
    assert previous['hashes'] == hashes
    assert a.reference is not None
    reference = json.loads((a.reference / 'check_1024_64.run_000.json').read_text())
    validations = []
    for threads in (2, 8):
        directory = root / f'correctness-t{threads}'
        directory.mkdir(exist_ok=False)
        argv = runner_args(a.binary.resolve(), a.bundle.resolve(), fixture,
                           directory / 'capture', repeats=1, threads=threads)
        argv[2] = ','.join(str(i * 2) for i in range(threads))
        argv[argv.index('--dump_full_logits=false')] = '--dump_full_logits=true'
        save(directory / 'command.json', dict(hashes, argv=argv,
             timings_valid_for_benchmark=False))
        with (directory / 'process.log').open('w') as log:
            proc = subprocess.run(argv, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        save(directory / 'status.json', {'returncode': proc.returncode})
        if proc.returncode:
            raise RuntimeError((directory / 'process.log').read_text()[-8000:])
        record = json.loads((directory / 'capture/performance_1024_64.run_000.json').read_text())
        for field in ('prompt_token_ids', 'forced_decode_token_ids'):
            assert record[field] == reference[field]
        files = sorted((directory / 'capture').glob('*.f32'))
        assert len(files) == 65
        size = 0
        for path in files:
            ref = a.reference / path.name.replace('performance_1024_64', 'check_1024_64')
            assert path.stat().st_size == 262144 * 4
            assert path.read_bytes() == ref.read_bytes(), path
            size += path.stat().st_size
        item = {'threads': threads, 'full_logit_vectors': len(files),
                'bytes_bit_identical': size, 'reference': str(a.reference)}
        validations.append(item)
        print('Validated', item, flush=True)
    previous['full_logit_validation'] = validations
    save(root / 'analysis.json', previous)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--reference', type=Path)
    a = p.parse_args()
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    fixture = repo / 'tensor/examples/gemma4/native/fixtures/performance_1024_64.tsv'
    (root / 'cpu-topology.txt').write_text(subprocess.check_output(
        ['lscpu', '-e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ'], text=True))
    assert set(range(0, 16, 2)).issubset(os.sched_getaffinity(0))
    hashes = {'binary_sha256': sha(a.binary), 'fixture_sha256': sha(fixture),
              'bundle_manifest_sha256': sha(a.bundle / 'manifest.json')}
    if a.validate_only:
        validate(a, root, repo, fixture, hashes)
        return
    result = {'hashes': hashes, 'captures': []}
    history = None
    for block, order in enumerate(((4, 2, 8), (8, 2, 4))):
        for threads in order:
            directory = root / f'b{block}-t{threads}'
            directory.mkdir(exist_ok=False)
            affinity = ','.join(str(i * 2) for i in range(threads))
            argv = runner_args(a.binary.resolve(), a.bundle.resolve(), fixture,
                               directory / 'capture', threads=threads)
            argv[2] = affinity
            save(directory / 'command.json', dict(hashes, argv=argv))
            (directory / 'processes-before.txt').write_text(subprocess.check_output(
                ['ps', '-eo', 'pid,pcpu,rss,comm', '--sort=-pcpu'], text=True))
            start = time.monotonic()
            with (directory / 'process.log').open('w') as log:
                proc = subprocess.Popen(argv, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                _, status, usage = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(status)
            save(directory / 'status.json', {'returncode': proc.returncode,
                 'wall_seconds': time.monotonic() - start,
                 'user_seconds': usage.ru_utime, 'system_seconds': usage.ru_stime})
            if proc.returncode:
                raise RuntimeError((directory / 'process.log').read_text()[-8000:])
            values = []
            for path in sorted((directory / 'capture').glob('*.run_*.json')):
                d = json.loads(path.read_text())
                assert d['timings_valid_for_benchmark'] and d['num_threads'] == threads
                assert len(d['passes']) == 65
                actual = [d['prompt_token_ids'], d['forced_decode_token_ids'],
                          [(v['input_ids'], v['argmax_id']) for v in d['passes']]]
                if history is None:
                    history = actual
                assert history == actual
                fields = ('elapsed_ms', 'post_attention_run_ms', 'attention_run_ms',
                          'projection_run_ms', 'forward_ms')
                row = {k: statistics.mean(v[k] for v in d['passes'][1:]) for k in fields}
                row['first_logit_ms'] = d['passes'][0]['elapsed_ms']
                values.append(row)
            assert len(values) == 3
            item = {'block': block, 'threads': threads, 'affinity': affinity,
                    'sessions': values,
                    'median_session': {k: statistics.median(v[k] for v in values) for k in values[0]}}
            result['captures'].append(item)
            save(root / 'analysis.json', result)
            print(directory.name, item['median_session'], flush=True)
    result['summary'] = {str(n): {
        k: statistics.median(v[k] for capture in result['captures'] if capture['threads'] == n
                             for v in capture['sessions'])
        for k in result['captures'][0]['sessions'][0]} for n in (2, 4, 8)}
    result['all_histories_and_argmax_match'] = True
    save(root / 'analysis.json', result)
    print('Completed', result['summary'], flush=True)


if __name__ == '__main__':
    main()
