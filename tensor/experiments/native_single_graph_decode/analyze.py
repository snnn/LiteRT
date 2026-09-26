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
"""Validate and summarize single-graph decode experiments."""
import argparse
import json
from pathlib import Path
import statistics as stats

from run import read, save


def history(d):
    return [d['prompt_token_ids'], d['forced_decode_token_ids'],
            [(p['input_ids'], p['argmax_id']) for p in d['passes']]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--save', type=Path)
    a = p.parse_args()
    root = a.out
    result = {'raw_artifacts': str(root), 'scope': 'Single-token decode; unchanged staged prefill; MTP excluded.',
              'build': read(root / 'build-v3/commands.json'),
              'correctness': read(root / 'correctness-joined/comparison.json'),
              'archives': read(root / 'archive-identity.json'), 'timing': {}, 'memory': {}}
    assert result['archives']['same_as_previous_experiment']
    assert not result['correctness']['different_files']
    old = Path('/data/bt/tmp/gemma4-native-next-20260923/correctness-baseline/capture')
    reference = root / 'correctness-staged/capture'
    original_files = sorted(old.glob('*.f32*'))
    assert len(original_files) == 3999
    for path in original_files:
        assert path.read_bytes() == (reference / path.name).read_bytes(), path
    result['unchanged_control'] = {'previous_reference': str(old), 'bit_identical_files': len(original_files)}
    truth = {}
    for path in reference.glob('*.run_*.json'):
        d = read(path)
        truth[d['case_id']] = history(d)
    for path in (root / 'correctness-joined/capture').glob('*.run_*.json'):
        d = read(path)
        assert history(d) == truth[d['case_id']]
        assert all(p['decode_runtime_invocations'] == 1 for p in d['passes'])
    shared_history = None
    for variant, invocations in [('staged',107), ('joined',1)]:
        sessions, blocks = [], []
        for block in range(2):
            d = root / f'timing-b{block}-{variant}'
            status = read(d / 'status.json')
            assert status['returncode'] == 0 and status['timings_valid_for_benchmark']
            rows = []
            for path in sorted((d / 'capture').glob('*.run_*.json')):
                r = read(path)
                assert len(r['passes']) == 65
                if shared_history is None:
                    shared_history = history(r)
                assert history(r) == shared_history
                assert all(v['decode_runtime_invocations'] == invocations for v in r['passes'])
                rows.append({**{k: stats.mean(v[k] for v in r['passes'][1:]) for k in
                    ('elapsed_ms','forward_ms','cache_handoff_or_update_ms')},
                    'first_logit_ms': r['passes'][0]['elapsed_ms']})
            assert len(rows) == 3
            blocks.append({k: stats.median(v[k] for v in rows) for k in rows[0]})
            sessions.extend(rows)
        result['timing'][variant] = {'sessions': sessions, 'block_medians': blocks,
            'median_six_session_means': {k: stats.median(v[k] for v in sessions) for k in sessions[0]},
            'decode_range_ms': [min(v['elapsed_ms'] for v in sessions), max(v['elapsed_ms'] for v in sessions)],
            'decode_runtime_invocations': invocations}
        d = root / ('memory-' + variant)
        assert read(d / 'status.json')['returncode'] == 0
        rows = [json.loads(line) for line in (d / 'capture/memory.jsonl').read_text().splitlines()]
        last = [r for r in rows if r['phase'] == 'after_last_decode'][-1]
        assert last['smaps_rollup_available'] and last['peak_rss_available']
        result['memory'][variant] = {
            'peak_rss_mib': read(d / 'capture/run.json')['peak_process_rss_kib'] / 1024,
            **{k.lower() + '_mib': last['smaps_rollup'][k] / 2**20 for k in ('Rss','Anonymous','Swap')},
            'allocation_counters': last['allocation_counters']}
    x, y = (result['timing'][v] for v in ('staged','joined'))
    result['timing']['decode_reduction_percent'] = 100 * (1-y['median_six_session_means']['elapsed_ms']/x['median_six_session_means']['elapsed_ms'])
    result['timing']['block_decode_reduction_percent'] = [100*(1-b['elapsed_ms']/a['elapsed_ms']) for a,b in zip(x['block_medians'],y['block_medians'])]
    # Logical concatenation output bytes, not measured DRAM traffic. New K/V
    # rows and alignment padding are included. This E2B has 12 local D=256
    # owners and 3 global D=512 owners; every owner has separate K and V.
    copy_bytes = []
    for start in range(1024,1088):
        padded_end=(start+1+31)//32*32
        begin=max(0,start-512+1)//32*32
        copy_bytes.append(2*(12*256*(padded_end-begin)+3*512*padded_end))
    result['joined_kv_concat_logical_bytes_per_decode'] = {
        'mean': stats.mean(copy_bytes), 'minimum': min(copy_bytes), 'maximum': max(copy_bytes)}
    save(root / 'analysis.json', result)
    if a.save:
        save(a.save,result)
    print('Correctness', result['correctness'])
    for k in ('staged','joined'):
        print(k, 'timing', result['timing'][k]['median_six_session_means'])
        print(k, 'memory', {n:v for n,v in result['memory'][k].items() if n!='allocation_counters'})
        print(k, 'allocations', result['memory'][k]['allocation_counters'])
    print('Decode reduction', result['timing']['decode_reduction_percent'], result['timing']['block_decode_reduction_percent'])


if __name__ == '__main__':
    main()
