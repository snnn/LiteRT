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
"""Audit same-binary correctness, timing blocks, and allocation owners."""
import argparse
import json
from pathlib import Path
import statistics as stats
from run import save


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/data/bt/tmp/gemma4-litert-chunked-prefill-20260924'))
    parser.add_argument('--label', default='final-')
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    def directory(name):
        return args.out / (args.label + name)
    result = {'artifacts': str(args.out), 'label': args.label,
              'scope': 'Gemma4 E2B CPU chunked prefill, 128 rows, KV capacity 8448; MTP excluded.',
              'correctness': {}, 'timing': {}, 'memory': {}, 'kernel_inventory': {}}
    hashes = set()
    histories = {}
    for variant in ('staged', 'native', 'litert'):
        diagnostic = directory('correctness-' + variant)
        hashes.add(read(diagnostic / 'command.json')['binary_sha256'])
        assert read(diagnostic / 'status.json')['returncode'] == 0
        if variant != 'staged':
            cmp = read(diagnostic / 'comparison.json')
            assert not cmp['different_files']
            result['correctness'][variant] = cmp
            result['kernel_inventory'][variant] = {phase: read(diagnostic / 'capture' / (phase + '_kernel_inventory.json')) for phase in ('prefill','decode')}
        samples = {}
        for block in range(2):
            path = directory(f'timing-b{block}-' + variant)
            hashes.add(read(path / 'command.json')['binary_sha256'])
            status = read(path / 'status.json')
            assert status['returncode'] == 0 and status['timings_valid_for_benchmark']
            for f in sorted((path / 'capture').glob('*.run_*.json')):
                run = read(f)
                assert run['timings_valid_for_benchmark'] and not run['memory_report']
                assert run['static_int2_operator_count'] == 60 and run['static_int2_tensor_count'] == 60
                assert run['prefill_rows'] == 128 and run['kv_capacity'] == 8448
                assert all(p['decode_runtime_invocations'] == 1 for p in run['passes'])
                case = run['case_id']
                history = [run['prompt_token_ids'], run['forced_decode_token_ids'], [p['argmax_id'] for p in run['passes']]]
                assert histories.setdefault(case,history) == history
                first = run['passes'][0]
                row = {'block': block, 'prefix_ms': first['prefill_prefix_ms'],
                       'first_logits_ms': first['elapsed_ms'], 'prefix_tokens': len(run['prompt_token_ids'])-1,
                       'prefill_runtime_invocations': first['prefill_runtime_invocations']}
                samples.setdefault(case, []).append(row)
        result['timing'][variant] = {}
        for case, rows in samples.items():
            assert len(rows) == 6
            ms = stats.median(r['prefix_ms'] for r in rows)
            result['timing'][variant][case] = {
                'median_prefix_ms': ms, 'prefix_tokens_per_second': rows[0]['prefix_tokens']*1000/ms,
                'median_first_logits_ms': stats.median(r['first_logits_ms'] for r in rows),
                'block_median_prefix_ms': [stats.median(r['prefix_ms'] for r in rows if r['block']==b) for b in range(2)],
                'samples': rows}
        path = directory('memory-' + variant)
        hashes.add(read(path / 'command.json')['binary_sha256'])
        status = read(path / 'status.json')
        assert status['returncode'] == 0 and not status['timings_valid_for_benchmark']
        memory = [json.loads(l) for l in (path / 'capture/memory.jsonl').read_text().splitlines()]
        result['memory'][variant] = {'peak_process_rss_mib': read(path/'capture/run.json')['peak_process_rss_kib']/1024, 'cases': {}}
        for case in samples:
            by_phase = {}
            for phase in ('after_prefill_prefix','after_last_decode'):
                row = [r for r in memory if r['case_id'] == case and r['phase']==phase][-1]
                assert row['smaps_rollup_available'] and row['peak_rss_available']
                assert row['smaps_rollup']['Swap']==0
                assert row['allocation_counters']['unique_workspace_count']==1
                by_phase[phase] = {**{k.lower()+'_mib': row['smaps_rollup'][k]/2**20 for k in ('Rss','Anonymous','Swap')},
                                 'allocation_counters': row['allocation_counters']}
            result['memory'][variant]['cases'][case] = by_phase
    assert len(hashes)==1
    result['binary_sha256']=hashes.pop()
    result['native_litert_kernel_inventories_equal']=result['kernel_inventory']['native']==result['kernel_inventory']['litert']
    assert result['native_litert_kernel_inventories_equal']
    for variant in ('native','litert'):
        for case, row in result['timing'][variant].items():
            row['latency_change_vs_staged_percent']=100*(row['median_prefix_ms']/result['timing']['staged'][case]['median_prefix_ms']-1)
    save(args.out / (args.label+'analysis.json'),result)
    if args.save: save(args.save,result)
    for case in result['timing']['staged']:
        print(case, {v: round(result['timing'][v][case]['median_prefix_ms'],3) for v in ('staged','native','litert')})
    print('Peak RSS MiB', {v: round(m['peak_process_rss_mib'],2) for v,m in result['memory'].items()})
    print('Bit identical; kernel inventories equal; one scratch workspace per variant.')


if __name__=='__main__':
    main()
