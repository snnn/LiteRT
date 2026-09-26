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
"""Audit correctness, kernel inventories, timing, and separate memory captures."""
import argparse
import json
from pathlib import Path
import statistics as stats
from run import save


def read(path):
    return json.loads(path.read_text())


def history(run):
    return [run['prompt_token_ids'], run['forced_decode_token_ids'],
            [(p['input_ids'], p['argmax_id']) for p in run['passes']]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/data/bt/tmp/gemma4-litert-single-graph-20260923'))
    parser.add_argument('--label', default='final-')
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    root = args.out
    def directory(name):
        return root / (args.label + name)
    comparison = read(directory('correctness-litert') / 'comparison.json')
    assert not comparison['different_files']
    result = {'artifacts': str(root), 'label': args.label,
              'scope': 'Joined decode through LiteRT Interpreter + XNNPACK; native staged prefill retained; MTP excluded.',
              'correctness': comparison, 'timing': {}, 'memory': {}, 'kernel_inventory': {}}
    binary_hashes = set()
    shared_history = None
    for variant in ('native', 'litert'):
        result['kernel_inventory'][variant] = read(directory('correctness-' + variant) / 'capture/decode_kernel_inventory.json')
        binary_hashes.add(read(directory('correctness-' + variant) / 'command.json')['binary_sha256'])
        sessions, blocks = [], []
        for block in range(2):
            path = directory(f'timing-b{block}-' + variant)
            status = read(path / 'status.json')
            assert status['returncode'] == 0 and status['timings_valid_for_benchmark']
            command = read(path / 'command.json')
            binary_hashes.add(command['binary_sha256'])
            rows = []
            for capture in sorted((path / 'capture').glob('*.run_*.json')):
                run = read(capture)
                assert len(run['passes']) == 65
                if shared_history is None:
                    shared_history = history(run)
                assert history(run) == shared_history
                assert all(p['decode_runtime_invocations'] == 1 for p in run['passes'])
                assert run['litert_decode_graph'] == (variant == 'litert')
                rows.append({**{key: stats.mean(p[key] for p in run['passes'][1:])
                    for key in ('elapsed_ms', 'forward_ms', 'cache_handoff_or_update_ms')},
                    'first_logit_ms': run['passes'][0]['elapsed_ms']})
            assert len(rows) == 3
            blocks.append({key: stats.median(r[key] for r in rows) for key in rows[0]})
            sessions.extend(rows)
        result['timing'][variant] = {'sessions': sessions, 'block_medians': blocks,
            'median_six_session_means': {key: stats.median(r[key] for r in sessions) for key in sessions[0]},
            'decode_range_ms': [min(r['elapsed_ms'] for r in sessions), max(r['elapsed_ms'] for r in sessions)]}
        path = directory('memory-' + variant)
        assert read(path / 'status.json')['returncode'] == 0
        binary_hashes.add(read(path / 'command.json')['binary_sha256'])
        memory = [json.loads(line) for line in (path / 'capture/memory.jsonl').read_text().splitlines()]
        last = [r for r in memory if r['phase'] == 'after_last_decode'][-1]
        assert last['smaps_rollup_available'] and last['peak_rss_available']
        result['memory'][variant] = {
            'peak_rss_mib': read(path / 'capture/run.json')['peak_process_rss_kib']/1024,
            **{k.lower() + '_mib': last['smaps_rollup'][k]/2**20 for k in ('Rss', 'Anonymous', 'Swap')},
            'allocation_counters': last['allocation_counters']}
    assert len(binary_hashes) == 1
    result['binary_sha256'] = binary_hashes.pop()
    result['kernel_inventory_equal'] = result['kernel_inventory']['native'] == result['kernel_inventory']['litert']
    x, y = [result['timing'][v] for v in ('native', 'litert')]
    result['timing']['litert_overhead_percent'] = 100*(y['median_six_session_means']['elapsed_ms']/x['median_six_session_means']['elapsed_ms']-1)
    result['timing']['block_litert_overhead_percent'] = [100*(b['elapsed_ms']/a['elapsed_ms']-1)
        for a,b in zip(x['block_medians'], y['block_medians'])]
    save(root / 'analysis.json', result)
    if args.save:
        save(args.save, result)
    print('Kernel inventories equal:', result['kernel_inventory_equal'])
    for variant in ('native', 'litert'):
        print(variant, 'timing:', result['timing'][variant]['median_six_session_means'])
        print(variant, 'memory:', result['memory'][variant])
    print('LiteRT overhead percent:', result['timing']['litert_overhead_percent'],
          result['timing']['block_litert_overhead_percent'])


if __name__ == '__main__':
    main()
