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
"""Validate the phone campaign and summarize timing, operators, and owners."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as stats

from analyze import json_lines, profile
from phone_campaign import DEVICES
from run_profile import save


def read(path):
    return json.loads(path.read_text())


def stage_profile(directory, prefix, expected_layers, expected_ops):
    ops = json_lines(directory / (prefix + '_operators.jsonl'))
    stages = json_lines(directory / (prefix + '_stages.jsonl'))
    assert len(stages) == expected_layers and len(ops) == expected_ops
    count = len(stages[0]['wall_ns'])
    assert count == 64
    groups, indices = defaultdict(float), defaultdict(float)
    for layer in stages:
        items = [o for o in ops if o['layer'] == layer['layer']]
        for i in range(count):
            assert sum(o['samples_ns'][i] for o in items) <= layer['wall_ns'][i]
    for op in ops:
        assert len(op['samples_ns']) == count
        value = stats.mean(op['samples_ns']) / 1e6
        groups[op['name'] + '/' + op['input_dtype'] + '->' + op['output_dtype']] += value
        indices[str(op['op_index'])] += value
    total = sum(stats.mean(s['wall_ns']) / 1e6 for s in stages)
    return {'stage_mean_ms': total, 'outside_invoke_mean_ms': total - sum(groups.values()),
            'samples_per_operator': count, 'stage_count': len(stages), 'operator_count': len(ops),
            'groups_mean_ms': dict(groups), 'op_index_mean_ms': dict(indices)}


def analyze(root):
    assert read(root / 'completed.json')['complete']
    identity = read(root / 'identity.json')
    assert identity['coefficient_files_checked'] == 1394
    records = read(root / 'capture-summary.json')
    history, result = None, {'identity': identity, 'threads': {}, 'memory': {}}
    timing = []
    for record in records:
        d = root / record['name']
        status = read(d / 'status.json')
        assert status['returncode'] == status['pull_returncode'] == 0
        diagnostic = record['diagnostic']
        assert status['timings_valid_for_benchmark'] == (not diagnostic)
        sessions = []
        for path in sorted((d / 'capture').glob('*.run_*.json')):
            run = read(path)
            assert len(run['passes']) == 65
            current = [run['prompt_token_ids'], run['forced_decode_token_ids'],
                       [(v['input_ids'], v['argmax_id']) for v in run['passes']]]
            if history is None:
                history = current
            assert current == history
            sessions.append({
                **{k: stats.mean(p[k] for p in run['passes'][1:]) for k in
                   ('elapsed_ms', 'post_attention_run_ms', 'attention_run_ms', 'projection_run_ms')},
                'first_logit_ms': run['passes'][0]['elapsed_ms']})
        if not diagnostic:
            assert len(sessions) == 3 and record['name'].startswith('timing-')
            timing.append({**record, 'sessions': sessions})
    for config in DEVICES[root.name]['configs']:
        runs = [r for r in timing if r['config'][0] == config[0]]
        assert len(runs) == 2
        sessions = [s for r in runs for s in r['sessions']]
        result['threads'][config[0]] = {
            'threads': config[1], 'affinity_mask_hex': config[2],
            'median_six_session_means_ms': {k: stats.median(s[k] for s in sessions) for k in sessions[0]},
            'session_decode_range_ms': [min(s['elapsed_ms'] for s in sessions), max(s['elapsed_ms'] for s in sessions)],
            'block_medians_ms': [r['median_session'] for r in runs]}
    result['post'] = profile(root / 'profile/capture')
    result['attention'] = stage_profile(root / 'attention-head-profile/capture', 'attention', 35, 210)
    result['head'] = stage_profile(root / 'attention-head-profile/capture', 'head', 1, 9)
    for variant in ('baseline', 'release'):
        d = root / ('memory-' + variant) / 'capture'
        rows = json_lines(d / 'memory.jsonl')
        last = [r for r in rows if r['phase'] == 'after_last_decode'][-1]
        assert all(last[k] for k in ('smaps_rollup_available', 'smaps_available', 'peak_rss_available'))
        peak = read(d / 'run.json')['peak_process_rss_kib'] * 1024
        assert peak >= max(r['peak_rss_bytes'] for r in rows)
        result['memory'][variant] = {
            'peak_rss_mib': peak / 2**20,
            **{k.lower() + '_mib': last['smaps_rollup'][k] / 2**20 for k in ('Rss', 'Anonymous', 'Swap')},
            'bundle_rss_mib': last['mapping_groups']['native_bundle_files']['Rss'] / 2**20,
            'allocation_counters': last['allocation_counters']}
    baseline, released = (result['memory'][k] for k in ('baseline', 'release'))
    result['memory']['saving_mib'] = {k: baseline[k] - released[k] for k in
        ('peak_rss_mib', 'rss_mib', 'anonymous_mib', 'bundle_rss_mib')}
    a, b = baseline['allocation_counters'], released['allocation_counters']
    assert a['static_int2_compact_bytes'] == 270 * 2**20 and b['static_int2_compact_bytes'] == 0
    assert a['graph_constant_capacity_bytes'] == 96 * 2**20 and b['graph_constant_capacity_bytes'] == 0
    for key in ('packed_weight_bytes', 'packed_weight_capacity_bytes', 'owned_external_output_bytes',
                'logical_kv_bytes', 'unique_workspace_bytes', 'unique_workspace_count'):
        assert a[key] == b[key]
    result['correctness'] = read(root / 'correctness.json')
    result['followup_correctness'] = read(root / 'followup-correctness.json')
    candidates = [x['capture'] for x in result['correctness']['results']] + ['attention-head-profile']
    if (root / 'cycles-correctness.json').exists():
        result['cycles_correctness'] = read(root / 'cycles-correctness.json')
        result['cycles'] = read(root / 'operator-cycles/analysis.json')
        candidates.append('operator-cycles')
    reference = root / 'correctness-baseline/capture'
    for name in candidates:
        files = sorted((root / name / 'capture').glob('*.f32'))
        assert len(files) == 65
        for path in files:
            assert path.stat().st_size == 262144 * 4
            assert path.read_bytes() == (reference / path.name).read_bytes()
    telemetry = [x for r in records for x in read(root / r['name'] / 'telemetry.json')]
    gates = [read(root / r['name'] / 'start-gate.json') for r in records]
    assert all(g[-1]['eligible'] and g[-2]['eligible'] for g in gates)
    result['telemetry'] = {
        'all_starts_passed_gate': True,
        'max_sampled_cpu_c': max(x['cpu_c'] for x in telemetry),
        'max_sampled_skin_c': max(x['skin_c'] for x in telemetry),
        'cpu_cooling_exposed': sorted(telemetry[0]['cpu_cooling_states']),
        'any_sampled_cpu_cooling': any(any(x['cpu_cooling_states'].values()) for x in telemetry),
        'fixed_clocks': False, 'energy_measured': False}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--save', type=Path)
    a = p.parse_args()
    result = {'raw_artifacts': str(a.out), 'mtp_in_scope': False,
              'devices': {name: analyze(a.out / name) for name in DEVICES}}
    save(a.out / 'analysis.json', result)
    if a.save:
        save(a.save, result)
    for name, d in result['devices'].items():
        print(name, 'decode_ms', {k: v['median_six_session_means_ms']['elapsed_ms'] for k, v in d['threads'].items()})
        print(name, 'memory_savings', d['memory']['saving_mib'])
