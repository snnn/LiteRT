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
"""Validate post-stage attribution and compare every dumped vocabulary logit."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as stats

from run_profile import save, sha


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def profile(root):
    ops = json_lines(root / 'post_operators.jsonl')
    stages = json_lines(root / 'post_stages.jsonl')
    count = len(stages[0]['wall_ns'])
    assert count > 0
    assert {s['layer'] for s in stages} == set(range(35))
    assert len(stages) == 35 and len(ops) == 1715
    assert all(len(o['samples_ns']) == count for o in ops)
    assert all(len(s['wall_ns']) == count for s in stages)
    groups, by_type, layers = defaultdict(list), defaultdict(list), defaultdict(list)
    fc_names = set()
    for o in ops:
        layers[o['layer']].append(o)
        if o['weight']:
            assert o['weight'] not in fc_names
            fc_names.add(o['weight'])
            prefix = 'model.layers.' + str(o['layer']) + '.'
            assert o['weight'].startswith(prefix)
            role = o['weight'].removeprefix(prefix).removesuffix('.weight')
            key = role + '/' + o['weight_dtype']
        else:
            key = o['name'] + '/' + o['input_dtype'] + '->' + o['output_dtype']
        groups[key].append(o)
        by_type[o['name'] + '/' + o['input_dtype'] + '->' + o['output_dtype']].append(o)
    assert len(fc_names) == 210

    def group_result(items):
        totals = [sum(o['samples_ns'][i] for o in items) / 1e6 for i in range(count)]
        return {'operator_count': len(items), 'mean_ms_per_token': stats.mean(totals),
                'median_ms_per_token': stats.median(totals),
                'source_bytes': sum(o['source_bytes'] for o in items)}

    total = [sum(s['wall_ns'][i] for s in stages) / 1e6 for i in range(count)]
    operator_total = group_result(ops)['mean_ms_per_token']
    layer_result = []
    for s in stages:
        items = layers[s['layer']]
        for i in range(count):
            assert sum(o['samples_ns'][i] for o in items) <= s['wall_ns'][i]
        layer_result.append({'layer': s['layer'],
            'stage_mean_ms': stats.mean(s['wall_ns']) / 1e6,
            'operators_mean_ms': group_result(items)['mean_ms_per_token'],
            'mlp_mean_ms': group_result([o for o in items if '.mlp.' in o['weight']])['mean_ms_per_token']})
    return {'samples_per_operator': count, 'operator_count': len(ops),
            'stage_count': len(stages), 'fc_count': len(fc_names),
            'stage_mean_ms': stats.mean(total), 'stage_median_ms': stats.median(total),
            'operator_mean_ms': operator_total,
            'outside_invoke_mean_ms': stats.mean(total) - operator_total,
            'fc': group_result([o for o in ops if o['weight']]),
            'mlp': group_result([o for o in ops if '.mlp.' in o['weight']]),
            'groups': {k: group_result(v) for k, v in groups.items()},
            'types': {k: group_result(v) for k, v in by_type.items()},
            'layers': layer_result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    a = p.parse_args()
    result = {'profile': profile(a.out / 'profile/capture'), 'runs': {}, 'correctness': {}}
    histories = None
    for variant in ('baseline', 'profile', 'off', 'correctness'):
        directory = a.out / variant
        assert json.loads((directory / 'status.json').read_text())['returncode'] == 0
        sessions = []
        for f in sorted((directory / 'capture').glob('*.run_*.json')):
            d = json.loads(f.read_text())
            assert len(d['passes']) == 65
            history = [d['prompt_token_ids'], d['forced_decode_token_ids'],
                       [(v['input_ids'], v['argmax_id']) for v in d['passes']]]
            if histories is None:
                histories = history
            assert history == histories
            assert d['timings_valid_for_benchmark'] == (variant in ('baseline', 'off'))
            sessions.append({k: stats.mean(v[k] for v in d['passes'][1:]) for k in
                            ('elapsed_ms', 'post_attention_run_ms', 'forward_ms',
                             'projection_run_ms', 'attention_run_ms')})
        assert len(sessions) == (1 if variant == 'correctness' else 3)
        result['runs'][variant] = {'sessions': sessions,
            'median_session': {k: stats.median(s[k] for s in sessions) for k in sessions[0]},
            'command': json.loads((directory / 'command.json').read_text())}
    files = sorted((a.out / 'correctness/capture').glob('*.f32'))
    assert len(files) == 65
    total = 0
    for f in files:
        ref = a.reference / f.name.replace('performance_1024_64', 'check_1024_64')
        assert f.stat().st_size == 262144 * 4
        assert f.read_bytes() == ref.read_bytes(), f
        total += f.stat().st_size
    result['correctness'] = {'full_logit_files': len(files), 'bytes_bit_identical': total,
                             'float32_values_bit_identical': total // 4,
                             'reference': str(a.reference), 'all_histories_and_argmax_match': True}
    result['build'] = json.loads((a.out / 'commands.json').read_text())
    result['build']['commands_sha256'] = sha(a.out / 'commands.json')
    save(a.out / 'analysis.json', result)
    print(json.dumps({k: v for k, v in result['profile'].items() if k not in ('groups', 'types', 'layers')}, indent=2))
    print(result['correctness'])


if __name__ == '__main__':
    main()
