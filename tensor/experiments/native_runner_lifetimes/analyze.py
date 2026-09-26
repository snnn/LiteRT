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
"""Validate results before summarizing memory or paired timing measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def runs(directory):
    assert read(directory / 'status.json')['returncode'] == 0, directory
    records = {}
    for path in sorted((directory / 'capture').glob('*.json')):
        if path.name == 'run.json':
            continue
        data = read(path)
        assert data['kv_capacity'] == 8448 and data['num_threads'] == 4
        assert data['reuse_runtimes'] and data['share_workspace'] and data['preserve_static_int2']
        assert data['static_int2_tensor_count'] == data['static_int2_operator_count'] == 60
        assert len(data['passes']) == 1 + len(data['forced_decode_token_ids'])
        for i, p in enumerate(data['passes']):
            assert p['context_length_after'] == len(data['prompt_token_ids']) + i
            expected = data['prompt_token_ids'] if i == 0 else [data['forced_decode_token_ids'][i-1]]
            assert p['input_ids'] == expected
        records[path.name] = data
    assert records
    return records


def identity(data):
    return (data['prompt_token_ids'], data['forced_decode_token_ids'],
            [p['argmax_id'] for p in data['passes']])


def correctness(root, suite, variants, suffix):
    baseline = root / f'{suite}-baseline{suffix}'
    ref_runs = runs(baseline)
    files = sorted(p for p in (baseline / 'capture').iterdir() if p.suffix in ('.f32', '.i8'))
    reference = {p.name: sha(p) for p in files}
    results = {'baseline': {'files': len(files), 'logit_vectors': sum(p.suffix == '.f32' for p in files),
                           'bytes': sum(p.stat().st_size for p in files)}, 'variants': {}}
    for variant in variants:
        directory = root / f'{suite}-{variant}{suffix}'
        candidate_runs = runs(directory)
        assert candidate_runs.keys() == ref_runs.keys()
        for name, data in candidate_runs.items():
            assert identity(data) == identity(ref_runs[name]), (variant, name)
        candidate = {p.name: sha(p) for p in (directory / 'capture').iterdir() if p.suffix in ('.f32', '.i8')}
        assert candidate == reference, variant
        results['variants'][variant] = {'bit_identical_files': len(candidate),
                                        'argmaxes_and_token_histories_match': True}
    return results


def memory(root, variants):
    result, references = {}, {}
    for variant in variants:
        directory = root / f'memory-{variant}'
        records = runs(directory)
        assert all(d['memory_report'] for d in records.values())
        for name, data in records.items():
            if name in references:
                assert identity(data) == references[name], (variant, name)
            else:
                references[name] = identity(data)
        rows = [json.loads(x) for x in (directory / 'capture/memory.jsonl').read_text().splitlines()]
        assert all(not row['timings_valid_for_benchmark'] for row in rows)
        assert all(row['smaps_rollup'].get('Swap', 0) == 0 for row in rows)
        last = [r for r in rows if r['phase'] == 'after_last_decode'][-1]
        result[variant] = {'peak_rss_mib': read(directory / 'status.json')['peak_rss_bytes'] / 2**20,
                           'last_decode_rss_mib': last['smaps_rollup']['Rss'] / 2**20,
                           'last_decode_anonymous_mib': last['smaps_rollup']['Anonymous'] / 2**20,
                           'allocations': last['allocation_counters'],
                           'phases': [{k: r[k] for k in ('phase', 'smaps_rollup', 'allocation_counters')}
                                      for r in rows if r['phase'] in ('after_pack', 'after_packed_source_release')]}
    return result


def timing(root, variants, blocks):
    samples, prediction_ref = {}, {}
    metrics = ('prefill_ms', 'decode_ms', 'prefill_stage_run_ms',
               'prefill_outside_stage_run_ms', 'decode_stage_run_ms',
               'decode_attention_ms', 'decode_projection_ms', 'decode_post_ms')
    for block in range(blocks):
        for variant in variants:
            directory = root / f'timing-{variant}-b{block}'
            assert read(directory / 'command.json')['timings_valid_for_benchmark']
            records = runs(directory)
            per_case = {}
            for data in records.values():
                assert not data['memory_report'] and not data['decode_trace_enabled']
                assert all(p['logits_file'] is None for p in data['passes'])
                case = data['case_id']
                if case in prediction_ref:
                    assert identity(data) == prediction_ref[case]
                else:
                    prediction_ref[case] = identity(data)
                if data['warmup']:
                    continue
                assert not data['compiled_this_run']
                per_case.setdefault(case, []).append({
                    'prefill_ms': data['passes'][0]['elapsed_ms'],
                    'decode_ms': statistics.mean(p['elapsed_ms'] for p in data['passes'][1:]),
                    'prefill_stage_run_ms': data['passes'][0]['forward_ms'],
                    'prefill_outside_stage_run_ms': data['passes'][0]['elapsed_ms'] - data['passes'][0]['forward_ms'],
                    'decode_stage_run_ms': statistics.mean(p['forward_ms'] for p in data['passes'][1:]),
                    'decode_attention_ms': statistics.mean(p['attention_run_ms'] for p in data['passes'][1:]),
                    'decode_projection_ms': statistics.mean(p['projection_run_ms'] for p in data['passes'][1:]),
                    'decode_post_ms': statistics.mean(p['post_attention_run_ms'] for p in data['passes'][1:])})
            for case, rows in per_case.items():
                assert len(rows) == 3
                samples.setdefault(case, {}).setdefault(variant, []).append({
                    metric: statistics.median(row[metric] for row in rows)
                    for metric in metrics})
    summary = {}
    for case, data in samples.items():
        summary[case] = {}
        for variant, block_rows in data.items():
            summaries = {}
            for metric in metrics:
                values = [row[metric] for row in block_rows]
                base = [row[metric] for row in data['baseline']]
                savings = [100 * (1 - v / b) for v, b in zip(values, base)]
                summaries[metric] = {'median': statistics.median(values),
                                   'block_values': values, 'paired_reduction_percent': savings,
                                   'median_paired_reduction_percent': statistics.median(savings)}
            summary[case][variant] = summaries
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=['smoke', 'correctness', 'memory', 'timing'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=['baseline', 'release', 'reshape', 'combined'])
    parser.add_argument('--suffix', default='')
    parser.add_argument('--blocks', type=int, default=3)
    args = parser.parse_args()
    if args.suite in ('smoke', 'correctness'):
        result = correctness(args.out, args.suite, [v for v in args.variants if v != 'baseline'], args.suffix)
    elif args.suite == 'memory':
        result = memory(args.out, args.variants)
    else:
        result = timing(args.out, args.variants, args.blocks)
    path = args.out / (args.suite + args.suffix + '-analysis.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(path)
    if args.suite in ('smoke', 'correctness'):
        print(json.dumps(result))
    else:
        print('All workload and completion assertions passed.')


if __name__ == '__main__':
    main()
