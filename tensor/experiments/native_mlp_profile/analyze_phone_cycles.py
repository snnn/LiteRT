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
"""Attribute Android CPU samples to measured XNNPACK operator intervals.

Both clocks are CLOCK_MONOTONIC. Startup, warmup, prefill, and time outside
these invokes are excluded. Period-weighted CPU samples are not wall times.
"""
import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

from analyze import json_lines
from phone_campaign import DEVICES, REMOTE
from run_profile import save


def role(stage, op):
    if stage == 'post':
        if '.mlp.' in op['weight']:
            return 'post/mlp/' + op['weight_dtype']
        return 'post/other_fc/' + op['weight_dtype'] if op['weight'] else 'post/other'
    if stage == 'attention':
        return {0: 'attention/convert', 1: 'attention/convert',
                2: 'attention/qk_bmm', 3: 'attention/mask_add',
                4: 'attention/softmax', 5: 'attention/av_bmm'}[op['op_index']]
    return 'head/fc' if op['weight'] else 'head/other'


def analyze(root, simpleperf_dir):
    sys.path.insert(0, str(simpleperf_dir))
    from simpleperf_report_lib import ReportLib
    capture = root / 'operator-cycles'
    intervals = []
    for stage, expected in [('post', 35), ('attention', 35), ('head', 1)]:
        stages = json_lines(capture / 'capture' / (stage + '_stages.jsonl'))
        ops = json_lines(capture / 'capture' / (stage + '_operators.jsonl'))
        assert len(stages) == expected
        for layer in stages:
            items = [o for o in ops if o['layer'] == layer['layer']]
            assert len(layer['invoke_start_ns']) == len(layer['wall_ns']) == 192
            assert all(len(o['samples_ns']) == 192 for o in items)
            for i, start in enumerate(layer['invoke_start_ns']):
                cursor = start
                for op in items:
                    end = cursor + op['samples_ns'][i]
                    intervals.append((cursor, end, role(stage, op)))
                    cursor = end
                assert cursor - start <= layer['wall_ns'][i]
    intervals.sort()
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    starts = [i[0] for i in intervals]
    symfs = capture / 'symfs'
    binary = symfs / (REMOTE + '/cycles').lstrip('/')
    binary.parent.mkdir(parents=True, exist_ok=True)
    if not binary.exists():
        binary.symlink_to(root.parent / 'build/followup-cycles/runner')
    lib = ReportLib()
    lib.SetSymfs(str(symfs))
    lib.SetRecordFile(str(capture / 'perf.data'))
    meta = lib.MetaInfo()
    command = lib.GetRecordCmd()
    assert '--clockid monotonic' in command and meta['clockid'] == 'monotonic'
    groups = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    parsed = retained = excluded_kernel = excluded_kernel_in_window = 0
    pids, cpus = set(), set()
    while (sample := lib.GetNextSample()) is not None:
        parsed += 1
        assert lib.GetEventOfCurrentSample().name == 'cpu-cycles:u'
        index = bisect.bisect_right(starts, sample.time) - 1
        inside = index >= 0 and sample.time < intervals[index][1]
        # Some ARM records report kernel-mode sample IPs even with :u. Keep
        # those out of the user-code attribution and record their count.
        if sample.in_kernel:
            excluded_kernel += 1
            excluded_kernel_in_window += int(inside)
            continue
        if not inside:
            continue
        retained += 1
        pids.add(sample.pid)
        cpus.add(sample.cpu)
        symbol = lib.GetSymbolOfCurrentSample().symbol_name
        group = groups[intervals[index][2]]
        group[symbol][0] += 1
        group[symbol][1] += sample.period
    lib.Close()
    assert len(pids) == 1 and retained > 1000
    result = {'parsed_samples': parsed, 'retained_samples': retained,
              'excluded_samples': parsed - retained, 'pids': sorted(pids),
              'excluded_kernel_flagged_samples': excluded_kernel,
              'excluded_kernel_flagged_inside_windows': excluded_kernel_in_window,
              'cpus': sorted(cpus), 'record_command': command, 'meta': meta,
              'intervals': len(intervals), 'clock': 'CLOCK_MONOTONIC',
              'interpretation': 'Period-weighted user cycles, not wall-time shares.',
              'groups': {}}
    for label, symbols in groups.items():
        total = sum(v[1] for v in symbols.values())
        result['groups'][label] = {
            'samples': sum(v[0] for v in symbols.values()), 'period': total,
            'symbols': [{'name': k, 'samples': v[0], 'period': v[1],
                         'percent': 100 * v[1] / total}
                        for k, v in sorted(symbols.items(), key=lambda x: -x[1][1])]}
    log = (capture / 'process.log').read_text()
    lost = re.search(r'Samples lost:\s*(\d+)', log)
    assert lost and int(lost[1]) == 0, log[-2000:]
    result['lost_samples'] = int(lost[1])
    assert json.loads((capture / 'status.json').read_text())['returncode'] == 0
    assert json.loads((root / 'cycles-correctness.json').read_text())['vectors_bit_identical'] == 65
    save(capture / 'analysis.json', result)
    print(root.name, 'retained', retained, 'of', parsed, 'samples', flush=True)
    for key in ('post/mlp/QCINT4', 'post/mlp/QCINT2', 'attention/qk_bmm', 'attention/av_bmm', 'head/fc'):
        print(key, [(s['name'], round(s['percent'], 2)) for s in result['groups'][key]['symbols'][:8]])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--simpleperf-dir', type=Path, required=True)
    a = p.parse_args()
    for name in DEVICES:
        analyze(a.out / name, a.simpleperf_dir)
