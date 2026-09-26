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
"""Report sampled user cycles within the interior of measured decode sections."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import subprocess

from run_profile import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--stat', type=Path, required=True)
    a = p.parse_args()
    root = a.record
    commands = {
        'report.txt': ['perf', 'report', '--stdio', '--no-children', '--percent-limit', '0', '-i', str(root / 'perf.data')],
        'samples.txt': ['perf', 'script', '--ns', '-F', 'time,period,ip,sym', '-i', str(root / 'perf.data')],
        'thread-main.txt': ['perf', 'annotate', '--stdio', '--symbol', 'thread_main', '-i', str(root / 'perf.data')],
    }
    for filename, command in commands.items():
        with (root / filename).open('w') as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.PIPE)
    assert 'Total Lost Samples: 0' in (root / 'report.txt').read_text()
    markers = json.loads((root / 'markers.json').read_text())
    assert len(markers) == 6
    # Start after enable ACK and stop 10 ms before end-marker receipt. The
    # asynchronous runner does not pause while perf processes control messages.
    windows = [(markers[i]['ack_ns'], markers[i + 1]['received_ns'] - 10000000)
               for i in range(0, len(markers), 2)]
    all_groups, kept_groups = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    parsed = kept = 0
    for line in (root / 'samples.txt').read_text().splitlines():
        match = re.fullmatch(r'\s*(\d+)\.(\d{9}):\s+(\d+)\s+([0-9a-f]+)\s+(.*)', line)
        assert match, repr(line)
        sec, ns, period, ip, symbol = match.groups()
        timestamp = int(sec) * 1000000000 + int(ns)
        period = int(period)
        parsed += 1
        all_groups[symbol][0] += 1
        all_groups[symbol][1] += period
        if any(begin <= timestamp <= end for begin, end in windows):
            kept += 1
            kept_groups[symbol][0] += 1
            kept_groups[symbol][1] += period
    result = {'all_samples': parsed, 'retained_samples': kept,
              'excluded_samples': parsed - kept, 'lost_samples': 0,
              'windows_ns': windows,
              'retained_seconds': sum(end - start for start, end in windows) / 1e9,
              'control_delay_ms': [(m['ack_ns'] - m['received_ns']) / 1e6 for m in markers]}
    for name, groups in [('all', all_groups), ('trimmed', kept_groups)]:
        total = sum(v[1] for v in groups.values())
        assert total > 0
        result[name] = {'total_period': total, 'symbols': [
            {'name': k, 'samples': v[0], 'period': v[1], 'percent': 100 * v[1] / total}
            for k, v in sorted(groups.items(), key=lambda x: -x[1][1])]}
    counters = {}
    for row in csv.reader((a.stat / 'stat.csv').read_text().splitlines()):
        if not row or row[0].startswith('#'):
            continue
        assert row[0].isdigit() and float(row[4]) >= 99.9
        counters[row[2]] = {'count': int(row[0]), 'runtime_ns': int(row[3]),
                            'running_percent': float(row[4])}
    result['stat'] = counters
    result['stat_ipc'] = counters['cpu_core/instructions/u']['count'] / counters['cpu_core/cycles/u']['count']
    reference = None
    for directory in (root, a.stat):
        assert json.loads((directory / 'status.json').read_text())['returncode'] == 0
        captures = sorted((directory / 'capture').glob('*.run_*.json'))
        assert len(captures) == 3
        for path in captures:
            d = json.loads(path.read_text())
            assert d['decode_trace_enabled'] and d['measured_decode_calls'] == 64
            history = [d['prompt_token_ids'], d['forced_decode_token_ids'],
                       [(v['input_ids'], v['argmax_id']) for v in d['passes']]]
            if reference is None:
                reference = history
            assert reference == history
    save(root / 'analysis.json', result)
    print('Samples:', parsed, 'retained:', kept, 'IPC:', result['stat_ipc'])


if __name__ == '__main__':
    main()
