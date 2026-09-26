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
"""Native-only phone thread, operator, lifetime, and numerical controls.

No MTP, root, governor changes, or thermal overrides. Each device serializes
its own work; independent phones can run concurrently.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time

from run_profile import save, sha

DEVICES = {
    'tecno': {'serial': '144512553T000099',
              'configs': [('t4', 4, 'f0'), ('t2', 2, 'c0'), ('all8', 8, 'ff')]},
    'pixel8': {'serial': '3A111FDJH005A0',
               'configs': [('t4', 4, '1e0'), ('t2', 2, '180'),
                           ('fast5', 5, '1f0'), ('all9', 9, '1ff')]},
}
OLD = '/data/local/tmp/gemma4-qat-q4_0-perf-20260914'
REMOTE = '/data/local/tmp/gemma4-native-phones-20260923'
EXPECTED_MANIFEST = '3cdd41d803c05a4b29729a9892451122987675867ae1841709b2b480ddd826bf'
EXPECTED_FIXTURE = '01009f2da16a6c32e7288323029d019b49e1507022fd711f471c8001824b4eef'


class Phone:
    def __init__(self, args, name, resume=False):
        self.args, self.name = args, name
        self.serial = DEVICES[name]['serial']
        self.root = args.out / name
        self.root.mkdir(parents=True, exist_ok=resume)
        self.last_end = 0
        self.history = None
        self.records = json.loads((self.root / 'capture-summary.json').read_text()) if resume else []
        if resume:
            reference = json.loads((self.root / 'correctness-baseline/capture/performance_1024_64.run_000.json').read_text())
            self.history = [reference['prompt_token_ids'], reference['forced_decode_token_ids'],
                            [(v['input_ids'], v['argmax_id']) for v in reference['passes']]]

    def adb(self, *argv, **kwargs):
        return subprocess.run(['adb', '-s', self.serial, *map(str, argv)], **kwargs)

    def shell(self, argv):
        command = argv if isinstance(argv, str) else shlex.join(map(str, argv))
        return self.adb('shell', command, capture_output=True, text=True, check=True, timeout=45).stdout

    def thermal(self):
        raw = self.shell(['dumpsys', 'thermalservice'])
        live = raw.split('Current temperatures from HAL:')[1].split('Current cooling devices from HAL:')[0]
        values = [(float(v), int(t), n) for v, t, n in re.findall(
            r'Temperature\{mValue=([-\d.]+), mType=([-\d]+), mName=([^,]+), mStatus=\d+\}', live)]
        cpu = max(v for v, t, _ in values if t == 0)
        skin = max(v for v, t, _ in values if t == 3)
        soc = next(v for v, t, n in values if t == 13 or n == 'soc_therm')
        cooling = {n: int(v) for v, n in re.findall(
            r'CoolingDevice\{mValue=(\d+), mType=2, mName=([^}]+)\}', raw)}
        if self.name == 'pixel8':
            assert len(cooling) == 3
        return {'epoch_seconds': time.time(), 'cpu_c': cpu, 'skin_c': skin,
                'soc_c': soc, 'cpu_cooling_states': cooling,
                'eligible': cpu <= 45 and skin <= 34 and soc <= 39 and not any(cooling.values()),
                'raw': raw,
                'policy_current_khz': self.shell('cat /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq').split()}

    def gate(self, directory):
        readings, consecutive = [], 0
        started = time.monotonic()
        while time.monotonic() - started < 600:
            record = self.thermal()
            readings.append(record)
            save(directory / 'start-gate.json', readings)
            consecutive = consecutive + 1 if record['eligible'] else 0
            if consecutive >= 2 and time.monotonic() - self.last_end >= 15:
                return
            if len(readings) % 4 == 0:
                print(self.name, 'cooling', record['cpu_c'], record['skin_c'], flush=True)
            time.sleep(5)
        raise RuntimeError(self.name + ': thermal start gate not reached')

    def prepare(self):
        self.shell(['mkdir', '-p', REMOTE])
        identities = {}
        for variant in ('baseline', 'probe', 'profile'):
            source = self.args.out / 'build' / variant / 'runner'
            remote = REMOTE + '/' + variant
            self.adb('push', source, remote, check=True, capture_output=True)
            self.shell(['chmod', '755', remote])
            actual = self.shell(['sha256sum', remote]).split()[0]
            assert actual == sha(source)
            identities[variant] = actual
        expected = self.args.expected_hashes.resolve()
        self.adb('push', expected, REMOTE + '/bundle.expected.sha256', check=True, capture_output=True)
        command = 'cd ' + shlex.quote(OLD + '/matched-bundle') + ' && sha256sum -c ' + shlex.quote(REMOTE + '/bundle.expected.sha256')
        checked = self.adb('shell', command, capture_output=True, text=True, timeout=180)
        (self.root / 'bundle-validation.log').write_text(checked.stdout + checked.stderr)
        assert checked.returncode == 0 and len(checked.stdout.splitlines()) == 1394
        assert all(line.endswith(': OK') for line in checked.stdout.splitlines())
        assert self.shell(['sha256sum', OLD + '/matched-bundle/manifest.json']).split()[0] == EXPECTED_MANIFEST
        assert self.shell(['sha256sum', OLD + '/fixtures/performance_1024_64.tsv']).split()[0] == EXPECTED_FIXTURE
        identities.update(serial=self.serial, manifest=EXPECTED_MANIFEST,
                          fixture=EXPECTED_FIXTURE, coefficient_files_checked=1394,
                          expected_hashes_sha256=sha(expected))
        save(self.root / 'identity.json', identities)
        save(self.root / 'device.json', {
            'fingerprint': self.shell(['getprop', 'ro.build.fingerprint']),
            'cpuinfo': self.shell(['cat', '/proc/cpuinfo']),
            'policies': self.shell('cat /sys/devices/system/cpu/cpufreq/policy*/related_cpus /sys/devices/system/cpu/cpufreq/policy*/cpuinfo_max_freq'),
            'meminfo': self.shell(['cat', '/proc/meminfo'])})
        print(self.name, 'prepared and verified all bundle files', flush=True)

    def capture(self, name, config, variant='baseline', profile=False, memory=False,
                dump=False, release=False, reshape=False, repeats=3, perf=False):
        directory = self.root / name
        directory.mkdir(exist_ok=False)
        self.gate(directory)
        label, threads, mask = config
        remote = REMOTE + '/' + self.name + '-' + name
        self.shell(['test', '!', '-e', remote])
        argv = ['taskset', mask, REMOTE + '/' + variant,
                '--bundle_dir=' + OLD + '/matched-bundle',
                '--cases_file=' + OLD + '/fixtures/performance_1024_64.tsv',
                '--output_dir=' + remote, '--num_threads=' + str(threads),
                '--cache_capacity=8448', '--prefill_chunk_rows=128', '--kv_alignment=32',
                '--reuse_runtimes=true', '--warmup_runs=1', '--measured_runs=' + str(repeats),
                '--share_workspace=true', '--preserve_static_int2=true',
                '--memory_report=' + str(memory).lower(), '--dump_full_logits=' + str(dump).lower()]
        if variant in ('profile', 'followup', 'cycles'):
            argv += ['--profile_post_ops=' + str(profile).lower()]
        if variant == 'probe':
            argv += ['--experiment_release_packed_sources=' + str(release).lower(),
                     '--experiment_cache_stage_shapes=' + str(reshape).lower()]
        diagnostic = profile or memory or dump or perf
        perf_remote = remote + '.perf.data'
        if perf:
            self.shell(['test', '!', '-e', perf_remote])
            argv = ['/system/bin/simpleperf', 'record', '-e', 'cpu-cycles:u',
                    '-f', '499', '--clockid', 'monotonic', '--use-cmd-exit-code',
                    '-o', perf_remote, '--', *argv]
        save(directory / 'command.json', {'serial': self.serial, 'argv': argv,
             'timings_valid_for_benchmark': not diagnostic})
        (directory / 'meminfo-before.txt').write_text(self.shell(['cat', '/proc/meminfo']))
        samples = [self.thermal()]
        start = time.monotonic()
        print(self.name, 'START', name, flush=True)
        with (directory / 'process.log').open('w') as log:
            proc = subprocess.Popen(['adb', '-s', self.serial, 'shell', shlex.join(argv)],
                                    stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if time.monotonic() - start > 600:
                    raise RuntimeError(self.name + ' remote capture exceeded deadline: ' + remote)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    samples.append(self.thermal())
                    save(directory / 'telemetry.json', samples)
        self.last_end = time.monotonic()
        samples.append(self.thermal())
        save(directory / 'telemetry.json', samples)
        (directory / 'meminfo-after.txt').write_text(self.shell(['cat', '/proc/meminfo']))
        pulled = self.adb('pull', remote + '/.', directory / 'capture', capture_output=True, text=True)
        (directory / 'pull.log').write_text(pulled.stdout + pulled.stderr)
        save(directory / 'status.json', {'returncode': proc.returncode,
             'pull_returncode': pulled.returncode, 'wall_seconds': time.monotonic() - start,
             'timings_valid_for_benchmark': not diagnostic})
        if proc.returncode or pulled.returncode:
            raise RuntimeError((directory / 'process.log').read_text()[-6000:])
        if perf:
            self.adb('pull', perf_remote, directory / 'perf.data',
                     check=True, capture_output=True)
        sessions = []
        for path in sorted((directory / 'capture').glob('*.run_*.json')):
            d = json.loads(path.read_text())
            assert len(d['passes']) == 65 and d['num_threads'] == threads
            history = [d['prompt_token_ids'], d['forced_decode_token_ids'],
                       [(v['input_ids'], v['argmax_id']) for v in d['passes']]]
            if self.history is None:
                self.history = history
            assert history == self.history, path
            row = {k: statistics.mean(v[k] for v in d['passes'][1:]) for k in
                   ('elapsed_ms', 'post_attention_run_ms', 'attention_run_ms', 'projection_run_ms', 'forward_ms')}
            row['first_logit_ms'] = d['passes'][0]['elapsed_ms']
            sessions.append(row)
        assert len(sessions) == repeats
        item = {'name': name, 'config': config, 'diagnostic': diagnostic, 'sessions': sessions,
                'median_session': {k: statistics.median(v[k] for v in sessions) for k in sessions[0]},
                'max_sampled_cpu_c': max(v['cpu_c'] for v in samples),
                'max_sampled_skin_c': max(v['skin_c'] for v in samples),
                'observed_cpu_cooling': any(any(v['cpu_cooling_states'].values()) for v in samples)}
        self.records.append(item)
        save(self.root / 'capture-summary.json', self.records)
        print(self.name, 'DONE', name, item['median_session'], 'cooling=', item['observed_cpu_cooling'], flush=True)
        return directory / 'capture'

    def run(self):
        self.prepare()
        configs = DEVICES[self.name]['configs']
        for block, order in enumerate((configs, list(reversed(configs)))):
            for config in order:
                self.capture(f'timing-b{block}-{config[0]}', config)
        medians = {c[0]: statistics.median(v['elapsed_ms'] for row in self.records
                    if row['config'][0] == c[0] for v in row['sessions']) for c in configs}
        best = min(configs, key=lambda c: medians[c[0]])
        save(self.root / 'thread-summary.json', {'median_decode_ms': medians, 'fastest_observed': best})
        baseline = configs[0]
        self.capture('profile', baseline, variant='profile', profile=True)
        reference = self.capture('correctness-baseline', baseline, dump=True, repeats=1)
        targets = [self.capture('correctness-profile', baseline, variant='profile', profile=True, dump=True, repeats=1),
                   self.capture('correctness-combined', baseline, variant='probe', release=True,
                                reshape=True, dump=True, repeats=1)]
        if best != baseline:
            targets.append(self.capture('correctness-best', best, dump=True, repeats=1))
        valid = []
        for target in targets:
            files = sorted(target.glob('*.f32'))
            assert len(files) == 65
            for path in files:
                assert path.stat().st_size == 262144 * 4
                assert path.read_bytes() == (reference / path.name).read_bytes(), path
            valid.append({'capture': target.parent.name, 'vectors_bit_identical': len(files),
                          'bytes_bit_identical': sum(p.stat().st_size for p in files)})
        save(self.root / 'correctness.json', {'reference': str(reference), 'results': valid})
        self.capture('memory-baseline', baseline, memory=True, repeats=1)
        self.capture('memory-release', baseline, variant='probe', release=True, memory=True, repeats=1)
        save(self.root / 'completed.json', {'complete': True, 'captures': len(self.records),
             'all_measured_histories_and_predictions_match': True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--expected-hashes', type=Path, required=True)
    p.add_argument('--devices', nargs='+', choices=list(DEVICES), default=list(DEVICES))
    a = p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.devices)) as pool:
        futures = [pool.submit(Phone(a, name).run) for name in a.devices]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == '__main__':
    main()
