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
"""Gate perf with the unchanged runner's measured-decode FIFO markers.

The runner does not wait for this controller. Record marker-receipt and perf
acknowledgement times so this small boundary uncertainty remains visible.
"""
import argparse
import os
from pathlib import Path
import select
import signal
import subprocess
import time

from run_profile import runner_args, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--mode', choices=['record', 'stat'], default='record')
    a = p.parse_args()
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    fixture = repo / 'tensor/examples/gemma4/native/fixtures/performance_1024_64.tsv'
    fifos, fds = {}, {}
    for name in ('trace', 'control', 'ack'):
        fifos[name] = root / (name + '.fifo')
        os.mkfifo(fifos[name], 0o600)
        fds[name] = os.open(fifos[name], os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
    argv = ['perf', a.mode, '--delay=-1',
            '--control=fifo:' + str(fifos['control']) + ',' + str(fifos['ack'])]
    if a.mode == 'record':
        argv += ['-e', 'cpu_core/cycles/u', '-F', '499', '-P', '--clockid=mono',
                 '--no-buildid-cache', '-o', str(root / 'perf.data')]
    else:
        argv += ['-e', 'cpu_core/cycles/u,cpu_core/instructions/u',
                 '-x', ',', '-o', str(root / 'stat.csv')]
    argv += ['--'] + runner_args(a.binary.resolve(), a.bundle.resolve(), fixture,
                                root / 'capture', a.repeats)
    env = dict(os.environ, LITERT_NATIVE_TRACE_FIFO=str(fifos['trace']))
    save(root / 'command.json', {'argv': argv, 'binary_sha256': sha(a.binary),
         'fixture_sha256': sha(fixture), 'bundle_manifest_sha256': sha(a.bundle / 'manifest.json'),
         'extra_environment': {'LITERT_NATIVE_TRACE_FIFO': str(fifos['trace'])},
         'timings_valid_for_benchmark': False})
    markers = []
    active = False
    buffered = b''
    start = time.monotonic()
    with (root / 'process.log').open('w') as log:
        proc = subprocess.Popen(argv, cwd=repo, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                if time.monotonic() - start > 240:
                    raise RuntimeError('Timed out waiting for runner')
                ready, _, _ = select.select([fds['trace']], [], [], 0.2)
                if not ready:
                    continue
                buffered += os.read(fds['trace'], 4096)
                while b'\n' in buffered:
                    line, buffered = buffered.split(b'\n', 1)
                    receipt = time.monotonic_ns()
                    begin = line == b'section_begin decode'
                    if line not in (b'section_begin decode', b'section_end decode') or begin == active:
                        raise RuntimeError('Unbalanced or unknown marker: ' + repr(line))
                    active = begin
                    command = b'enable\n' if begin else b'disable\n'
                    os.write(fds['control'], command)
                    ack_start = time.monotonic_ns()
                    ready, _, _ = select.select([fds['ack']], [], [], 10)
                    ack = os.read(fds['ack'], 4096) if ready else b''
                    # This perf build writes the C string terminator as well.
                    if ack not in (b'ack\n', b'ack\n\x00'):
                        raise RuntimeError('Missing perf control acknowledgement')
                    markers.append({'marker': line.decode(), 'received_ns': receipt,
                                    'command_sent_ns': ack_start, 'ack_ns': time.monotonic_ns()})
                    save(root / 'markers.json', markers)
            if proc.returncode or active or buffered or len(markers) != a.repeats * 2:
                raise RuntimeError('Failed process or incomplete decode markers')
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            raise
        finally:
            for fd in fds.values():
                os.close(fd)
            save(root / 'status.json', {'returncode': proc.returncode,
                 'wall_seconds': time.monotonic() - start, 'marker_count': len(markers)})
    print('Completed', root, flush=True)


if __name__ == '__main__':
    main()
