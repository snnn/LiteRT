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
"""Run attention/head diagnostics after each phone's main campaign finishes."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time

from phone_campaign import Phone, DEVICES, REMOTE
from run_profile import save, sha


def run(args, name):
    deadline = time.monotonic() + 1800
    while not (args.out / name / 'completed.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError(name + ': main campaign did not finish')
        time.sleep(5)
    phone = Phone(args, name, resume=True)
    phone.last_end = time.monotonic()
    source = args.out / 'build/followup/runner'
    binary = REMOTE + '/followup'
    phone.adb('push', source, binary, capture_output=True, check=True)
    phone.shell(['chmod', '755', binary])
    digest = phone.shell(['sha256sum', binary]).split()[0]
    assert digest == sha(source)
    save(phone.root / 'followup-deployment.json', {'binary_sha256': digest})
    target = phone.capture('attention-head-profile', DEVICES[name]['configs'][0],
                           variant='followup', profile=True, dump=True, repeats=1)
    reference = phone.root / 'correctness-baseline/capture'
    files = sorted(target.glob('*.f32'))
    assert len(files) == 65
    for path in files:
        assert path.stat().st_size == 262144 * 4
        assert path.read_bytes() == (reference / path.name).read_bytes(), path
    save(phone.root / 'followup-correctness.json', {'vectors_bit_identical': len(files),
         'bytes_bit_identical': sum(p.stat().st_size for p in files), 'reference': str(reference)})
    print(name, 'attention/head full logits validated', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--devices', nargs='+', choices=list(DEVICES), default=list(DEVICES))
    a = p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.devices)) as pool:
        futures = [pool.submit(run, a, name) for name in a.devices]
        for future in concurrent.futures.as_completed(futures):
            future.result()
