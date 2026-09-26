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
"""Relink isolated native probes against an existing matched host/Android build.

Does not invoke CMake or rebuild dependencies. Both executables use exactly the
same archived kernels/libraries and compiler options, isolating the experiment.
For a normal repository build, use its Bazel targets instead.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo, build, out = args.repo.resolve(), args.build.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    driver = 'tensor/examples/gemma4/native/driver.cc'
    header = 'tensor/examples/gemma4/native/stage_runner.h'
    patch = Path(__file__).with_name('probe.patch')
    entry = next(e for e in json.loads((build / 'compile_commands.json').read_text())
                 if e['file'].endswith('/' + driver))
    commands = subprocess.check_output(
        ['ninja', '-C', str(build), '-t', 'commands', 'bin/gemma4_native_runner'], text=True)
    link_template = shlex.split(commands.splitlines()[-1])
    if link_template[:2] == [':', '&&']:
        link_template = link_template[2:]
    if link_template[-2:] == ['&&', ':']:
        link_template = link_template[:-2]
    assert not any(t in ('&&', ';', '|') for t in link_template)
    # Relinking the probe must not replace archived build dependency metadata.
    filtered = []
    skip = False
    for i, value in enumerate(link_template):
        if skip:
            skip = False
            continue
        if value == '-Xlinker' and link_template[i + 1].startswith('--dependency-file='):
            skip = True
            continue
        if value.startswith('-Wl,--dependency-file='):
            continue
        filtered.append(value)
    link_template = filtered
    archives = {str((build / p).resolve()): sha(build / p)
                for p in link_template if p.endswith('.a')}
    # Record the current header inputs as well as archived objects. No global
    # configuration/kernel flags differ between the two rebuilt executables.
    sources = {str(p.relative_to(repo)): sha(p) for p in (repo / 'tensor').rglob('*')
               if p.is_file() and p.suffix in ('.h', '.cc') and 'experiments' not in p.parts}
    (out / 'inputs.json').write_text(json.dumps({
        'archives': archives, 'tensor_sources': sources,
        'patch_sha256': sha(patch), 'compiler': subprocess.check_output(
            [shlex.split(entry['command'])[0], '--version'], text=True),
        'git_head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    }, indent=2) + '\n')
    for variant in ('baseline', 'probe'):
        overlay = out / variant
        for name in (driver, header):
            dest = overlay / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / name, dest)
        if variant == 'probe':
            subprocess.run(['patch', '--batch', '-p1', '-i', str(patch.resolve())],
                           cwd=overlay, check=True)
        obj, exe = overlay / 'driver.o', overlay / 'runner'
        compile_cmd = shlex.split(entry['command'])
        compile_cmd.insert(1, '-I' + str(overlay))
        compile_cmd[compile_cmd.index('-o') + 1] = str(obj)
        compile_cmd[compile_cmd.index('-c') + 1] = str(overlay / driver)
        link = list(link_template)
        indices = [i for i, x in enumerate(link) if x.endswith('/' + driver + '.o')]
        assert len(indices) == 1
        link[indices[0]] = str(obj)
        link[link.index('-o') + 1] = str(exe)
        record = {'compile': compile_cmd, 'link': link, 'cwd': str(build),
                  'overlay': {n: sha(overlay / n) for n in (driver, header)}}
        (overlay / 'commands.json').write_text(json.dumps(record, indent=2) + '\n')
        with (overlay / 'build.log').open('w') as log:
            for command in (compile_cmd, link):
                result = subprocess.run(command, cwd=build, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    print((overlay / 'build.log').read_text()[-12000:])
                    raise SystemExit(result.returncode)
        record['executable_sha256'] = sha(exe)
        (overlay / 'commands.json').write_text(json.dumps(record, indent=2) + '\n')
        print(variant, record['executable_sha256'], flush=True)


if __name__ == '__main__':
    main()
