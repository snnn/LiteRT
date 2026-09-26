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
"""Build an isolated operator profiler using the preceding matched archives."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
record = json.loads((a.baseline / 'commands.json').read_text())
driver_name = 'tensor/examples/gemma4/native/driver.cc'
before = (a.baseline / driver_name).read_text()
source = before.replace('#include "tensor/examples/gemma4/native/memory_snapshot.h"',
                       '#include "tensor/examples/gemma4/native/memory_snapshot.h"\n'
                       '#include "tensor/experiments/native_mlp_profile/post_profile.h"\n'
                       'ABSL_FLAG(bool, profile_post_ops, false, "Collect diagnostic post-attention operator timings.");')
old = 'StageRunner::Create({cut.output},pool,cache,0,workspace)'
assert source.count(old) == 1
source = source.replace(old, 'StageRunner::Create({cut.output},pool,cache,\n'
                        '        decode && absl::GetFlag(FLAGS_profile_post_ops) ? XNN_FLAG_BASIC_PROFILING : 0,workspace)')
old = '    LRT_TENSOR_RETURN_IF_ERROR(stage.post->PrepareRuntime());'
assert source.count(old) == 1
source = source.replace(old, old + '\n'
    '    if (decode && absl::GetFlag(FLAGS_profile_post_ops))\n'
    '      LRT_TENSOR_RETURN_IF_ERROR(post_operator_profile.Register(i, *stage.post, loaded.weights_handle));')
old = '    elapsed=Milliseconds(at);times.forward+=elapsed;times.post+=elapsed;'
assert source.count(old) == 1
source = source.replace(old, old + '\n'
    '    if (sig.decode && absl::GetFlag(FLAGS_profile_post_ops))\n'
    '      LRT_TENSOR_RETURN_IF_ERROR(post_operator_profile.Sample(i, elapsed));')
old = '    int32_t token=i?item.forced[i-1]:item.prompt.back();'
source = source.replace(old, old + '\n'
    '    post_operator_profile.collecting = absl::GetFlag(FLAGS_profile_post_ops) && !warmup && i > 0;')
old = '  return WriteLiveRun(item,index,warmup,threads,reused,compiled,rt,load_ms,passes,dir,trace.enabled(),decode_section.elapsed_ms());'
source = source.replace(old, '  post_operator_profile.collecting = false;\n' + old)
old = '  rt.reset();\n  LRT_TENSOR_RETURN_IF_ERROR(RecordMemory("after_runtime_destroy"'
source = source.replace(old, '  if (absl::GetFlag(FLAGS_profile_post_ops))\n'
    '    LRT_TENSOR_RETURN_IF_ERROR(post_operator_profile.Write(dir));\n' + old)
old = '  bool reuse=absl::GetFlag(FLAGS_reuse_runtimes);'
source = source.replace(old, old + '\n'
    '  if (absl::GetFlag(FLAGS_profile_post_ops) && !reuse)\n'
    '    return absl::InvalidArgumentError("Profiler requires reused runtimes");')
source = source.replace('(absl::GetFlag(FLAGS_memory_report)?"false":"true")',
                        '((absl::GetFlag(FLAGS_memory_report)||absl::GetFlag(FLAGS_profile_post_ops))?"false":"true")')
source_path = a.out / 'driver.cc'
source_path.write_text(source)
patch = ''.join(difflib.unified_diff(before.splitlines(True), source.splitlines(True),
                                   fromfile='a/' + driver_name, tofile='b/' + driver_name))
Path(__file__).with_name('driver.patch').write_text(patch)
helper = Path(__file__).with_name('post_profile.h')
helper_copy = a.out / 'overlay/tensor/experiments/native_mlp_profile/post_profile.h'
helper_copy.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(helper, helper_copy)
compile_cmd = list(record['compile'])
compile_cmd.insert(1, '-I' + str((a.out / 'overlay').resolve()))
compile_cmd[compile_cmd.index('-c') + 1] = str(source_path.resolve())
obj, exe = a.out / 'driver.o', a.out / 'runner'
compile_cmd[compile_cmd.index('-o') + 1] = str(obj.resolve())
link = []
skip = False
for i, arg in enumerate(record['link']):
    if skip:
        skip = False
        continue
    if arg == '-Xlinker' and record['link'][i+1].startswith('--dependency-file='):
        skip = True
        continue
    if arg.startswith('-Wl,--dependency-file='):
        continue
    link.append(arg)
old_obj = record['compile'][record['compile'].index('-o') + 1]
link[link.index(old_obj)] = str(obj.resolve())
link[link.index('-o') + 1] = str(exe.resolve())
new_record = {'compile': compile_cmd, 'link': link, 'cwd': record['cwd'],
              'driver_sha256': sha(source_path), 'helper_sha256': sha(helper_copy),
              'baseline_binary_sha256': record['executable_sha256']}
(a.out / 'commands.json').write_text(json.dumps(new_record, indent=2) + '\n')
with (a.out / 'build.log').open('w') as log:
    for command in [compile_cmd, link]:
        result = subprocess.run(command, cwd=record['cwd'], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((a.out / 'build.log').read_text()[-12000:])
            raise SystemExit(result.returncode)
new_record['executable_sha256'] = sha(exe)
(a.out / 'commands.json').write_text(json.dumps(new_record, indent=2) + '\n')
print('Built', exe, new_record['executable_sha256'])
