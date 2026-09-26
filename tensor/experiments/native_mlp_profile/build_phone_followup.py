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
"""Extend the isolated profiler to attention and the output-head runtime."""
import argparse
import difflib
import json
from pathlib import Path
import subprocess

from run_profile import save, sha


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--profile', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--timestamps', action='store_true',
               help='Save monotonic invoke timestamps for filtering CPU samples.')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
record = json.loads((a.profile / 'commands.json').read_text())
helper_name = 'tensor/experiments/native_mlp_profile/post_profile.h'
before = (a.profile / 'driver.cc').read_text()
source = before
source = replace_once(source, 'StageRunner::Create({output},pool,cache,0,workspace)',
    'StageRunner::Create({output},pool,cache,rows==1 && absl::GetFlag(FLAGS_profile_post_ops) ? XNN_FLAG_BASIC_PROFILING : 0,workspace)')
needle = '    LRT_TENSOR_ASSIGN_OR_RETURN(stage.post,StageRunner::Create({cut.output},pool,cache,'
source = replace_once(source, needle,
    '    if (decode && absl::GetFlag(FLAGS_profile_post_ops))\n'
    '      LRT_TENSOR_RETURN_IF_ERROR(attention_operator_profile.Register(i, *stage.attention.runner, loaded.weights_handle));\n' + needle)
source = replace_once(source, 'StageRunner::Create({sig->outputs.logits},pool,cache,0,workspace)',
    'StageRunner::Create({sig->outputs.logits},pool,cache,absl::GetFlag(FLAGS_profile_post_ops) ? XNN_FLAG_BASIC_PROFILING : 0,workspace)')
needle = '    LRT_TENSOR_RETURN_IF_ERROR(sig->tail->PrepareRuntime());'
source = replace_once(source, needle, needle + '\n'
    '    if (absl::GetFlag(FLAGS_profile_post_ops))\n'
    '      LRT_TENSOR_RETURN_IF_ERROR(head_operator_profile.Register(-1, *sig->tail, loaded.weights_handle));')
needle = '    elapsed=Milliseconds(at);times.forward+=elapsed;times.attention+=elapsed;'
source = replace_once(source, needle, needle + '\n'
    '    if (sig.decode && absl::GetFlag(FLAGS_profile_post_ops))\n'
    '      LRT_TENSOR_RETURN_IF_ERROR(attention_operator_profile.Sample(i, elapsed));')
source = replace_once(source,
    'at=BenchClock::now();LRT_TENSOR_RETURN_IF_ERROR(sig.tail->Run());times.forward+=Milliseconds(at);',
    'at=BenchClock::now();LRT_TENSOR_RETURN_IF_ERROR(sig.tail->Run());\n'
    '    const double head_elapsed=Milliseconds(at);times.forward+=head_elapsed;\n'
    '    if (absl::GetFlag(FLAGS_profile_post_ops))\n'
    '      LRT_TENSOR_RETURN_IF_ERROR(head_operator_profile.Sample(-1, head_elapsed));')
needle = '    post_operator_profile.collecting = absl::GetFlag(FLAGS_profile_post_ops) && !warmup && i > 0;'
source = replace_once(source, needle, needle + '\n'
    '    attention_operator_profile.collecting = head_operator_profile.collecting = post_operator_profile.collecting;')
source = replace_once(source, '  post_operator_profile.collecting = false;',
    '  post_operator_profile.collecting = attention_operator_profile.collecting = head_operator_profile.collecting = false;')
source = replace_once(source,
    '  if (absl::GetFlag(FLAGS_profile_post_ops))\n    LRT_TENSOR_RETURN_IF_ERROR(post_operator_profile.Write(dir));',
    '  if (absl::GetFlag(FLAGS_profile_post_ops)) {\n'
    '    LRT_TENSOR_RETURN_IF_ERROR(post_operator_profile.Write(dir));\n'
    '    LRT_TENSOR_RETURN_IF_ERROR(attention_operator_profile.Write(dir,"attention",35));\n'
    '    LRT_TENSOR_RETURN_IF_ERROR(head_operator_profile.Write(dir,"head",1));\n  }')
helper_before = (a.profile / 'overlay' / helper_name).read_text()
helper = replace_once(helper_before, 'absl::Status Write(const std::string& directory) const {',
    'absl::Status Write(const std::string& directory, const std::string& prefix="post", size_t expected_layers=35) const {')
helper = helper.replace('directory + "/post_', 'directory + "/" + prefix + "_')
helper = helper.replace('layers_.size() != 35', 'layers_.size() != expected_layers')
helper = replace_once(helper,
    '          if (found == names.end())\n'
    '            return absl::InternalError("FC source pointer was not identified");\n'
    '          item.weight = found->second.first;\n'
    '          item.source_bytes = found->second.second;',
    '          if (found == names.end()) {\n'
    '            if (layer != -1 || value.datatype != xnn_datatype_qcint2 ||\n'
    '                value.shape.num_dims != 2 || value.shape.dim[0] != 262144 || value.shape.dim[1] != 1536)\n'
    '              return absl::InternalError("Unidentified FC source is not the expected compact head");\n'
    '            item.weight = "lm_head.weight.runtime_compact_int2";\n'
    '            item.source_bytes = xnn_runtime_tensor_get_size(&value);\n'
    '          } else {\n'
    '            item.weight = found->second.first;\n'
    '            item.source_bytes = found->second.second;\n'
    '          }')
helper = replace_once(helper, 'inline PostOperatorProfile post_operator_profile;',
    'inline PostOperatorProfile post_operator_profile, attention_operator_profile, head_operator_profile;')
if a.timestamps:
    helper = replace_once(helper, '    std::vector<uint64_t> wall_ns;',
        '    std::vector<uint64_t> wall_ns, invoke_start_ns;')
    helper = replace_once(helper, '    data.wall_ns.reserve(1024);',
        '    data.wall_ns.reserve(1024);\n    data.invoke_start_ns.reserve(1024);')
    helper = replace_once(helper, '    int64_t previous = Nanoseconds(data.runtime->start_ts);',
        '    int64_t previous = Nanoseconds(data.runtime->start_ts);\n'
        '    data.invoke_start_ns.push_back(previous);')
    helper = replace_once(helper, '      WriteArray(stages, data.wall_ns); stages << "}\\n";',
        '      WriteArray(stages, data.wall_ns);\n'
        '      stages << ",\\\"invoke_start_ns\\\":";\n'
        '      WriteArray(stages, data.invoke_start_ns); stages << "}\\n";')
driver = a.out / 'driver.cc'
driver.write_text(source)
header = a.out / 'overlay' / helper_name
header.parent.mkdir(parents=True, exist_ok=True)
header.write_text(helper)
patch = ''.join(difflib.unified_diff(before.splitlines(True), source.splitlines(True),
                 fromfile='post-only/driver.cc', tofile='all-stages/driver.cc'))
patch += ''.join(difflib.unified_diff(helper_before.splitlines(True), helper.splitlines(True),
                 fromfile='post-only/post_profile.h', tofile='all-stages/post_profile.h'))
(a.out / 'followup.patch').write_text(patch)
compile_cmd = list(record['compile'])
compile_cmd.insert(1, '-I' + str((a.out / 'overlay').resolve()))
obj, exe = (a.out / 'driver.o').resolve(), (a.out / 'runner').resolve()
old_obj = compile_cmd[compile_cmd.index('-o') + 1]
compile_cmd[compile_cmd.index('-c') + 1] = str(driver.resolve())
compile_cmd[compile_cmd.index('-o') + 1] = str(obj)
link = list(record['link'])
link[link.index(old_obj)] = str(obj)
link[link.index('-o') + 1] = str(exe)
new_record = {'compile': compile_cmd, 'link': link, 'cwd': record['cwd'],
              'driver_sha256': sha(driver), 'helper_sha256': sha(header),
              'absolute_monotonic_timestamps': a.timestamps,
              'post_only_executable_sha256': record['executable_sha256']}
save(a.out / 'commands.json', new_record)
with (a.out / 'build.log').open('w') as log:
    for command in (compile_cmd, link):
        result = subprocess.run(command, cwd=record['cwd'], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((a.out / 'build.log').read_text()[-8000:])
            raise SystemExit(result.returncode)
new_record['executable_sha256'] = sha(exe)
save(a.out / 'commands.json', new_record)
print('Built follow-up', new_record['executable_sha256'])
