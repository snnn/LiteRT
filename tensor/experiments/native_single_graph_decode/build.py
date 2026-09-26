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
"""Relink an isolated decode-graph experiment against audited native archives."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def once(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
repo = Path(__file__).resolve().parents[3]
here = Path(__file__).resolve().parent
prefix = 'tensor/examples/gemma4/native/'
names = ['driver.cc', 'model/helpers/active_graph_context.h',
         'model/helpers/attention.h', 'model/gemma4_graph.h']
before = {n: (repo / prefix / n).read_text() for n in names}
after = dict(before)
s = after['model/helpers/active_graph_context.h']
s = once(s, '  int layer = 0;', '  int layer = 0;\n  bool joined_decode = false;')
s = once(s, '  TensorHandle hidden_input, ple_input;',
    '  TensorHandle hidden_input, ple_input;\n'
    '  TensorHandle past_key, past_value, pad_key, pad_value, active_key, active_value, mask;')
after['model/helpers/active_graph_context.h'] = s
s = after['model/gemma4_graph.h']
for old in [
    '      hidden_states = Tensor<Mixins...>({.name=absl::StrCat("active_hidden_",layer_idx),\n'
    '                      .type=Type::kFP32,.shape=hidden_states.GetShape()});',
    '      per_layer_input = Tensor<Mixins...>({.name=absl::StrCat("active_ple_",layer_idx),\n'
    '                       .type=Type::kFP32,.shape=per_layer_input.GetShape()});',
    '    hidden_states = Tensor<Mixins...>({.name="active_final_hidden",.type=Type::kFP32,.shape=hidden_states.GetShape()});',
]:
    s = once(s, old, '    if (!active_graph_context->joined_decode) {\n' + old + '\n    }')
after['model/gemma4_graph.h'] = s
s = after['model/helpers/attention.h']
needle = '    Tensor<Mixins...> context({.name=absl::StrCat(name,".active_context"),'
s = once(s, needle, (here / 'joined_attention.inc').read_text() + '\n' + needle)
after['model/helpers/attention.h'] = s
s = after['driver.cc']
needle = 'ABSL_FLAG(int, kv_alignment, 32,'
s = once(s, needle,
    'ABSL_FLAG(bool, joined_decode_graph, false, "Experimental: one graph per decode invocation.");\n' + needle)
needle = '  struct Scope { Scope(ActiveGraphContext* p)'
s = once(s, needle,
    '  sig->cuts.joined_decode = decode && absl::GetFlag(FLAGS_joined_decode_graph);\n' + needle)
needle = '  std::vector<TensorHandle> roots{sig->cuts.initial_hidden};'
s = once(s, needle, '''  if (sig->cuts.joined_decode) {
    std::vector<TensorHandle> roots{sig->outputs.logits};
    for (const auto& spec : specs) {
      roots.push_back(sig->cuts.layers[spec.owner].new_key);
      roots.push_back(sig->cuts.layers[spec.owner].new_value);
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(sig->tail,StageRunner::Create(roots,pool,cache,0,workspace));
    LRT_TENSOR_RETURN_IF_ERROR(sig->tail->PrepareRuntime());
    const auto* sg=static_cast<const XnnpackGraph&>(sig->tail->graph()).GetSubgraph();
    int fc=0,bmm=0,concat=0;
    for(size_t i=0;i<sg->num_nodes;++i) {
      fc+=sg->nodes[i].type==xnn_node_type_fully_connected;
      bmm+=sg->nodes[i].type==xnn_node_type_batch_matrix_multiply;
      concat+=sg->nodes[i].type==xnn_node_type_concatenate;
    }
    if(fc!=277 || bmm!=70 || concat<30)
      return absl::InternalError("Joined graph work differs from expected model");
    std::cerr<<"Joined decode graph: FC="<<fc<<" BMM="<<bmm<<" concat="<<concat<<" runtimes=1\\n";
    return sig;
  }
''' + needle)
s = once(s, 'int chunks=0;};', 'int chunks=0,decode_invocations=0;};')
needle = 'absl::Status RunLiveTokens(LiveSignature& sig'
s = once(s, needle, (here / 'joined_run.inc').read_text() + '\n' + needle)
needle = '  const int rows=sig.rows,start=bank.length(),count=tokens.size(),end=start+count;'
s = once(s, needle,
    '  if(sig.cuts.joined_decode)return RunJoinedDecode(sig,config,loaded,bank,tokens,times);\n' + needle)
# Count calls at the actual invocation sites, including preprocessing and head.
for call in ['sig.preprocess->Run()', 'stage.projection->Run()', 'attention.runner->Run()',
             'stage.post->Run()', 'sig.tail->Run()']:
    old = 'LRT_TENSOR_RETURN_IF_ERROR(' + call + ');'
    s = once(s, old, old + 'if(sig.decode)++times.decode_invocations;')
needle = '       <<",\\\"invocations\\\":"<<p.times.chunks'
s = once(s, needle,
    '       <<",\\\"decode_runtime_invocations\\\":"<<p.times.decode_invocations\n' + needle)
needle = '  const std::string dir=absl::GetFlag(FLAGS_output_dir);'
s = once(s, needle, '''  if(absl::GetFlag(FLAGS_joined_decode_graph) &&
      (absl::GetFlag(FLAGS_fixed_attention_extent)||absl::GetFlag(FLAGS_trace_position)>=0))
    return absl::InvalidArgumentError("Joined experiment does not support fixed extents or layer trace dumps");
''' + needle)
after['driver.cc'] = s
patch = ''
for name in names:
    dest = a.out / 'overlay' / prefix / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(after[name])
    patch += ''.join(difflib.unified_diff(before[name].splitlines(True), after[name].splitlines(True),
        fromfile='a/' + prefix + name, tofile='b/' + prefix + name))
(a.out / 'driver.patch').write_text(patch)
record = json.loads((a.baseline / 'commands.json').read_text())
archives = {str((Path(record['cwd']) / x).resolve()): sha((Path(record['cwd']) / x).resolve())
            for x in record['link'] if x.endswith('.a')}
previous_archives = json.loads((a.baseline.parent / 'inputs.json').read_text())['archives']
assert archives == previous_archives, 'Archived libraries changed since the baseline build'
(a.out.parent / 'archive-identity.json').write_text(json.dumps({
    'same_as_previous_experiment': True, 'archives': archives}, indent=2) + '\n')
compile_cmd = list(record['compile'])
compile_cmd.insert(1, '-I' + str((a.out / 'overlay').resolve()))
old_obj = compile_cmd[compile_cmd.index('-o') + 1]
obj, exe = (a.out / 'driver.o').resolve(), (a.out / 'runner').resolve()
compile_cmd[compile_cmd.index('-c') + 1] = str((a.out / 'overlay' / prefix / 'driver.cc').resolve())
compile_cmd[compile_cmd.index('-o') + 1] = str(obj)
link = list(record['link'])
link[link.index(old_obj)] = str(obj)
link[link.index('-o') + 1] = str(exe)
new_record = {'compile': compile_cmd, 'link': link, 'cwd': record['cwd'],
              'original_sources_sha256': {n: hashlib.sha256(before[n].encode()).hexdigest() for n in names},
              'overlay_sources_sha256': {n: hashlib.sha256(after[n].encode()).hexdigest() for n in names},
              'baseline_executable_sha256': record['executable_sha256'],
              'patch_sha256': sha(a.out / 'driver.patch')}
def save():
    (a.out / 'commands.json').write_text(json.dumps(new_record, indent=2) + '\n')
save()
with (a.out / 'build.log').open('w') as log:
    for command in (compile_cmd, link):
        result = subprocess.run(command, cwd=record['cwd'], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((a.out / 'build.log').read_text()[-7000:])
            raise SystemExit(result.returncode)
new_record['executable_sha256'] = sha(exe)
save()
print('Built', new_record['executable_sha256'])
