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
"""Prepare the audited joined graph for a Bazel native/LiteRT comparison."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, default=Path(
    '/data/bt/tmp/gemma4-native-single-graph-20260923/build-v3/overlay'))
a = p.parse_args()
here = Path(__file__).resolve().parent
repo = here.parents[2]
native = 'tensor/examples/gemma4/native/'
prefix = 'tensor/experiments/litert_joined_decode/'
files = ['driver.cc', 'driver_support.h', 'model/gemma4_graph.h',
         'model/helpers/active_graph_context.h', 'model/helpers/attention.h',
         'model/helpers/transformer.h']
sources = {}
for name in files:
    source = a.source / native / name
    if not source.exists():
        source = repo / native / name
    s = source.read_text()
    sources[name] = hashlib.sha256(s.encode()).hexdigest()
    for f in files[1:]:
        s = s.replace('"' + native + f + '"', '"' + prefix + f + '"')
    if name == 'driver.cc':
        def once(old, new):
            global s
            assert s.count(old) == 1, old
            s = s.replace(old, new)
        once('#include "xnnpack/subgraph.h"',
             '#include "xnnpack/subgraph.h"\n#include "' + prefix + 'mapped_runner.h"')
        once('ABSL_FLAG(bool, joined_decode_graph,',
             'ABSL_FLAG(bool, litert_decode_graph, false, "Run joined decode through LiteRT.");\n'
             'ABSL_FLAG(bool, match_delegate_worker_policy, true, "Use delegate worker policy in native joined decode.");\n'
             'ABSL_FLAG(bool, joined_decode_graph,')
        once('  SR preprocess, tail;',
             '  SR preprocess, tail;\n  std::unique_ptr<MappedRunner> mapped_tail;')
        once('  std::unique_ptr<pthreadpool,PoolDeleter> pool;',
             '  SharedCpuResources cpu;\n  pthreadpool_t pool = nullptr;')
        once('    int capacity,pthreadpool_t pool,xnn_weights_cache_t cache, xnn_workspace_t workspace) {',
             '    int capacity,pthreadpool_t pool,xnn_weights_cache_t cache, xnn_workspace_t workspace,\n'
             '    TfLiteExternalContext* cpu_context) {')
        once('  if(threads>1){rt->pool.reset(pthreadpool_create(threads));if(!rt->pool)return absl::ResourceExhaustedError("Threadpool allocation failed");}',
             '  rt->pool = rt->cpu.Initialize(threads);')
        s = s.replace('rt->pool.get(),raw,rt->workspace.get()',
                      'rt->pool,raw,rt->workspace.get(),&rt->cpu.context')
        once('template<class Data>\nabsl::Status BindLive(StageRunner& runner,',
             'template<class Runner, class Data>\nabsl::Status BindLive(Runner& runner,')
        needle = '    LRT_TENSOR_ASSIGN_OR_RETURN(sig->tail,StageRunner::Create(roots,pool,cache,0,workspace));'
        once(needle, '''    if (absl::GetFlag(FLAGS_litert_decode_graph)) {
      LRT_TENSOR_ASSIGN_OR_RETURN(sig->mapped_tail, MappedRunner::Create(
          roots, pool, cache, cpu_context, absl::GetFlag(FLAGS_output_dir)));
      return sig;
    }
''' + '''    const uint32_t decode_flags = absl::GetFlag(FLAGS_match_delegate_worker_policy)
        ? XNN_FLAG_DONT_SPIN_WORKERS : 0;
    LRT_TENSOR_ASSIGN_OR_RETURN(sig->tail,StageRunner::Create(roots,pool,cache,decode_flags,workspace));''')
        once('    const auto* sg=static_cast<const XnnpackGraph&>(sig->tail->graph()).GetSubgraph();',
             '    WriteKernelInventory(sig->tail->runtime(), absl::GetFlag(FLAGS_output_dir));\n'
             '    const auto* sg=static_cast<const XnnpackGraph&>(sig->tail->graph()).GetSubgraph();')
        once('    LRT_TENSOR_RETURN_IF_ERROR(audit_static(signature->tail));',
             '    LRT_TENSOR_RETURN_IF_ERROR(audit_static(signature->tail));\n'
             '    if(signature->mapped_tail) {\n'
             '      LRT_TENSOR_ASSIGN_OR_RETURN(auto n, CountStaticInt2RuntimeOperators(signature->mapped_tail->runtime()));\n'
             '      rt->static_int2_operators += n;\n    }')
        once('      LRT_TENSOR_RETURN_IF_ERROR(stage(signature->tail));',
             '      LRT_TENSOR_RETURN_IF_ERROR(stage(signature->tail));\n'
             '      if(signature->mapped_tail) signature->mapped_tail->AddMemoryCounters(counts);')
        once('  const auto runtime=sig.tail->runtime();',
             '  const auto runtime=sig.mapped_tail ? sig.mapped_tail->runtime() : sig.tail->runtime();')
        once('absl::Status RunJoinedDecode(LiveSignature& sig,const Config& config,',
             'template<class Runner>\nabsl::Status RunJoinedDecodeImpl(Runner& runner, LiveSignature& sig,const Config& config,')
        once('  auto& runner=*sig.tail;\n', '')
        # The two typed reads now occur in a function template.
        s = s.replace('runner.ReadOutputAs<int8_t>', 'runner.template ReadOutputAs<int8_t>')
        once('absl::Status RunLiveTokens(LiveSignature& sig', '''absl::Status RunJoinedDecode(LiveSignature& sig,const Config& config,
    const LoadedTensors& loaded,ActiveKvBank& bank,
    absl::Span<const int32_t> tokens,StageTimes& times) {
  if(sig.mapped_tail) return RunJoinedDecodeImpl(*sig.mapped_tail,sig,config,loaded,bank,tokens,times);
  return RunJoinedDecodeImpl(*sig.tail,sig,config,loaded,bank,tokens,times);
}

absl::Status RunLiveTokens(LiveSignature& sig''')
        once('rt.decode->tail->ReadOutputAs<float>(rt.decode->outputs.logits)',
             '(rt.decode->mapped_tail ? rt.decode->mapped_tail->ReadOutputAs<float>(rt.decode->outputs.logits) : rt.decode->tail->ReadOutputAs<float>(rt.decode->outputs.logits))')
        once('  const std::string dir=absl::GetFlag(FLAGS_output_dir);',
             '  if(absl::GetFlag(FLAGS_litert_decode_graph) && !absl::GetFlag(FLAGS_joined_decode_graph))\n'
             '    return absl::InvalidArgumentError("LiteRT comparison requires joined decode");\n'
             '  const std::string dir=absl::GetFlag(FLAGS_output_dir);')
        once('     <<",\\\"prefill_rows\\\":"<<rt.prefill->rows',
             '     <<",\\\"litert_decode_graph\\\":"<<(rt.decode->mapped_tail ? "true" : "false")\n'
             '     <<",\\\"prefill_rows\\\":"<<rt.prefill->rows')
    dest = here / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(s)
(here / 'source_identity.json').write_text(json.dumps(sources, indent=2) + '\n')
