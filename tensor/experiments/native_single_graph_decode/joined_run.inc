// Copyright 2026 Google LLC.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

absl::Status AuditJoinedAttention(LiveSignature& sig,const Config& config,int start,int end) {
  std::multiset<std::array<size_t,6>> expected,actual;
  const int alignment=absl::GetFlag(FLAGS_kv_alignment);
  const int padded_end=(end+alignment-1)/alignment*alignment;
  for(const auto& cut:sig.cuts.layers) {
    int begin=cut.global?0:std::max(0,start-config.sliding_window_size+1);
    begin=begin/alignment*alignment;
    const size_t m=config.num_heads,d=cut.dim,s=padded_end-begin;
    expected.insert({m,d,s,d,m,s});
    expected.insert({m,s,s,d,m,d});
  }
  const auto runtime=sig.tail->runtime();
  for(size_t i=0;i<runtime->num_ops;++i) {
    const auto& op=runtime->opdata[i];
    if(op.type!=xnn_node_type_batch_matrix_multiply)continue;
    int objects=0;
    for(auto* object:op.operator_objects)if(object) {
      ++objects;
      if(object->type!=xnn_operator_type_batch_matrix_multiply_nc_f32_qc8w)
        return absl::InternalError("Joined attention changed the BMM kernel family");
    }
    if(!objects)continue;
    if(objects!=1||op.num_inputs!=2||op.num_outputs!=1)
      return absl::InternalError("Unexpected joined BMM arity");
    const auto& a=runtime->values[op.inputs[0]];
    const auto& b=runtime->values[op.inputs[1]];
    const auto& c=runtime->values[op.outputs[0]];
    if(a.datatype!=xnn_datatype_fp32||b.datatype!=xnn_datatype_qcint8||c.datatype!=xnn_datatype_fp32)
      return absl::InternalError("Joined attention changed operand precision");
    for(auto* value:{&a,&b,&c})if(value->shape.num_dims!=4||value->shape.dim[0]!=1||value->shape.dim[1]!=1)
      return absl::InternalError("Unexpected joined attention rank/batch");
    actual.insert({a.shape.dim[2],a.shape.dim[3],b.shape.dim[2],b.shape.dim[3],c.shape.dim[2],c.shape.dim[3]});
  }
  if(actual!=expected)return absl::InternalError("Joined attention work extents differ from staged attention");
  return absl::OkStatus();
}

absl::Status RunJoinedDecode(LiveSignature& sig,const Config& config,
    const LoadedTensors& loaded,ActiveKvBank& bank,
    absl::Span<const int32_t> tokens,StageTimes& times) {
  if(tokens.size()!=1 || !sig.decode || sig.rows!=1)
    return absl::InvalidArgumentError("Joined experiment supports decode only");
  const int start=bank.length(),end=start+1;
  const int alignment=absl::GetFlag(FLAGS_kv_alignment);
  const int padded_end=(end+alignment-1)/alignment*alignment;
  LRT_TENSOR_RETURN_IF_ERROR(bank.BeginAppend(1));
  struct Transaction {ActiveKvBank& bank;~Transaction(){if(bank.append_in_flight())bank.Abort();}} transaction{bank};
  auto& runner=*sig.tail;
  std::vector<float> embeddings(config.embed_dim);
  std::vector<std::vector<float>> ple(config.num_layers,
      std::vector<float>(config.per_layer_input_dim));
  LRT_TENSOR_RETURN_IF_ERROR(loaded.token_embedding->Lookup(tokens,absl::MakeSpan(embeddings)));
  LRT_TENSOR_RETURN_IF_ERROR(loaded.emb_per_layer_table->LookupPerLayer(tokens,
      config.num_layers,config.per_layer_input_dim,absl::MakeSpan(ple)));
  LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput(sig.inputs.embedded_input,embeddings));
  for(int i=0;i<config.num_layers;++i)
    LRT_TENSOR_RETURN_IF_ERROR(BindLive(runner,sig.inputs.per_layer_token_embeddings[i],ple[i]));
  std::vector<float> positions{float(start)};
  LRT_TENSOR_RETURN_IF_ERROR(BindLive(runner,sig.inputs.weights.at("matched.rope.positions"),positions));
  std::vector<int8_t> zero_padding(size_t(alignment)*config.global_key_size+XNN_EXTRA_BYTES,0);
  std::vector<std::vector<float>> masks(config.num_layers);
  for(int i=0;i<config.num_layers;++i) {
    auto& cut=sig.cuts.layers[i];
    int begin=cut.global?0:std::max(0,start-config.sliding_window_size+1);
    begin=begin/alignment*alignment;
    const int extent=padded_end-begin;
    if(cut.owner==i) {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto k,bank.Keys(i,begin,start));
      LRT_TENSOR_ASSIGN_OR_RETURN(auto v,bank.Values(i,begin,start));
      for(auto h:{cut.past_key,cut.past_value})
        LRT_TENSOR_RETURN_IF_ERROR(runner.ReshapeInput(h,Shape{1,1,start-begin,cut.dim}));
      for(auto h:{cut.pad_key,cut.pad_value}) {
        LRT_TENSOR_RETURN_IF_ERROR(runner.ReshapeInput(h,Shape{1,1,padded_end-end,cut.dim}));
        const auto padding=absl::Span<const int8_t>(
            zero_padding.data(),size_t(padded_end-end)*cut.dim);
        LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput(h,padding));
      }
      LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput(cut.past_key,k));
      LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput(cut.past_value,v));
    }
    LRT_TENSOR_RETURN_IF_ERROR(runner.ReshapeInput(cut.mask,Shape{1,1,config.num_heads,extent}));
    masks[i].assign(size_t(config.num_heads)*extent,std::numeric_limits<float>::lowest());
    int lower=cut.global?0:std::max(0,start-config.sliding_window_size+1);
    for(int h=0;h<config.num_heads;++h)
      for(int p=std::max(lower,begin);p<=start;++p)masks[i][size_t(h)*extent+p-begin]=0;
    LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput(cut.mask,masks[i]));
  }
  // Exactly one runtime invocation: every projection, attention, MLP, and the
  // head are connected by graph edges. No callbacks or host layer execution.
  auto at=BenchClock::now();
  LRT_TENSOR_RETURN_IF_ERROR(runner.Run());
  times.forward+=Milliseconds(at);
  ++times.decode_invocations;
  if(absl::GetFlag(FLAGS_dump_full_logits))
    LRT_TENSOR_RETURN_IF_ERROR(AuditJoinedAttention(sig,config,start,end));
  // Host commit is a batch of only the new owner rows, after graph completion.
  at=BenchClock::now();
  for(const auto& spec:bank.specs()) {
    const auto& cut=sig.cuts.layers[spec.owner];
    LRT_TENSOR_ASSIGN_OR_RETURN(auto k,runner.ReadOutputAs<int8_t>(cut.new_key));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto v,runner.ReadOutputAs<int8_t>(cut.new_value));
    LRT_TENSOR_RETURN_IF_ERROR(bank.Append(spec.owner,k,v));
  }
  times.append+=Milliseconds(at);
  LRT_TENSOR_RETURN_IF_ERROR(bank.Commit());++times.chunks;
  return absl::OkStatus();
}
