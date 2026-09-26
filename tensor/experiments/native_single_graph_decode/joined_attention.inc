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

    if (ctx.joined_decode) {
      if (cut.owner == ctx.layer) {
        auto make_cache_input = [&](const char* suffix, const TensorHandle& cache) {
          return Tensor<Mixins...>({
              .name=absl::StrCat(name, suffix), .type=Type::kI8,
              .shape={1,1,1,head_dim}, .quantization=CloneKvQuantization(cache)});
        };
        Tensor past_k = make_cache_input(".past_k", key_cache);
        Tensor past_v = make_cache_input(".past_v", value_cache);
        Tensor pad_k = make_cache_input(".pad_k", key_cache);
        Tensor pad_v = make_cache_input(".pad_v", value_cache);
        cut.past_key=past_k; cut.past_value=past_v;
        cut.pad_key=pad_k; cut.pad_value=pad_v;
        auto keys=Concatenation({past_k,nk,pad_k},2);
        auto values=Concatenation({past_v,nv,pad_v},2);
        keys.SetQuantization(CloneKvQuantization(key_cache));
        values.SetQuantization(CloneKvQuantization(value_cache));
        cut.active_key=keys; cut.active_value=values;
      } else {
        cut.active_key=ctx.layers[cut.owner].active_key;
        cut.active_value=ctx.layers[cut.owner].active_value;
      }
      Tensor<Mixins...> mask({.name=absl::StrCat(name,".joined_mask"),
          .type=Type::kFP32,.shape={1,1,config.num_heads,3}});
      cut.mask=mask;
      auto scores=BatchMatMul(Tensor<Mixins...>(cut.query),
          Cast(Tensor<Mixins...>(cut.active_key),Type::kFP32),false,true);
      if(config.attn_logits_soft_cap) {
        float cap=*config.attn_logits_soft_cap;
        scores=Mul(Tanh(Mul(scores,1.0f/cap)),cap);
      }
      auto context=BatchMatMul(Softmax(Add(scores,mask)),
          Cast(Tensor<Mixins...>(cut.active_value),Type::kFP32),false,false);
      cut.context_input=context;
      context=Reshape(context,{batch_size,config.num_heads,seq_len,head_dim});
      context=Transpose(context,{0,2,1,3});
      context=Reshape(context,{batch_size,seq_len,q_out_dim});
      return {MobileFullyConnected(context,o_proj,&weights),nk,nv,nk,nv};
    }
