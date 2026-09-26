/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Reuse deterministic inputs and simple checked-status helpers. The test's
// graph exports intermediates; the timed graph below exports ONLY context.
#define main unused_active_attention_test_main
#include "tensor/examples/gemma4/native/tests/active_attention_test.cc"
#undef main

#include <chrono>
#include <memory>
#include <sstream>

namespace {
using Clock = std::chrono::steady_clock;

uint64_t Digest(const Inputs& in) {
  uint64_t value = 1469598103934665603ULL;
  auto consume = [&](const auto& data) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(data.data());
    for (size_t i = 0; i < data.size() * sizeof(data[0]); ++i) {
      value ^= bytes[i];
      value *= 1099511628211ULL;
    }
  };
  consume(in.keys); consume(in.values); consume(in.transposed_values);
  consume(in.q); consume(in.mask); consume(in.folded_mask);
  return value;
}

struct AttentionCostCase {
  bool folded, transposed;
  XT q, k, v, mask, output;
  XnnpackRunner runner;
  std::string audit;

  AttentionCostCase(const Inputs& in, float ks, float vs, bool fold, bool transpose,
        int threads)
      : folded(fold), transposed(transpose),
        q(F("q", folded ? Shape{1,1,kHeads,in.dim} : Shape{1,kHeads,1,in.dim})),
        k(I("k", {1,1,in.extent,in.dim}, ks)),
        v(I("v", transposed ? Shape{1,1,in.dim,in.extent}
                            : Shape{1,1,in.extent,in.dim}, vs)),
        mask(F("mask", folded ? Shape{1,1,kHeads,in.extent}
                               : Shape{1,1,1,in.extent})),
        output(BatchMatMul(Softmax(Add(BatchMatMul(q, Cast(k, Type::kFP32),
                    false, true), mask)), Cast(v, Type::kFP32),
                    false, transposed)),
        runner(Take(XnnpackRunner::Create({output}))) {
    runner.SetNumThreads(threads);
    Check(runner.SetInput(q, in.q));
    const absl::Span<const int8_t> keys(
        in.keys.data(), size_t(in.extent) * in.dim);
    const absl::Span<const int8_t> values(
        transposed ? in.transposed_values.data() : in.values.data(),
        size_t(in.extent) * in.dim);
    Check(runner.SetInput(k, keys));
    Check(runner.SetInput(v, values));
    Check(runner.SetInput(mask, folded ? in.folded_mask : in.mask));
    Check(runner.Run());
    Inspect(in);
  }

  void Inspect(const Inputs& in) {
    auto* rt = runner.runtime();
    int bmm = 0, adapters = 0, softmax = 0;
    std::ostringstream out;
    out << "{\"workspace_bytes\":" << rt->workspace->size << ",\"operators\":[";
    bool first = true;
    auto shape = [&](const xnn_shape& s) {
      out << '[';
      for (size_t d = 0; d < s.num_dims; ++d) {
        if (d) out << ',';
        out << s.dim[d];
      }
      out << ']';
    };
    for (size_t i = 0; i < rt->num_ops; ++i) {
      const auto& op = rt->opdata[i];
      for (const auto* object : op.operator_objects) {
        if (!object) continue;
        const std::string name = xnn_operator_type_to_string(object->type);
        if (!first) out << ',';
        first = false;
        out << "{\"name\":\"" << name << "\",\"workspace_bytes\":"
            << op.workspace_size << ",\"inputs\":[";
        for (size_t e = 0; e < op.num_inputs; ++e) {
          if (e) out << ',';
          shape(rt->values[op.inputs[e]].shape);
        }
        out << ']';
        if (object->type == xnn_operator_type_batch_matrix_multiply_nc_f32_qc8w) {
          ++bmm;
          const auto& rhs = rt->values[op.inputs[1]];
          Require(rhs.datatype == xnn_datatype_qcint8, "BMM RHS must remain QC8");
          size_t size = 1;
          for (size_t d = 0; d < rhs.shape.num_dims; ++d) size *= rhs.shape.dim[d];
          Require(size == size_t(in.extent) * in.dim, "BMM extent mismatch");
          out << ",\"compute\":[";
          for (size_t c = 0; c < object->num_compute_invocations; ++c) {
            if (c) out << ',';
            const auto& compute = object->compute[c];
            out << "{\"type\":" << int(compute.type) << ",\"ranges\":["
                << compute.range[0] << ',' << compute.range[1] << ','
                << compute.range[2] << "],\"tiles\":[" << compute.tile[0]
                << ',' << compute.tile[1] << "]}";
          }
          out << ']';
        } else if (object->type == xnn_operator_type_convert_nc_qs8_qc8) {
          ++adapters;
        } else if (name.find("Convert") != std::string::npos ||
                   name.find("Batch Matrix Multiply") != std::string::npos) {
          throw std::runtime_error("Unexpected materialized or quantized path: " + name);
        } else if (name.find("Softmax") != std::string::npos) {
          ++softmax;
        }
        out << '}';
      }
    }
    out << "]}";
    Require(bmm == 2 && adapters == 2 && softmax == 1,
            "Expected two fused F32/QC8 BMMs, two QS8/QC8 adapters, one softmax");
    audit = out.str();
  }

  std::vector<float> Read() {
    auto data = Take(runner.ReadOutputAs<float>(output));
    return {data.begin(), data.end()};
  }
};

std::vector<float> ReferenceContext(const Inputs& in, float ks, float vs) {
  std::vector<float> result(size_t(kHeads) * in.dim);
  std::vector<double> scores(in.extent);
  for (int h = 0; h < kHeads; ++h) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (int t = 0; t < in.extent; ++t) {
      double score = 0;
      for (int d = 0; d < in.dim; ++d)
        score += double(in.q[size_t(h) * in.dim + d]) *
                 in.keys[size_t(t) * in.dim + d] * ks;
      scores[t] = in.mask[t] == 0 ? score : -std::numeric_limits<double>::infinity();
      maximum = std::max(maximum, scores[t]);
    }
    double sum = 0;
    for (double& score : scores) { score = std::exp(score - maximum); sum += score; }
    for (int d = 0; d < in.dim; ++d) {
      double value = 0;
      for (int t = 0; t < in.extent; ++t)
        value += scores[t] / sum * in.values[size_t(t) * in.dim + d] * vs;
      result[size_t(h) * in.dim + d] = float(value);
    }
  }
  return result;
}

std::vector<int> ParseList(const std::string& value) {
  std::vector<int> output;
  std::stringstream input(value);
  std::string item;
  while (std::getline(input, item, ',')) output.push_back(std::stoi(item));
  Require(!output.empty(), "Empty dimension/depth list");
  return output;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int threads = 4, passes = 3;
    double target_ms = 200;
    std::vector<int> depths{544,1088,2048,4096,8448}, dims{256,512};
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      auto eq = arg.find('=');
      Require(eq != std::string::npos, "Expected --option=value");
      const auto key = arg.substr(0,eq), value = arg.substr(eq+1);
      if (key == "--threads") threads = std::stoi(value);
      else if (key == "--passes") passes = std::stoi(value);
      else if (key == "--ms") target_ms = std::stod(value);
      else if (key == "--depths") depths = ParseList(value);
      else if (key == "--dims") dims = ParseList(value);
      else throw std::runtime_error("Unknown option " + key);
    }
    Require(threads >= 1 && threads <= 32 && passes >= 1 && passes <= 20 &&
            std::isfinite(target_ms) && target_ms >= 1 && target_ms <= 10000,
            "Invalid measurement controls");
    std::cout << std::setprecision(12);
    for (int dim : dims) {
      Require(dim == 256 || dim == 512, "Supported dimensions: 256,512");
      auto spec = std::find_if(PublishedE2BKvOwnerSpecs().begin(),
          PublishedE2BKvOwnerSpecs().end(),
          [&](const auto& s) { return s.head_dim == dim; });
      Require(spec != PublishedE2BKvOwnerSpecs().end(), "Missing published scale");
      for (int depth : depths) {
        Require(depth >= 1 && depth <= 16384, "Invalid depth");
        Inputs in(depth, dim, {depth-1,1}, false);
        // Keep query and valid cache bytes independent of the allocated width.
        for (int h = 0; h < kHeads; ++h) for (int d = 0; d < dim; ++d) {
          const int code = int(Hash(h * 1009U + d * 17U + 43U) & 1023U) - 512;
          in.q[size_t(h) * dim + d] = float(code) / (512.0f * std::sqrt(float(dim)));
        }
        const int valid = std::min(depth,1088);
        for (int t = valid; t < depth; ++t) {
          in.mask[t] = std::numeric_limits<float>::lowest();
          for (int h = 0; h < kHeads; ++h)
            in.folded_mask[size_t(h)*depth+t] = std::numeric_limits<float>::lowest();
        }
        const uint64_t before = Digest(in);
        const auto reference = ReferenceContext(in,spec->key_scale,spec->value_scale);
        std::vector<std::unique_ptr<AttentionCostCase>> probes;
        for (bool folded : {true,false}) for (bool transposed : {false,true})
          probes.push_back(std::make_unique<AttentionCostCase>(in,spec->key_scale,
              spec->value_scale,folded,transposed,threads));
        const auto baseline = probes[0]->Read();
        std::vector<double> ref_errors, layout_errors;
        for (const auto& probe : probes) {
          auto values = probe->Read();
          Error reference_error, layout_error;
          reference_error.Compare(values,reference);
          layout_error.Compare(values,baseline);
          Require(reference_error.max_abs < 2e-5 && layout_error.max_abs < 2e-5,
                  "Attention differs from FP64 reference or layout baseline");
          ref_errors.push_back(reference_error.max_abs);
          layout_errors.push_back(layout_error.max_abs);
        }
        for (int pass = 0; pass < passes; ++pass) {
          for (size_t step = 0; step < probes.size(); ++step) {
            const size_t index = (step + pass) % probes.size();
            auto& probe = *probes[index];
            // All construction, binding, reference checks and warmup excluded.
            for (int warmup = 0; warmup < 5; ++warmup) Check(probe.runner.Run());
            const auto begin = Clock::now();
            size_t calls = 0;
            double elapsed = 0;
            do {
              for (int batch = 0; batch < 8; ++batch) {
                Check(probe.runner.Run()); ++calls;
              }
              elapsed = std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
            } while (elapsed < target_ms);
            Require(Digest(in) == before, "Attention modified immutable input");
            Error after_error;
            after_error.Compare(probe.Read(),reference);
            Require(after_error.max_abs < 2e-5, "Output changed during measurement");
            std::cout << "{\"schema\":1,\"kind\":\"hot_cache_synthetic_attention\","
                      << "\"depth\":" << depth << ",\"valid_tokens\":" << valid
                      << ",\"head_dim\":" << dim << ",\"heads\":" << kHeads
                      << ",\"threads\":" << threads << ",\"pass\":" << pass
                      << ",\"folded\":" << (probe.folded ? "true" : "false")
                      << ",\"transposed_v\":" << (probe.transposed ? "true" : "false")
                      << ",\"calls\":" << calls << ",\"elapsed_ms\":" << elapsed
                      << ",\"mean_run_us\":" << elapsed * 1000 / calls
                      << ",\"reference_max_abs\":" << ref_errors[index]
                      << ",\"layout_max_abs\":" << layout_errors[index]
                      << ",\"input_digest\":\"" << before << "\",\"audit\":"
                      << probe.audit << "}\n" << std::flush;
          }
        }
      }
    }
    std::cerr << "PASS all reference, runtime, input-integrity checks\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
