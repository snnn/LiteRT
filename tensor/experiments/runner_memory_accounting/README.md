<!--
Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Runner memory accounting diagnostics

`analyze.py` preserves the published-container accounting, `/proc/PID/smaps`
sampling, and isolated embedding-layout probes used by the
[memory analysis](../../../local_docs/gemma4_e2b_model_size_vs_memory.md).
It imports the checked-in LiteRT-LM header schema from the production bundle
tools and requires the `ai_edge_litert` Python schema, FlatBuffers, and NumPy.
It no longer imports code from the separate SDPA candidate experiment.

The script retains the original model, prepared-bundle, and runner artifact
paths for reproducibility. Its `static` command retrieves the current public
container header and inspects the archived source model. Set
`GEMMA4_MEMORY_OUTPUT_DIR` to a fresh directory for another capture, then run
`analyze.py static`, `analyze.py resident`, and `analyze.py lookup layer` /
`analyze.py lookup token` as needed. `resident` samples memory and its timings
must not be reported as performance measurements. The lookup probes are not
full model inference runs.

`madvise_audit.c` is the original Linux `LD_PRELOAD` observer. It counts pageout
requests and forwards them to libc unchanged. Build it with
`cc -shared -fPIC -std=c11 madvise_audit.c -ldl -o /tmp/madvise_audit.so`, then set
`LD_PRELOAD=/tmp/madvise_audit.so` for a separate diagnostic process.

The JSON files preserve the original accounting, lookup, pageout, production
correctness, timing, and test summaries. In particular, `validation.json`
records the limitation of the later cross-platform argmax comparison. Do not
infer new timing or cross-platform numerical guarantees from these diagnostics.
