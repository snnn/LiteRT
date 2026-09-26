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

# Production LiteRT versus native XNNPACK on three phones

This directory preserves the exact campaign logic used for the TECNO, Pixel 8,
and Samsung comparison. The controllers now locate the main repository from
`__file__`; they no longer require the deleted PR worktree. Their original
model, binary, capture, device-serial, and affinity settings remain explicit.

- `benchmark_final.py` uses the final inclusive token timer.
- `benchmark_phones.py` preserves the preceding campaign controller.
- `analyze_final.py` validates all predictions and summarizes the final captures.
- `write_report.py` regenerates the production example's phone report.
- `results.json`, `historical-native.json`, and the identity/hash manifests
  preserve the small reference records.

Raw captures and binaries remain in
`/data/bt/tmp/gemma4-litert-production-phones-20260924`. Analysis scripts default
to that directory. The controllers refuse to overwrite existing per-run capture
directories. To run another campaign, choose fresh host/phone artifact paths and
check the device configuration; do not reuse the existing capture destinations.

Use the [phone reproduction guide](../../../local_docs/gemma4_phone_reproduction/README.md)
for a portable new comparison with LiteRT-LM. The
[production phone report](../../examples/gemma4/litert/PHONE_BENCHMARKS.md)
describes the historical native comparison and its limitations.
