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

# Gemma4 Linux memory comparison: LM-XNNPACK at capacity 8,448

**September 23, 2026.** On this Intel Core i9-12900K Linux desktop,
LiteRT-LM/XNNPACK with normal prefill selection used **5.59 GiB peak RSS** for
a 1,024-token prompt and capacity 8,448. Optimized native used **2.00 GiB**.
LM therefore used **2.79 times** the peak memory, a **3.59 GiB** difference.
Native with its shared-workspace and compact-INT2 flags left at their disabled
defaults used **2.79 GiB**, so LM was also about twice that configuration.

This is a new Linux/XNNPACK experiment. It is separate from the
[Pixel 8 YNN allocation audit](gemma4_memory_allocation_audit.md) and the older
capacity-2,048 phone measurements in the
[prefill proposal](prefill_attention_runtime_proposal.md#8-what-the-existing-measurements-establish).

The follow-up [native runner improvement experiments](gemma4_native_runner_improvements.md)
use this optimized native configuration as their baseline and test releasing
packing-only sources and caching stage shapes. Those gains are additional to
the shared-workspace/compact-INT2 settings measured here.

## Matched workload and meaning of defaults

All four processes ran sequentially on Linux x86-64 with four threads pinned
to CPUs `0,2,4,6`, a 1,024-token prompt, 64 identical forced continuation
tokens, capacity 8,448, one warmup and one measured session, and reused
runtimes. Memory after decode is measured while both paths still own KV.
No phone inference was run; an existing native coefficient export was copied
from the Pixel and checked against all 1,394 saved payload/manifest hashes.

The LM input is the **original published Gemma4 E2B bundle**, SHA256
`ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42`.
There is no SDPA rewrite in this experiment. The native export identifies the
same source SHA. Both Linux executables were already built and were used
unchanged. Their hashes and every command are retained with the captures.

For LM, "default" here means its normal CPU/XNNPACK prefill policy with the
original signatures available. `enable_ynnpack=false`, four threads, capacity
8,448, an in-memory packed-weight cache, and repetition/diagnostic controls
are explicit. The effective settings report `prefill_chunk_size=-1`; this
static model uses the separate greedy signature scheduler, which selects
`prefill_1024` for this prompt. The Gemma4 engine's default disabling delegate
clustering is retained. Registration messages for the available YNN accelerator
do not mean it was selected: the requested and verified backend option is XNN.

For native, both cases keep the default 128-row prefill and 32-row KV alignment.
The optimization-defaults case omits `share_workspace` and
`preserve_static_int2`, whose actual executable defaults are both `false`.
The optimized case sets both to `true`, as in the earlier native comparison.
Thread count, capacity, runtime reuse, repetitions, and diagnostics are matched
explicitly; this is not a comparison of completely unspecified CLI invocations.

## Results

Values are MiB (`1 MiB = 1,048,576 bytes`). Each row is one process containing
one warmup followed by one measured session.

| Configuration | Process peak RSS | Current RSS after first prediction | Current RSS after last decode |
| --- | ---: | ---: | ---: |
| LM-XNNPACK, normal 1,024-row prefill selection | 5,728.9 | 5,417.5 | 5,417.6 |
| Native, compact INT2 and shared workspace enabled | 2,051.2 | 2,043.4 | 2,043.4 |
| Native, both optimization flags at disabled defaults | 2,852.0 | 2,852.8 | 2,852.8 |
| LM-XNNPACK, existing 128-row prefill selected as a control | 4,137.8 | 4,103.0 | 4,103.0 |

Peak RSS comes from the child process's `wait4` resource usage, cross-checked
with the driver's `getrusage` record. Current RSS comes from `smaps_rollup`.
These kernel counters differ slightly: in the native-defaults row the latter
is 0.75 MiB above the recorded high-water value. They are preserved as recorded.

LM's existing decode-boundary pause option allows an external snapshot just
after the first prediction and after the final forced decode. Native's
existing memory-report option records those same phases and additional setup
and prefill snapshots. No heap trimming or page-residency controls were applied.
These are memory diagnostics, not latency measurements or statistical medians.
Every observed process `VmSwap`/`Swap` value was zero.

The two native runs confirm that the requested configurations took effect:

| Native allocation counter | Optimized | Optimization defaults |
| --- | ---: | ---: |
| Stage runtimes | 151 | 151 |
| Unique XNN workspace count | 1 | 151 |
| Unique workspace capacity | 8.750 MiB | 252.054 MiB |
| Packed-weight capacity | 751.875 MiB | 1,021.875 MiB |
| Compact INT2 MLP operators | 60 | 0 |
| Persistent INT8 KV payload | 74.250 MiB | 74.250 MiB |
| Owning external stage buffers | 48.035 MiB | 48.035 MiB |

These allocation capacities are separate from RSS and should not be added
to it or assumed to describe every resident page.

## What the smaller-prefill control establishes

A copied original bundle renames `prefill_1024` to `ignored_1024` in the
signature and subgraph names. This causes the existing scheduler to choose
`prefill_128`. The byte comparison verifies that only **14 bytes** changed;
all weights, operators, subgraphs, and section offsets are preserved.

With the same LM executable and XNN backend, this lowers:

- Peak RSS by **1,591.1 MiB**, from 5.59 to 4.04 GiB.
- Current RSS after decode by **1,314.7 MiB**, from 5.29 to 4.01 GiB.

The retained difference includes 776.2 MiB less anonymous memory and
538.4 MiB less non-anonymous memory. In the saved mapping snapshots, resident
model-source pages fall from 902.3 to 363.6 MiB, while the XNN packed-weight
memfd mappings have the same 753.4 MiB RSS. Thus the measured process delta
must not all be labeled attention scratch or TFLite arena savings.

The XNNPACK weight-cache path already requests reclamation of original weight
pages after packing using `MarkMemoryNotNeeded` (`MADV_PAGEOUT` on Linux).
Remaining resident model pages do not prove that no such reclamation was
attempted. The [native improvement report](gemma4_native_runner_improvements.md#litert-lm-already-has-a-related-source-weight-reclamation-path)
documents the source/binary check and distinguishes this from native's extra
366 MiB of heap conversion sources.

Even with smaller prefill, LM-XNNPACK remains above native in this capture.
This experiment establishes the memory ordering and the effect of prefill
selection; it does not provide a complete allocation-owner decomposition of
the remaining gap. The phone audit's YNN arena/workspace numbers cannot be
substituted for this XNN backend on another architecture.

## Validation and artifacts

### Follow-up: effect of `MADV_PAGEOUT`

A matched control on the same desktop confirms that the original LM memory
measurements **already include roughly 739 MiB of source-weight reclamation**.
The same archived executable was run with a small `LD_PRELOAD` shim that either
passes `MADV_PAGEOUT` through to the kernel or makes just that advice a successful
no-op. Other advice values, mappings, coefficients, runtime options, kernels,
and the capture protocol are unchanged. Both sides use the same shim.

| Prefill selection | Pageout | Peak RSS, MiB | Final-decode RSS, MiB |
| --- | --- | ---: | ---: |
| 1,024 rows | Enabled | 5,734.15 | 5,422.99 |
| 1,024 rows | Suppressed | 6,473.63 | 6,162.41 |
| 128 rows | Enabled | 4,138.13 | 4,103.07 |
| 128 rows | Suppressed | 4,877.11 | 4,842.47 |

Each configuration issues 277 requests covering 739.543 MiB of unique original
model-file ranges. All actual syscalls in the enabled runs succeed. Suppressing
them increases final-decode RSS by 739.42 MiB with 1,024-row prefill and 739.40
MiB with 128-row prefill; anonymous memory is unchanged within 4 KiB. Source-file
RSS accounts for the increase. The 1,024-row enabled case was repeated after
the suppressed case and returned to 5,423.01 MiB, within 0.02 MiB of its first
final-decode capture. Every run matches all 65 predictions and token histories,
including warmups, and no process swaps. Full logits were not dumped.

This has three implications for the earlier comparison:

- The reported LM RSS/peak values are actual measurements with source-page
  reclamation active. Do not subtract the source-weight size again.
- The similar benefit in both prefill configurations means their RSS difference
  cannot be explained as one having pageout enabled and the other lacking it.
  Remaining source-file residency also varies, so that difference still must
  not all be attributed to attention scratch.
- File mappings stay valid and pages can be reloaded. Retained model mappings
  do not demonstrate that all original weights stay resident. The native
  366 MiB improvement releases its additional heap conversion buffers and is
  a different allocation/lifetime change.

These are new diagnostic captures, not replacements for the earlier raw
numbers: the new default-prefill enabled capture is about 5.35 MiB above the
earlier one, primarily in source-file residency. The causal comparison uses
enabled/suppressed captures within this control. No latency conclusion is drawn.

Artifacts are in `/data/bt/tmp/gemma4-lm-pageout-20260923`:
[summary](/data/bt/tmp/gemma4-lm-pageout-20260923/summary.json),
[preload source](/data/bt/tmp/gemma4-lm-pageout-20260923/pageout_probe.c),
[capture controller](/data/bt/tmp/gemma4-lm-pageout-20260923/run_capture.py), and
[validator](/data/bt/tmp/gemma4-lm-pageout-20260923/analyze.py).
The shim was built with `cc -O2 -std=c11 -fPIC -shared -Wall -Wextra -Werror`;
its source/library hashes and injected command are recorded per capture. For a
fresh run, set `GEMMA_PAGEOUT_MODE=pass` or `suppress` and invoke the controller
with an unused `lm-xnn-rN` or `lm-xnn128-rN` capture name.

### Original comparison validation

All four processes completed both sessions. All **65 recorded argmax IDs
match across all four configurations**, and warmup predictions repeat the
measured-session predictions. Prompt IDs, forced inputs, positions, context
lengths, thread count, and capacity are checked. This is prediction/history
validation; full-logit numerical parity and model-quality evaluation were
not rerun for these memory captures.

- [Machine-readable results and validation][summary].
- [Capture controller][controller] and [analysis/validation script][analysis].
- [Smaller-prefill control and byte validation][control].
- [Native coefficient checksum validation][payload].
- [LM default capture][lm], [optimized native capture][native],
  [native optimization-defaults capture][defaults], and [LM 128-row capture][small].
- [Preserved LM Linux build recipe][lm-build] and
  [native runner defaults and ownership](../examples/gemma4/native/driver.cc).

Each capture directory contains the command, binary/model/fixture hashes,
host configuration, process log, raw output JSON, sampled `/proc` memory,
and final process resource usage. To repeat a case, use a fresh capture name
such as `python3 run_capture.py lm-xnn-r1` from the artifact directory; the
controller refuses to overwrite existing captures. Production runtime source
was not changed for this experiment.

[summary]: /data/bt/tmp/gemma4-linux-xnn-memory-20260923/summary.json
[controller]: /data/bt/tmp/gemma4-linux-xnn-memory-20260923/run_capture.py
[analysis]: /data/bt/tmp/gemma4-linux-xnn-memory-20260923/analyze.py
[control]: /data/bt/tmp/gemma4-linux-xnn-memory-20260923/prefill128-control.json
[payload]: /data/bt/tmp/gemma4-linux-xnn-memory-20260923/native-payload-validation.json
[lm]: /data/bt/tmp/gemma4-linux-xnn-memory-20260923/lm-xnn-r0
[native]: /data/bt/tmp/gemma4-linux-xnn-memory-20260923/native-optimized-r0
[defaults]: /data/bt/tmp/gemma4-linux-xnn-memory-20260923/native-flag-defaults-r0
[small]: /data/bt/tmp/gemma4-linux-xnn-memory-20260923/lm-xnn128-r0
[lm-build]: /data/bt/tmp/gemma4-ynn-sdpa-export-20260916/runtime/host-build.json
