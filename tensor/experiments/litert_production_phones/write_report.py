import json
from pathlib import Path
root=Path('/data/bt/tmp/gemma4-litert-production-phones-20260924')
repo=Path(__file__).resolve().parents[3]
r=json.loads((root/'results.json').read_text())
assert len(r['devices'])==3
names={'tecno':'TECNO LJ9 / MT6897','pixel8':'Pixel 8 / Tensor G3','samsung':'Samsung SM-S937U1 / SM8750'}
text='''<!--
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

# Gemma4 production LiteRT runner on three Android phones

The production runner executes both complete signatures through LiteRT's public
`CompiledModel` API and existing XNNPACK delegate on all three phones. Samsung
decode matches the native runner; TECNO has a small decode cost. Pixel decode
remains slower and should not be described as complete performance parity.
The production runner reaches the first token sooner on all three phones.

## Matched configuration

Measurements on 2026-09-23 use the pinned E2B bundle documented in the
[example README](README.md): the same 1,024 prompt tokens and 64 forced decode
inputs, four CPU threads, KV capacity 8,448, chunk size 128, and KV alignment 32.
The fixture SHA-256 is
`01009f2da16a6c32e7288323029d019b49e1507022fd711f471c8001824b4eef`.
All 1,394 existing bundle files, including the manifest, were SHA-256 verified
on each phone; their 1,393 coefficient/scale/constant payload files match the
new preparation tools exactly. Model and executable hashes were also verified.

TECNO and Samsung use CPUs 4–7 (`taskset f0`); Pixel uses CPUs 5–8
(`taskset 1e0`). Pixel's separately measured five-core configuration is not
mixed into these four-thread comparisons. Device versions were Android 15
(TECNO), Android 17 (Pixel), and Android 16 (Samsung).

Each phone ran production/native/native/production in four separate processes.
Each process discarded one warmup session and measured three sessions. Tables
report medians of six session means, each averaging 64 forced decode calls.
The archived native binaries were rerun unchanged to check historical drift.
Production uses a device-generated, warm file-backed packed-weight cache.
Both signatures were fully delegated: 1,815 prefill nodes and 3,830 decode nodes,
one delegate partition per signature.

Each process starts after two eligible live HAL readings five seconds apart:
CPU at most 45 C, skin at most 34 C, SoC at most 39 C where exposed, and exposed
CPU cooling states zero. At least 15 seconds separates processes. Clocks and
thermal policies are unchanged; telemetry is sampled every five seconds.
This controls starting conditions, not DVFS or temperature throughout a run.

The new `token_ms` includes the runner call, logits access, finite validation,
and greedy selection, matching the native timer's scope. Output logging and
correctness dumps are excluded. `decode_ms` remains available for the runner
call alone. Initial captures that lacked `token_ms` are excluded from these
comparisons. First-token latency includes the prompt prefix and final prompt
token. Native also allocates a new KV bank per session; production reuses its
bank allocated during model setup. Model loading and compilation are excluded.

## Decode

Positive deltas mean production is slower. Historical records are from
September 23 for TECNO/Pixel and September 16 for Samsung.

| Phone | Historical native ms/token | Fresh native ms/token | Production ms/token | Change vs historical | Change vs fresh |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
for name,d in r['devices'].items():
    text+=f"| {names[name]} | {d['historical_native']['decode_ms']:.2f} | {d['native']['decode_ms']:.2f} | {d['production']['decode_ms']:.2f} | {d['production_vs_historical_decode_percent']:+.1f}% | {d['production_vs_fresh_native_decode_percent']:+.1f}% |\n"
text+='''
| Phone | Native session range ms/token | Production session range ms/token | Production minus native, block 0 / block 1 |
| --- | ---: | ---: | ---: |
'''
for name,d in r['devices'].items():
    a,b=d['native'],d['production']
    ar,br=a['decode_session_range_ms'],b['decode_session_range_ms']
    diffs=[100*(y['decode_ms']/x['decode_ms']-1) for x,y in zip(a['blocks'],b['blocks'])]
    text+=f"| {names[name]} | {ar[0]:.2f}–{ar[1]:.2f} | {br[0]:.2f}–{br[1]:.2f} | {diffs[0]:+.1f}% / {diffs[1]:+.1f}% |\n"
text+='''
These compare complete runners, including their memory layouts and worker
policies. The native controls use archived NDK r27 libraries; production was
built with Bazel and NDK r28c for API 35. The delegate sets
`XNN_FLAG_DONT_SPIN_WORKERS`, while the archived native stage runtimes use
runtime flags zero. The measurements do not isolate which difference causes
the Pixel decode gap.

## First token and complete request

| Phone | Historical native first token ms | Fresh native first token ms | Production first token ms | First-token change vs fresh | Complete 1,024+64 request change |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
for name,d in r['devices'].items():
    text+=f"| {names[name]} | {d['historical_native']['first_logits_ms']:.1f} | {d['native']['first_token_ms']:.1f} | {d['production']['first_token_ms']:.1f} | {d['production_vs_fresh_native_first_token_percent']:+.1f}% | {d['production_vs_fresh_native_request_percent']:+.1f}% |\n"
text+='''
The complete-request metric sums first-token latency and all 64 measured decode
calls within each session before taking the median. It is not a sum of
independently computed medians. Longer generations give decode latency more
weight, so the Pixel prefill improvement does not remove its decode regression.

## Residency and validation

| Phone | Native process peak RSS MiB | Production process peak RSS MiB | Maximum sampled CPU / skin C |
| --- | ---: | ---: | ---: |
'''
for name,d in r['devices'].items():
    text+=f"| {names[name]} | {d['native']['peak_rss_mib']:.1f} | {d['production']['peak_rss_mib']:.1f} | {d['max_sampled_cpu_c']:.1f} / {d['max_sampled_skin_c']:.1f} |\n"
text+='''
RSS includes startup and warmup. The production process has a separate export
lifetime and uses the standard file-backed cache, whose cold packing path can
page out source weights. Warm cache loading can avoid touching those source
pages. The native controls retain source and packed weights. The RSS difference
is not an isolated allocation saving from joining graphs or sharing scratch.
No sampled CPU cooling device reported a nonzero state; this does not establish
absence of DVFS or unexposed throttling.

All 65 argmax predictions in every measured session match the device's native
record: 780 checks per phone across the two implementations, 2,340 in total.
These are argmax comparisons, not full-logit byte comparisons on Android.
The separate desktop validation covers full logits and KV snapshots.

The Android smoke test found that Abseil strips flag names on Android by
default. The two example CLI targets now set `ABSL_FLAGS_STRIP_NAMES=0`.
The benchmark also adds `token_ms` to make generation timing boundaries explicit.
No attention or delegate algorithm changes were needed to run the model.

Raw commands, hashes, telemetry, per-session results, historical references,
and the analysis scripts are retained at
`/data/bt/tmp/gemma4-litert-production-phones-20260924` (`results.json` is the
machine-readable summary). The timed Android binary SHA-256 is
`30df08aa49ecd2daa74ef8aab3f9922493d0cc1869acdc923bcfb77b79438453`.
Build and model-preparation commands are in the [README](README.md#android).
'''
(repo/'tensor/examples/gemma4/litert/PHONE_BENCHMARKS.md').write_text(text)
print('Wrote phone report')
