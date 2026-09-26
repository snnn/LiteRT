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

# Native Gemma4 MLP/FC profiler

The [report](../../docs/gemma4_native_mlp_profile.md) attributes the approximately
17 ms decode post-attention interval to exact model FCs and smaller operators.
[results.json](results.json) preserves the validated summaries.

This is a Linux/Android diagnostic for the pinned XNNPACK revision and the
35-layer E2B model. It uses internal nanosecond profiling timestamps; it is
not a stable Tensor API or portable XNNPACK profiling implementation.

`build_profile.py` applies [driver.patch](driver.patch) to a copied baseline
driver, adds [post_profile.h](post_profile.h), and links the existing archived
libraries. It does not invoke CMake, rebuild dependencies, or edit production
sources. This preserves the previously audited kernels; ordinary repository
development should use the documented Bazelisk workflow.

The checked-in patch only enables profiling on decode post-attention stages.
The helper matches FC filter pointers to loaded weight names, requires all
35 stages and consistent timestamps, and collects only measured forced-decode
calls. Profile collection occurs after the stage wall timer. Whole-step
timings with the profiler enabled are marked diagnostic.

## Reproduction on the audited desktop

Use a fresh output directory; capture directories cannot be overwritten.
The baseline build directory and its `commands.json` come from
[the preceding lifetime experiment](../native_runner_lifetimes/README.md).

```bash
python3 tensor/experiments/native_mlp_profile/build_profile.py \
  --baseline /data/bt/tmp/gemma4-native-next-20260923/baseline \
  --out /data/bt/tmp/gemma4-native-mlp-profile-new

python3 tensor/experiments/native_mlp_profile/run_profile.py \
  --out /data/bt/tmp/gemma4-native-mlp-profile-new \
  --baseline /data/bt/tmp/gemma4-native-next-20260923/baseline/runner \
  --bundle /data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle

python3 tensor/experiments/native_mlp_profile/analyze.py \
  --out /data/bt/tmp/gemma4-native-mlp-profile-new \
  --reference /data/bt/tmp/gemma4-native-next-20260923/correctness-baseline/capture
```

The four sequential captures are unchanged baseline, profiling enabled,
profiling disabled in the diagnostic executable, and full-logit correctness.
Each warms up once. Timing/profile captures measure three 64-token sessions;
correctness measures one and compares all 65 vocabulary vectors, including
the first-logit vector, against the preceding baseline.

Sample and count CPU events on the unchanged baseline separately:

```bash
python3 tensor/experiments/native_mlp_profile/perf_decode.py \
  --out /data/bt/tmp/gemma4-native-mlp-perf-new \
  --binary /data/bt/tmp/gemma4-native-next-20260923/baseline/runner \
  --bundle /data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle

python3 tensor/experiments/native_mlp_profile/perf_decode.py \
  --mode stat --out /data/bt/tmp/gemma4-native-mlp-stat-new \
  --binary /data/bt/tmp/gemma4-native-next-20260923/baseline/runner \
  --bundle /data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle

python3 tensor/experiments/native_mlp_profile/analyze_perf.py \
  --record /data/bt/tmp/gemma4-native-mlp-perf-new \
  --stat /data/bt/tmp/gemma4-native-mlp-stat-new
```

The controller bridges the runner's existing decode markers to perf's control
FIFO. Events start disabled. It checks balanced markers, successful process
completion, and every control acknowledgement. The runner does not wait for
acknowledgements, so marker/acknowledgement timestamps are saved and boundary
samples are excluded by the analyzer. CPU samples include worker threads and
represent the whole decode, not just MLPs. The `cpu_core` PMU and affinity
`0,2,4,6` are specific to the tested desktop.

The reported raw operator captures are under
`/data/bt/tmp/gemma4-native-mlp-profile-20260923/v2`; finalized CPU captures are
in its sibling directories `perf-record-v2` and `perf-stat`. Earlier development
captures elsewhere in that root are not the reported results. The full
commands, binary/fixture/manifest hashes, logs, timestamps, and profiles remain
in those artifact directories.

## Follow-up thread sweep

`thread_sweep.py` tests two, four, and eight distinct P cores in two process
blocks with reversed order. Each process warms up once and measures three
sessions, using the unchanged baseline executable. It records exact affinity,
commands, hashes, and outputs, and checks every measured prediction/history.

```bash
python3 tensor/experiments/native_mlp_profile/thread_sweep.py \
  --out /data/bt/tmp/gemma4-native-thread-sweep-new \
  --binary /data/bt/tmp/gemma4-native-next-20260923/baseline/runner \
  --bundle /data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle

python3 tensor/experiments/native_mlp_profile/thread_sweep.py \
  --out /data/bt/tmp/gemma4-native-thread-sweep-new \
  --binary /data/bt/tmp/gemma4-native-next-20260923/baseline/runner \
  --bundle /data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle \
  --validate-only \
  --reference /data/bt/tmp/gemma4-native-next-20260923/correctness-baseline/capture
```

The second command runs separate full-logit diagnostics at two and eight
threads. Their timings are not benchmark results. The reported capture is
`/data/bt/tmp/gemma4-native-thread-sweep-20260923`; its validated summary is
also saved in [thread_results.json](thread_results.json).

## Phone follow-up

The [phone report](../../docs/gemma4_native_phone_optimizations.md) and
[phone_results.json](phone_results.json) contain the validated TECNO LJ9 and
Pixel 8 experiments. MTP is excluded. Scripts contain the exact device
serials, topology-specific masks, and retained bundle/fixture paths used here.
Use fresh host and phone output roots when rerunning: the controllers reject
existing capture directories. Change the `REMOTE` constant in
`phone_campaign.py` to a fresh phone path for a new campaign; `OLD` locates
the retained model assets. No device root or governor changes are needed.

Relink the Android baseline and probe using the archived Android build, then
build the operator profiler. The archive relink is deliberate; ordinary
development should use Bazelisk as required by the repository guidelines.

```bash
python3 tensor/experiments/native_runner_lifetimes/build_probe.py \
  --build .native-tensor-build/android \
  --out /data/bt/tmp/gemma4-native-phones-new/build

python3 tensor/experiments/native_mlp_profile/build_profile.py \
  --baseline /data/bt/tmp/gemma4-native-phones-new/build/baseline \
  --out /data/bt/tmp/gemma4-native-phones-new/build/profile

python3 tensor/experiments/native_mlp_profile/phone_campaign.py \
  --out /data/bt/tmp/gemma4-native-phones-new \
  --expected-hashes /data/bt/os/llama.cpp/tmp_models/gemma4_qat_q4_0_perf_graph_compare_20260914/native-adapter/matched-bundle.expected.sha256
```

Each phone serializes its captures. The main campaign runs reversed-order
thread sweeps, post-operator profiles, full-logit checks, and source-release
memory controls. The phones can run concurrently. Thermal gates use live HAL
readings; raw readings/frequencies are saved before and during captures.

After the main campaign, build/run the attention/head diagnostic, followed by
the separate CPU-sampling diagnostic:

```bash
python3 tensor/experiments/native_mlp_profile/build_phone_followup.py \
  --profile /data/bt/tmp/gemma4-native-phones-new/build/profile \
  --out /data/bt/tmp/gemma4-native-phones-new/build/followup

python3 tensor/experiments/native_mlp_profile/phone_followup.py \
  --out /data/bt/tmp/gemma4-native-phones-new

python3 tensor/experiments/native_mlp_profile/build_phone_followup.py \
  --profile /data/bt/tmp/gemma4-native-phones-new/build/profile \
  --out /data/bt/tmp/gemma4-native-phones-new/build/followup-cycles --timestamps

python3 tensor/experiments/native_mlp_profile/phone_cycles.py \
  --out /data/bt/tmp/gemma4-native-phones-new

python3 tensor/experiments/native_mlp_profile/analyze_phone_cycles.py \
  --out /data/bt/tmp/gemma4-native-phones-new \
  --simpleperf-dir /data/bt/os/android_sdk/ndk/27.3.13750724/simpleperf

python3 tensor/experiments/native_mlp_profile/analyze_phones.py \
  --out /data/bt/tmp/gemma4-native-phones-new
```

The CPU analyzer reconstructs measured operator intervals from absolute
monotonic invocation starts and per-operator durations. It filters out all
other samples, checks zero reported losses, and reports period-weighted user
cycles. Both follow-up profilers validate every full vocabulary vector against
the same phone's baseline. Instrumented/dump timings never enter the thread
benchmark table. The final analyzer also rechecks completion, history,
operator counts/timestamps, full logits, and memory-owner counters before
writing its summary. Raw captures used for the report are in
`/data/bt/tmp/gemma4-native-phones-20260923`.
