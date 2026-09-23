# Native Gemma4 development consolidated into LiteRT

The development home is now LiteRT's `tensor/` tree. The optimized matched-bundle
runner, shared Gemma4/mobile-checkpoint fixes, standalone builds, regression
tests, fixture preparation and saved-output comparison tools are available
here. The original XNNPACK checkout and external experiments are preserved as
fallback evidence; the new native build does not compile from them or link
retained archives from them.

This is a local integration, not a claim that all changes have merged into
upstream. Start with the [native runner overview](examples/gemma4/native/README.md)
and [build/test instructions](standalone/README.md).

## Upstream audit

Audited on 2026-09-15 after fetching the upstream branches:

| Source | Revision |
| --- | --- |
| Original XNNPACK checkout | `bf3ee43b63070284f85a4a882f298b6ca5273f01` plus preserved local changes |
| LiteRT checkout HEAD before migration | `345852646f25bc3e8d3935fa589243bad8bc5288` |
| Latest fetched LiteRT upstream/main | `9ca88353f966a6d602ed529e7ea349857b7f59a8` |

LiteRT HEAD already contained that upstream main, and its `tensor/` tree was
byte-identical to upstream before integration. Of 61 shared core C++ files,
58 were identical after normalizing repository packaging. The remaining
three were the local `XnnpackRunner` runtime-flags feature. No missing local
core kernel optimization was identified.

Five Gemma4 fixes were still missing from main: logical INT4 embedding width,
explicit separate LM head, generation-limit accounting, single-token prefill,
and avoiding an explicit singleton KV-head broadcast. Some had published
branches; branch existence did not establish that main included them. The
previous decode-mask binding fix was already upstream and was preserved.

Both the audited normal LiteRT Bazel and integrated CMake dependency declarations
pin XNNPACK `d89ef6669a14db203b3b7935b1b3862cb63fb6df`. That revision includes
QC2 fully connected operators, shared runtime workspace, FP32/QCINT8 BMM and
the corresponding dequantization fusion. The standalone migration build pins
the frozen runner's exact `bf3ee43...` revision to keep relocation separate
from a dependency upgrade. Native diagnostic code uses private XNNPACK structs,
so later upgrades need validation even when public APIs are compatible.

## Where the work lives

| Previous local work | Current location and integration |
| --- | --- |
| Tensor core and common/XNNPACK conversion | Existing upstream `tensor/`, `backends/` and `runners/` implementations are reused. |
| Optional runtime flags | `runners/xnnpack/runner.{h,cc}` and move/lazy-creation/profiling regressions. |
| Five small Gemma4 fixes | Existing `examples/gemma4` graph, weights, driver and helpers, with regression coverage. |
| Mobile compressed-tensors parsing and scientific notation | Shared `examples/utils/safetensor_loader` and `minijson`, preserving the upstream legacy loader and normal tracing. |
| BF16/FP16 lazy embedding rows, INT4 validation, static activation scales | Shared Gemma4 embedding and mobile FC helpers/tests. |
| Raw-CT diagnostics and reference tools | Standard `examples/gemma4` driver and Python tools; upstream Perfetto, tokenizer and disk cache support remain available. |
| Fast external staged runner and overlays | Explicit `examples/gemma4/native` package, with private matched-bundle graph helpers and extracted driver utilities. |
| Retained-archive Android build script | Replaced by source-based `standalone/CMakeLists.txt` and `examples.cmake`, plus native Bazel targets. |
| Bundle export, fixtures, logits/cache comparisons | Portable `examples/gemma4/native/tools` and `fixtures`. Large weights/captures are data, not source dependencies. |
| Android test and migration comparison launchers | `standalone/run_android_tests.py` and `compare_native_android.py`; devices are selected explicitly. |
| Local YNNPACK attention-layout benchmark | Source, original patch and license under `experiments/ynnpack_attention_layout`; separate from the native runner. |
| Developer READMEs and historical investigations | Missing READMEs migrated; dated reports and original snapshots under `experiments/history`. |

The normal raw-CT graph and optimized matched-bundle graph have distinct
numerical contracts. Raw CT retains its FP32-activation output-head helper and
growing FP32 KV workflow. The fast path retains the published bundle's QD8/QC2
head and INT8 KV quantization. Sharing their different head/cache policies would
change the comparison. Private native helpers keep those differences explicit
while reusing the common Tensor API, configuration, embeddings and utilities.

## Protected optimizations

| Optimization | Preserved mechanism and verification |
| --- | --- |
| Work proportional to active context | Token-major INT8 bank; views bounded to valid global prefix or local window, aligned to 32 rows. Attention runtime audit checks actual operator types and dimensions. |
| No full FP32 KV materialization for attention | FP32 queries/probabilities × INT8 K/V BMM fusion; unit tests and full-logit runs audit the executed attention operators. QS8/QC8 adapters can still copy INT8 bytes. |
| Reusable chunked prefill | One reusable 128-row prefill signature invoked repeatedly; the final prompt token runs through decode for first logits. Capacity is separate from prefill rows. |
| Cache-only prefix work | Prefill stops after the final distinct KV owner; decode executes all 35 layers. The 15 owners have 30 persistent K/V buffers. |
| Compact original INT2 weights | 60 static MLP weights/operators, 283,115,520 coefficient bytes rather than 566,231,040 widened bytes; original codes/scales retained. |
| Shared packed weights and worker pool | Stage runners borrow resources owned by the encompassing runtime; lifetime tests check creation, reuse and destruction. |
| Shared scratch without corrupting stage outputs | Serialized stage invocation, explicit externally owned outputs and common XNN workspace; allocation/ownership regression tests retained. |
| Preserved numerical behavior | Exact source dependency and Android compiler configuration, `-ffp-contract=off`, fixed histories and complete-vocabulary same-phone comparisons. |

## Validation

The initial consolidation used the fresh standalone Android binary SHA256
`6530b9fb548a9352c1917a9670bf747e4bdc80fb8a90b4c65b00d5b788d6f02e`.
The frozen reference SHA256 was
`fbe9897ce1e6bf05ca00536077dbad255b57204d4bf9d5ce666bb38064df6f74`.
Both use the published E2B matched bundle, capacity 8448, four threads,
128-row prefill, alignment 32, compact INT2 and shared workspace.

| Check | Executed result |
| --- | --- |
| Fresh standalone Linux and Android builds | Passed; build commands contain no old XNNPACK/experiment source or retained-archive paths. Android links only libc, libm, libdl and liblog dynamically. |
| CMake host tests | All 35 test executables passed; the final four restored attention cases also passed in a targeted rerun. |
| Android unit tests on TECNO and Pixel 8 | All 35 executables passed on each phone. Final inventory: 636 checks passed, 26 existing skips, zero failures per phone. |
| Bazel normal Gemma4 demo | Compile/link passed, including preserved Perfetto and TFLite disk-cache integration. |
| Bazel native target and tests | Compile/link and 6/6 synthetic tests passed with bf3 override and global `--cxxopt=-ffp-contract=off`. |
| Same-phone full-model comparison | **75/75 complete vocabulary vectors bitwise identical on each phone**, 150 total, spanning 8/128/1024/4096-token prompts and forced decode. Actual attention types/extents and compact-weight counts audited. |
| Compact INT2 loader, offline | All 60 matrices remain compact; 283,115,520 coefficient bytes plus 960 padding bytes, and all widened MLP mappings released. No inference. |
| Relocated bundle tools | Fresh schema/inventory/trace/KV summary generated; independent readback validated all 1,393 files, 5,030,936,576 weight codes, scales/constants and four graph signatures against original data. No new multi-GB export was written. |
| Portable comparison/fixture tools | 18 comparator tests passed; all five default fixtures regenerated byte for byte, with checkout-independent manifest metadata. |
| Source and dependency audit | 1,361 dependency compilation entries match source/compiler options; all executable, allocated non-code and relevant relocation sections match in 1,343 linked objects. |

The 26 Android skips are inherited generic backend/debugger cases, including
unimplemented operators and an existing quantized-FC numerical skip. They are
not counted as passes. Full-model checks above provide separate evidence for
the actual native E2B graph. The extent-rounding test explores other alignments
and bounds discrepancies; it does not establish bitwise equality at every
experimental alignment.

No Linux full-model inference or performance test ran. Samsung was disconnected
and excluded. Host work comprised builds, synthetic tests and offline data/code
checks. The normal raw-CT demo was compiled but its full-checkpoint generation
workflow was not rerun; full-model parity applies to the tested standalone
matched-bundle binary, not an arbitrary Bazel configuration or another checkpoint.

### Performance preservation

Same-phone frozen → consolidated medians, with one warmup and three measured
sessions per capture, 64 forced decode inputs, and no diagnostic dumps. Prefill
includes the first vocabulary prediction. Initial captures used alternating
binary order and 30-second cooldowns; all samples remain in the
[performance summary](experiments/history/migration-20260915/validation/performance-summary.json).

| Phone | Prompt | Prefill seconds: frozen → consolidated | Decode tokens/s: frozen → consolidated |
| --- | ---: | ---: | ---: |
| TECNO | 128 | 0.373 → 0.373 | 27.97 → 27.38 |
| TECNO | 1024 | 3.318 → 3.315 | 25.16 → 25.01 |
| TECNO | 4096 | 16.401 → 16.311 | 19.43 → 20.05 |
| Pixel 8 | 128 | 0.579 → 0.743 | 22.82 → 20.81 |
| Pixel 8 | 1024 | 5.087 → 5.704 | 19.83 → 20.40 |
| Pixel 8 | 4096 | 19.656 → 18.307 | 18.38 → 17.98 |

TECNO reproduces the frozen performance closely: prefill differences are under
0.6%, and decode spans −2.1% to +3.2%. Pixel 8 had substantial run-to-run spread:
the frozen 1024-token prefill alone ranged from 4.18 to 6.05 seconds in the first
capture. The short cases were remeasured with reversed binary order and longer
pauses:

| Pixel recheck | Prefill seconds: frozen → consolidated | Decode tokens/s: frozen → consolidated |
| --- | ---: | ---: |
| 128 tokens | 0.564 → 0.568 | 22.30 → 22.83 |
| 1024 tokens | 4.927 → 4.036 | 19.76 → 21.42 |

The Pixel slowdown did not repeat. These captures do not establish a migration
speedup either. Source/operator audits and identical dependency instructions
provide no evidence of a lost optimization; precise Pixel performance remains
sensitive to conditions not controlled by these measurements. The dependency
comparison excludes a kernel/compiler downgrade but does not isolate device
scheduling, frequencies, caches or final executable layout.

Compact machine-readable evidence is preserved under
[validation](experiments/history/migration-20260915/validation/), including the
[full-logit captures](experiments/history/migration-20260915/validation/capture-index.json),
[final Android unit inventory](experiments/history/migration-20260915/validation/final-phone-unit-inventory.json),
and [compiled dependency comparison](experiments/history/migration-20260915/validation/compiled-dependency-comparison.md).
Raw logs, command manifests and captures remain in the recorded local artifact
directories. No claim of exported model quality beyond the tested numerical
contracts is made.

## Preservation and development boundaries

All 142 inventoried XNNPACK source/documentation files still match their
pre-migration hashes. Every inventoried path has a current LiteRT destination
or explicit historical preservation mapping in
[migration-map.json](experiments/history/migration-20260915/migration-map.json).
The exact original source snapshots include the external native experiments;
[snapshot hashes](experiments/history/migration-20260915/snapshot-sha256.json)
allow later recovery and comparison. These archives are historical fallback,
not competing active implementations.

Existing unrelated LiteRT changes are preserved. No branch switch, rebase,
commit, push, GitHub comment or deletion of the old source checkout was needed.
Continue native development in this tree and treat the original runner as a
frozen reference.

Build outputs are accessible through `.native-tensor-build/`. On this machine
that path points to `/data/bt/tmp/gemma4-litert-build-storage-20260915/native`
because the system disk filled during dependency downloads. The task's
isolated Bazel cache was similarly relocated to the data volume. The source
and build definitions remain in LiteRT; these generated cache locations are
not prerequisites for a fresh build.

The independent YNNPACK layout experiment is preserved and its patch checked
against the recorded base, but was not built or benchmarked during this move.
Historical reports retain their original measurements and paths; the current
instructions are the standalone and native READMEs.
