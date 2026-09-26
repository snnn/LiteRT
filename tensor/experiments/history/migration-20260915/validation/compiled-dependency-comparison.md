# Android compiled dependency comparison

No dependency compilation downgrade was found between the frozen native runner and the consolidated LiteRT build. This report records the already completed read-only comparison; it does not add inference or performance measurements.

The complete paths, SHA256 values and methods are in [compiled-dependency-comparison.json](compiled-dependency-comparison.json).

## Compared inputs

- Old build: `/tmp/gemma4-local-port-nggiqf9x/android`.
- New build: `/data/home/chasun/src/LiteRT/.native-tensor-build/android`.
- New build resolves to `/data/bt/tmp/gemma4-litert-build-storage-20260915/native/android`.
- The old archive SHA256 values match the frozen native Android build manifest identified in the JSON.

All **1,361 source/target compilation entries** matched source SHA256 and compiler/non-path options: 942 XNNPACK, 380 KleidiAI, 36 cpuinfo (including its secondary target), and 3 pthreadpool. No source/target was missing or added.

## Compiled object results

| Archive | Objects with identical executable sections | Executable bytes | Allocated non-code sections | Relevant relocation sections |
| --- | ---: | ---: | ---: | ---: |
| `libXNNPACK.a` | 108 | 611,356 | 1,264 | 1,876 |
| `libxnnpack-microkernels-prod.a` | 834 | 1,053,284 | 1,646 | 1,025 |
| `libkleidiai.a` | 380 | 458,180 | 682 | 1,054 |
| `libcpuinfo.a` | 18 | 39,340 | 79 | 120 |
| `libpthreadpool.a` | 3 | 58,264 | 6 | 103 |
| **Total** | **1,343** | **2,220,424** | **3,677** | **4,178** |

Every one of the **6,591 executable sections** matched byte for byte. Kernel constants and other allocated non-code section contents/sizes also matched, as did relocation-section bytes targeting allocated code/data. Executable-section counts include empty sections. Whole archive hashes differ because this is a rebuilt archive; nonallocated/debug metadata was not exhaustively categorized.

## Method and limits

The checker parsed GNU archive members and ELF64 little-endian section tables directly. It paired duplicate object basenames by archive occurrence and compared section names, sizes and payloads. `SHT_NOBITS` sections were compared by declared size. It excluded debug-only relocation sections. Source/target command comparison retained compiler/architecture/optimization/feature options and excluded relocated include/source/output paths.

This is strong evidence against a lost XNNPACK, microkernel, KleidiAI, cpuinfo or pthreadpool compilation optimization. It does **not** compare the entire reorganized C++ driver or final executable addresses, and it does not rule out link-layout/cache effects or device frequency, thermal and scheduling variation. It does not identify the cause of the Pixel timing spread or establish statistical performance equivalence.
