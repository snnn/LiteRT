# Historical native-runner reports

These files preserve earlier investigations and their original commands/results. The 2026-09-15 prefix is the archive date, not a claim that each experiment ran that day. Use [the current tensor guide](../../README.md), [standalone build instructions](../../standalone/README.md) and [native runner documentation](../../examples/gemma4/native/README.md) for current development.

- [Native runner improvements](2026-09-15-native-runner-improvements.md): optimization and validation history from the original local experiment.
- [Gemma4 XNNPACK README snapshot](2026-09-15-gemma4-xnnpack-readme.md): original safetensors driver instructions and numerical investigations.
- [FP16 runtime rewrite investigation](2026-09-15-fp16-runtime-rewrite-and-litert-integration.md): July 2026 source analysis.
- [Migration source snapshots](migration-20260915/): frozen source and manifests retained during consolidation.
- [Documentation provenance](docs-provenance.json): original locations and hashes.

The three report snapshots are byte-for-byte originals. Their historical relative links and absolute paths refer to the original checkouts and can be stale after relocation; they are preserved as evidence, not current instructions. Adapted developer READMEs link to current LiteRT paths.

[XNNPACK's BSD license](XNNPACK_LICENSE.BSD) is retained for XNNPACK-derived content. Files carrying their own Apache-2.0 notices retain those notices; the license archive does not relicense LiteRT source.
