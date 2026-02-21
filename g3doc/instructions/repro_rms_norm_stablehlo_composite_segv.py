#!/usr/bin/env python3
"""Minimal repro for StableHLO composite parse crash in LiteRT runtime.

This script has two phases:
1) Generate a minimal TFLite model (via litert-converter) that contains
   an RMSNorm composite pattern.
2) Attempt to load the model with ai_edge_litert CompiledModel.

On affected builds, phase (2) crashes with SIGSEGV in
`tflite::ParseStablehloComposite`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _add_converter_repo(converter_repo: Path) -> None:
    root = str(converter_repo.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def generate_model(out_path: Path, converter_repo: Path) -> None:
    _add_converter_repo(converter_repo)

    import torch
    from mlir_backend.importer.exported_program_importer import (
        ImportConfig,
        import_exported_program,
    )
    from mlir_backend.passes.legalize_to_tfl import legalize_aten_to_tfl
    from mlir_backend.passes.rewrite_l2_norm import rewrite_l2_norm_to_builtin
    from mlir_backend.passes.verify import verify_no_unsupported_ops
    from mlir_backend.translate.tflite_flatbuffer import (
        TranslationConfig,
        serialize_tflite,
    )

    class RmsNormModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(8))

        def forward(self, x):
            return torch.ops.aten.rms_norm.default(x, [8], self.w, 1.0e-6)

    m = RmsNormModel().eval()
    x = torch.randn(2, 8, dtype=torch.float32)
    exported = torch.export.export(m, (x,), strict=False)
    module = import_exported_program(
        exported,
        ImportConfig(model_name="rms_norm_repro", input_shape_nchw=tuple(int(d) for d in x.shape)),
    )
    module = legalize_aten_to_tfl(module)
    module = rewrite_l2_norm_to_builtin(module)
    verify_no_unsupported_ops(module)

    data, meta = serialize_tflite(module, TranslationConfig(strict=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(f"[generate] wrote {out_path}")
    print(f"[generate] valid_flatbuffer={bool(meta.get('is_valid_tflite_flatbuffer', False))}")
    tfl_ops = sorted({op.name for op in module.ops if op.stage == "tfl"})
    print(f"[generate] tfl_ops={tfl_ops}")


def load_model(model_path: Path) -> None:
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator

    print(f"[load] loading {model_path}")
    # On affected runtime builds, this call segfaults.
    _ = CompiledModel.from_file(str(model_path), hardware_accel=HardwareAccelerator.CPU)
    print("[load] success (no crash)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("generate", "load", "both"),
        default="both",
        help="generate model, load model, or both",
    )
    ap.add_argument(
        "--out",
        default="/tmp/rms_norm_stablehlo_composite_crash_repro.tflite",
        help="output model path for generate mode",
    )
    ap.add_argument(
        "--model",
        default="",
        help="model path for load mode (defaults to --out when empty)",
    )
    ap.add_argument(
        "--converter-repo",
        default="/home/chasun/src/litert-converter",
        help="path to litert-converter repo used for model generation",
    )
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    model_path = Path(args.model).resolve() if args.model else out_path
    converter_repo = Path(args.converter_repo).resolve()

    if args.mode in ("generate", "both"):
        generate_model(out_path, converter_repo)
    if args.mode in ("load", "both"):
        load_model(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
