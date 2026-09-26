#!/usr/bin/env python3
# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inspect or copy the supported Gemma4 E2B bundle without prefill_1024.

Removal only edits the SignatureDefs offset vector. It preserves section sizes,
subgraphs, weights, and all absolute/external-buffer offsets. This is a narrow
reproduction helper, not a general .litertlm repacker or graph-pruning tool.
"""

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import struct
import sys

SOURCE_SHA256 = "ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42"
SOURCE_BYTES = 2583085056


def inspect(view, model_class, header_class):
    if view[:8] != b"LITERTLM":
        raise ValueError("Expected a LiteRT-LM bundle")
    header_end = struct.unpack_from("<Q", view, 24)[0]
    if not 32 < header_end <= len(view):
        raise ValueError("Invalid bundle header")
    sections = header_class.GetRootAs(view, 32).SectionMetadata()
    report, models = [], []
    for i in range(sections.ObjectsLength()):
        section = sections.Objects(i)
        if section.DataType() != 3:  # AnySectionDataType.TFLiteModel
            continue
        begin, end = section.BeginOffset(), section.EndOffset()
        if not header_end <= begin < end <= len(view):
            raise ValueError("Invalid TFLite section bounds")
        model = model_class.GetRootAs(view, begin)
        signatures = [
            {
                "name": model.SignatureDefs(j).SignatureKey().decode(),
                "subgraph": model.SignatureDefs(j).SubgraphIndex(),
            }
            for j in range(model.SignatureDefsLength())
        ]
        report.append({
            "section": i, "begin": begin, "end": end,
            "subgraphs": model.SubgraphsLength(), "signatures": signatures,
        })
        models.append(model)
    return report, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--schema-dir", required=True, type=Path)
    parser.add_argument("--litert-src", required=True, type=Path)
    parser.add_argument("--mode", choices=("inspect", "remove", "rename"), default="remove")
    args = parser.parse_args()
    sys.path.insert(0, str(args.schema_dir.resolve()))
    sys.path.insert(0, str(args.litert_src.resolve() /
                          "tensor/examples/gemma4/litert/tools/schema"))
    from tflite.Model import Model
    from litertlm_header_schema_py_generated import LiteRTLMMetaData

    with args.input.open("rb") as src, mmap.mmap(
        src.fileno(), 0, access=mmap.ACCESS_READ
    ) as view:
        before, models = inspect(view, Model, LiteRTLMMetaData)
        if args.mode == "inspect":
            print(json.dumps(before, indent=2))
            return
        if args.output is None or args.output.exists():
            raise ValueError("Use --output with a NEW file path")
        source_hash = hashlib.file_digest(src, "sha256").hexdigest()
        if len(view) != SOURCE_BYTES or source_hash != SOURCE_SHA256:
            raise ValueError("Editing is restricted to the documented Gemma4 E2B bundle")
        candidates = [
            i for i, section in enumerate(before)
            if "prefill_1024" in [s["name"] for s in section["signatures"]]
        ]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one model containing prefill_1024")
        index = candidates[0]
        names = [s["name"] for s in before[index]["signatures"]]
        if names != ["decode", "prefill_1024", "prefill_128", "verify"]:
            raise ValueError(f"Unexpected signature layout: {names}")
        patches = []
        if args.mode == "remove":
            model = models[index]
            # Model.signature_defs is field 7: vtable offset 4 + 2 * 7 = 18.
            field = model._tab.Offset(18)
            start = model._tab.Vector(field)
            slots = [start + 4 * j for j in range(len(names))]
            targets = [slot + struct.unpack_from("<I", view, slot)[0] for slot in slots]
            if targets != [model.SignatureDefs(j)._tab.Pos for j in range(len(names))]:
                raise ValueError("Signature vector layout disagrees with generated schema")
            keep = [j for j, name in enumerate(names) if name != "prefill_1024"]
            patches.append((start - 4, struct.pack("<I", len(keep))))
            for new_index, old_index in enumerate(keep):
                slot = start + 4 * new_index
                patches.append((slot, struct.pack("<I", targets[old_index] - slot)))
        else:
            # Reproduce the historical memory control exactly: rename both the
            # SignatureDef key and matching subgraph name, without pruning.
            old, new = b"prefill_1024\0", b"ignored_1024\0"
            position = 0
            while (position := view.find(old, position)) >= 0:
                patches.append((position, new))
                position += len(old)
            if len(patches) != 2:
                raise ValueError("Expected exactly two historical rename locations")
        expected_changes = {}
        for position, data in patches:
            for j, value in enumerate(data):
                if view[position + j] != value:
                    expected_changes[position + j] = value

    # Exclusive creation prevents accidentally replacing a model or a previous run.
    with args.input.open("rb") as src, args.output.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    with args.output.open("r+b") as dst:
        for position, data in patches:
            dst.seek(position)
            dst.write(data)

    # Verify the complete output against the source plus the planned byte edits.
    target_hash = hashlib.sha256()
    with args.input.open("rb") as src, args.output.open("rb") as dst:
        position = 0
        while chunk := src.read(8 * 1024 * 1024):
            expected = bytearray(chunk)
            for offset, value in expected_changes.items():
                if position <= offset < position + len(expected):
                    expected[offset - position] = value
            actual = dst.read(len(chunk))
            if actual != expected:
                raise ValueError("Output differs outside the planned signature edits")
            target_hash.update(actual)
            position += len(chunk)
        if dst.read(1):
            raise ValueError("Unexpected output size")
    with args.output.open("rb") as dst, mmap.mmap(
        dst.fileno(), 0, access=mmap.ACCESS_READ
    ) as view:
        after, _ = inspect(view, Model, LiteRTLMMetaData)
    expected_report = json.loads(json.dumps(before))
    signatures = expected_report[index]["signatures"]
    if args.mode == "remove":
        signatures[:] = [s for s in signatures if s["name"] != "prefill_1024"]
    else:
        for signature in signatures:
            if signature["name"] == "prefill_1024":
                signature["name"] = "ignored_1024"
    if after != expected_report:
        raise ValueError("Unexpected change to model sections or signatures")
    report = {
        "mode": args.mode, "source": str(args.input), "output": str(args.output),
        "source_sha256": source_hash, "output_sha256": target_hash.hexdigest(),
        "bytes": SOURCE_BYTES, "changed_byte_offsets": sorted(expected_changes),
        "all_other_bytes_identical": True, "before": before, "after": after,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
