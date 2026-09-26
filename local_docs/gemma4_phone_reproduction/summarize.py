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
"""Summarize matched production/LiteRT-LM 1024+64 phone benchmark captures."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as stats


def require(condition, message):
    if not condition:
        raise ValueError(message)


def session(label, decode, first, predictions, peak):
    require(len(decode) == 64 and len(predictions) == 65, f"Incomplete session: {label}")
    return {"id": label, "decode_ms": stats.mean(decode), "first_token_ms": first,
            "request_ms": first + sum(decode), "argmax": predictions,
            "peak_rss_mib": peak / 1024}


def production(directory):
    rows = [json.loads(line) for line in (directory / "timings.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    peak = max(row["peak_rss_kib"] for row in rows)
    for row in rows:
        require(row["case"] == "performance_1024_64", "Unexpected production fixture")
        if row["repetition"] >= 0:
            groups[row["repetition"]].append(row)
    require(sorted(groups) == [0, 1, 2], f"Expected three measured sessions in {directory}")
    result = []
    for repetition, values in sorted(groups.items()):
        values.sort(key=lambda v: v["step"])
        require([v["step"] for v in values] == list(range(65)), "Missing production steps")
        require([v["context"] for v in values] == list(range(1024, 1089)), "Wrong production contexts")
        result.append(session(f"{directory}:{repetition}",
                              [v["token_ms"] for v in values[1:]],
                              values[0]["prefix_ms"] + values[0]["token_ms"],
                              [v["argmax"] for v in values], peak))
    return result


def lm(directory, prompt, forced):
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest["status"] == "completed", "Incomplete LM process")
    require(manifest["timings_valid_for_benchmark"], "Trace/profile/pause run is not a benchmark")
    require(not manifest["enable_ynnpack"], "This comparison expects XNNPACK")
    require(manifest["num_threads"] == 4 and manifest["max_num_tokens"] == 8448,
            "LM threads/capacity differ from documented settings")
    require(manifest["reuse_runtimes"] and manifest["warmup_runs"] == 1 and
            manifest["measured_runs"] == 3, "LM warmup/reuse settings differ")
    result = []
    for path in sorted(directory.glob("*.run_*.json")):
        run = json.loads(path.read_text())
        require(run["status"] == "completed" and not run["warmup"], "Incomplete LM session")
        require(run["prompt_token_ids"] == prompt and run["forced_decode_token_ids"] == forced,
                "LM input IDs differ from bundled fixture")
        values = run["passes"]
        require([v["context_length_after"] for v in values] == list(range(1024, 1089)),
                "Wrong LM contexts")
        require(all(v["consumed_token_ids_verified"] for v in values), "Unverified LM history")
        require(all(v.get("logits_file") is None for v in values), "Do not summarize dump runs")
        result.append(session(str(path), [v["elapsed_ms"] for v in values[1:]],
                              values[0]["elapsed_ms"], [v["argmax_id"] for v in values],
                              manifest["peak_process_rss_kib"]))
    require(len(result) == 3, f"Expected three measured LM sessions in {directory}")
    return result


def summarize(rows):
    decode = stats.median(r["decode_ms"] for r in rows)
    return {
        "sessions": len(rows), "decode_ms": decode, "decode_tokens_per_second": 1000 / decode,
        "decode_session_range_ms": [min(r["decode_ms"] for r in rows), max(r["decode_ms"] for r in rows)],
        "first_token_ms": stats.median(r["first_token_ms"] for r in rows),
        "request_1024_plus_64_ms": stats.median(r["request_ms"] for r in rows),
        "peak_rss_mib": max(r["peak_rss_mib"] for r in rows),
        "per_session": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", required=True, nargs=2, type=Path, metavar="CAPTURE")
    parser.add_argument("--lm", required=True, nargs=2, type=Path, metavar="CAPTURE")
    parser.add_argument("--lm-label", default="LiteRT-LM XNNPACK, prefill_128")
    args = parser.parse_args()
    fixture = Path(__file__).parent / "fixtures/performance_1024_64.tsv"
    line = next(line for line in fixture.read_text().splitlines() if line and not line.startswith("#"))
    _, p, f = line.split("\t")
    prompt, forced = [int(x) for x in p.split(",")], [int(x) for x in f.split(",")]
    prod = [row for directory in args.production for row in production(directory)]
    baseline = [row for directory in args.lm for row in lm(directory, prompt, forced)]
    reference = prod[0]["argmax"]
    differences = []
    for label, rows in (("production", prod), (args.lm_label, baseline)):
        for row in rows:
            steps = [i for i, (a, b) in enumerate(zip(reference, row["argmax"])) if a != b]
            if steps:
                differences.append({"runner": label, "session": row["id"], "steps": steps})
    a, b = summarize(prod), summarize(baseline)
    print(json.dumps({
        "production": a, "lm_label": args.lm_label, "litert_lm": b,
        "production_decode_change_percent": 100 * (a["decode_ms"] / b["decode_ms"] - 1),
        "all_argmax_match": not differences, "argmax_differences": differences,
        "note": "Argmax agreement is a smoke check, not full-logit or quality equivalence.",
    }, indent=2))


if __name__ == "__main__":
    main()
