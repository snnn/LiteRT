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
"""Run a cooled production/LM/LM/production CPU comparison on one phone.

Requires the documented files in --remote, a primed production weight cache,
and Android thermalservice HAL temperature reporting. Never changes phone policy.
"""

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def parse_thermal(raw):
    if "Current temperatures from HAL:" not in raw:
        raise RuntimeError("No live HAL temperatures; adapt the gate explicitly for this phone")
    live = raw.split("Current temperatures from HAL:", 1)[1].split("Current cooling devices from HAL:", 1)[0]
    readings = [(float(v), int(t), n) for v, t, n in re.findall(
        r"Temperature\{mValue=([-\d.]+), mType=([-\d]+), mName=([^,]+), mStatus=\d+\}", live)]
    cpu_values = [v for v, t, _ in readings if t == 0]
    skin_values = [v for v, t, _ in readings if t == 3]
    if not cpu_values or not skin_values:
        raise RuntimeError("CPU/skin temperatures missing; adapt the gate explicitly for this phone")
    cpu, skin = max(cpu_values), max(skin_values)
    soc_values = [v for v, t, n in readings if t == 13 or n == "soc_therm"]
    soc = max(soc_values, default=None)
    cooling = {n: int(v) for v, n in re.findall(
        r"CoolingDevice\{mValue=(\d+), mType=2, mName=([^}]+)\}", raw)}
    return {"time": time.time(), "cpu_c": cpu, "skin_c": skin, "soc_c": soc,
            "cpu_cooling_states": cooling,
            "eligible": cpu <= 45 and skin <= 34 and (soc is None or soc <= 39) and not any(cooling.values()),
            "raw": raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lm-model", default="gemma4-prefill128.litertlm")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]+", args.mask):
        raise ValueError("Expected a hexadecimal CPU mask without 0x")
    args.output.mkdir(parents=True, exist_ok=False)
    adb = ["adb", "-s", args.serial]

    def shell(command, timeout=60):
        if not isinstance(command, str):
            command = shlex.join(map(str, command))
        return subprocess.check_output(adb + ["shell", command], text=True, timeout=timeout)

    def thermal():
        return parse_thermal(shell(["dumpsys", "thermalservice"]))

    def gate(directory, last_end):
        samples, consecutive = [], 0
        start = time.monotonic()
        while time.monotonic() - start < 600:
            sample = thermal()
            samples.append(sample)
            save(directory / "start-gate.json", samples)
            consecutive = consecutive + 1 if sample["eligible"] else 0
            if consecutive >= 2 and time.monotonic() - last_end >= 15:
                return
            print(f"Cooling: CPU {sample['cpu_c']:.1f} C, skin {sample['skin_c']:.1f} C", flush=True)
            time.sleep(5)
        raise RuntimeError("Temperature gate was not reached in ten minutes")

    save(args.output / "device.json", {
        "serial": args.serial, "mask": args.mask,
        "model": shell(["getprop", "ro.product.model"]).strip(),
        "soc": shell(["getprop", "ro.soc.model"]).strip(),
        "fingerprint": shell(["getprop", "ro.build.fingerprint"]).strip(),
        "meminfo": shell(["cat", "/proc/meminfo"]),
    })
    # Use a fresh remote name even if another result directory exists locally.
    run_root = args.remote.rstrip("/") + "/" + args.output.name
    shell(["test", "!", "-e", run_root])
    shell(["mkdir", "-p", run_root])
    base = args.remote.rstrip("/")
    last_end = 0
    for name, kind in (("production-b0", "production"), ("lm-b0", "lm"),
                       ("lm-b1", "lm"), ("production-b1", "production")):
        directory = args.output / name
        directory.mkdir()
        gate(directory, last_end)
        remote_output = run_root + "/" + name
        command = ["taskset", args.mask, base + ("/runner.android" if kind == "production" else "/litertlm.android"),
                   "--cases_file", base + "/performance_1024_64.tsv", "--output_dir", remote_output,
                   "--num_threads", "4", "--warmup_runs", "1", "--measured_runs", "3"]
        if kind == "production":
            command += ["--bundle_dir", base + "/weights", "--model_path", base + "/gemma4.tflite",
                        "--cache_capacity", "8448", "--weight_cache_path", base + "/gemma4.xnnpack_cache"]
        else:
            command += ["--model_path", base + "/" + args.lm_model, "--max_num_tokens", "8448",
                        "--reuse_runtimes", "true", "--dump_full_logits", "false",
                        "--enable_ynnpack", "false", "--enable_profiling", "false",
                        "--trace_signatures", "false"]
        save(directory / "command.json", {"argv": command, "serial": args.serial})
        samples = [thermal()]
        print("Starting", name, flush=True)
        start = time.monotonic()
        with (directory / "process.log").open("w") as log:
            process = subprocess.Popen(adb + ["shell", shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    samples.append(thermal())
                    save(directory / "telemetry.json", samples)
        last_end = time.monotonic()
        samples.append(thermal())
        save(directory / "telemetry.json", samples)
        save(directory / "status.json", {"returncode": process.returncode,
                                         "wall_seconds": last_end - start})
        if process.returncode:
            raise RuntimeError(f"{name} failed; inspect {directory / 'process.log'}")
        subprocess.run(adb + ["pull", remote_output + "/.", str(directory / "capture")], check=True)
        print("Finished", name, flush=True)


if __name__ == "__main__":
    main()
