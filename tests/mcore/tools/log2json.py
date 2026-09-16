# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""parse log file"""

import re
import json
import sys
import time
from typing import Dict, List, Tuple, Optional

# Regex to strip ANSI codes
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi(text: str) -> str:
    return ansi_escape.sub('', text)

# Flexible patterns for individual fields
patterns = {
    "iteration": re.compile(r"iteration\s+(\d+)/\s*(\d+)"),
    "elapsed_time_ms": re.compile(r"elapsed time per iteration \(ms\):\s*([\d\.]+)"),
    "throughput": re.compile(r"throughput \(token/sec/GPU\):\s*([\d\.]+)"),
    "lm_loss": re.compile(r"lm loss:\s*([\d\.E\+\-]+)"),
    "grad_norm": re.compile(r"grad norm:\s*([\d\.E\+\-]+)"),
    "mem_allocated_avg_MB": re.compile(r"mem-allocated-bytes-avg\(MB\):\s*([\d\.]+)"),
    "mem_max_allocated_avg_MB": re.compile(r"mem-max-allocated-bytes-avg\(MB\):\s*([\d\.]+)")
}

phase_pattern = re.compile(r"training_phase\s*\.*\s*(\w+)")

def _process_buffer(text: str) -> Optional[Dict[str, float]]:
    if not text:
        return None
    
    text_clean = strip_ansi(text).replace("\n", " ").replace("\r", " ")
    
    iter_match = patterns["iteration"].search(text_clean)
    if not iter_match:
        return None

    try:
        data = {
            "iteration": int(iter_match.group(1))
        }
        
        for key, pattern in patterns.items():
            if key == "iteration":
                continue
            m = pattern.search(text_clean)
            if m:
                data[key] = float(m.group(1))
        
        if "elapsed_time_ms" in data or "lm_loss" in data:
            return data
    except ValueError:
        return None
        
    return None

def parse_log_file(log_path: str) -> Tuple[str, List[Dict[str, float]]]:
    phase = "unknown"
    results: List[Dict[str, float]] = []
    buffer = ""

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if phase == "unknown":
                pm = phase_pattern.search(line)
                if pm:
                    phase = pm.group(1)

            is_start_node = "iteration" in line and "/" in line and any(c.isdigit() for c in line)
            if is_start_node:
                if buffer:
                    res = _process_buffer(buffer)
                    if res:
                        results.append(res)
                buffer = line
            else:
                buffer += line

        if buffer:
            res = _process_buffer(buffer)
            if res:
                results.append(res)

    return phase, results

def write_json(output_path: str, phase: str, records: List[Dict[str, float]]) -> None:
    final = {phase: records}
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)

def main() -> int:
    log_path = ""
    output_path = ""

    if len(sys.argv) >= 3:
        log_path = sys.argv[1]
        output_path = sys.argv[2]

    print(f"[{time.strftime('%H:%M:%S')}] Starting log processing...")
    print(f"[{time.strftime('%H:%M:%S')}] Reading from: {log_path}")

    try:
        phase, results = parse_log_file(log_path)
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return 1
    except Exception as e:
        print(f"Error processing: {e}")
        return 1

    print(f"[{time.strftime('%H:%M:%S')}] Writing {len(results)} records to {output_path}...")
    try:
        write_json(output_path, phase, results)
        print("Done.")
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
