import re
import json
import sys
import time

# Default paths
log_path = ""
output_path = ""

# Support command line args
if len(sys.argv) >= 3:
    log_path = sys.argv[1]
    output_path = sys.argv[2]

print(f"[{time.strftime('%H:%M:%S')}] Starting log processing...")
print(f"[{time.strftime('%H:%M:%S')}] Reading from: {log_path}")

# Regex to strip ANSI codes
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi(text):
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

phase = "unknown"
results = []
buffer = ""
lines_processed = 0

def process_buffer(text):
    if not text:
        return None
        
    # Clean text: strip ANSI, replace newlines
    text_clean = strip_ansi(text).replace("\n", " ").replace("\r", " ")
    
    # Must have iteration to be valid
    iter_match = patterns["iteration"].search(text_clean)
    if not iter_match:
        return None

    try:
        data = {
            "iteration": int(iter_match.group(1))
        }
        
        # Extract other fields
        for key, pattern in patterns.items():
            if key == "iteration": continue
            m = pattern.search(text_clean)
            if m:
                data[key] = float(m.group(1))
            else:
                # If a critical metric is missing, decide whether to skip or keep
                # For now, we prefer valid records with most data
                # But 'throughput' and 'loss' are usually essential
                pass
        
        # Simple validation: if we have time or loss, we count it
        if "elapsed_time_ms" in data or "lm_loss" in data:
            return data
            
    except ValueError:
        pass
        
    return None

try:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        print(f"[{time.strftime('%H:%M:%S')}] File opened safely. Scanning lines...")
        
        for line in f:
            lines_processed += 1
            if lines_processed % 2000 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Processed {lines_processed} lines... Found {len(results)} records.", end='\r')

            # 1. Phase detection
            if phase == "unknown":
                pm = phase_pattern.search(line)
                if pm:
                    phase = pm.group(1)
                    print(f"\n[{time.strftime('%H:%M:%S')}] Detected Training Phase: {phase}")

            # 2. Split logic
            # Megatron logs usually have "iteration X/Y" clearly
            # We treat "iteration ... / ..." as a start separator
            is_start_node = "iteration" in line and "/" in line and any(c.isdigit() for c in line)
            
            if is_start_node:
                # Flush previous buffer
                if buffer:
                    res = process_buffer(buffer)
                    if res:
                        results.append(res)
                buffer = line
            else:
                buffer += line

        # Final flush
        if buffer:
            res = process_buffer(buffer)
            if res:
                results.append(res)

    print(f"\n[{time.strftime('%H:%M:%S')}] Finished reading. Total lines: {lines_processed}")

except FileNotFoundError:
    print(f"Error: Log file not found at {log_path}")
    sys.exit(1)
except Exception as e:
    print(f"Error processing: {e}")
    sys.exit(1)

# Output
print(f"[{time.strftime('%H:%M:%S')}] Writing {len(results)} records to {output_path}...")
final = {phase: results}

try:
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)
    print("Done.")
except Exception as e:
    print(f"Error saving JSON: {e}")
