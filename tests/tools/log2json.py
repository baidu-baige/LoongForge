import re
import json

log_path = "/Users/chen/Desktop/model/FT/AIAK-Training-Omni/add.log"
output_path = "/Users/chen/Desktop/model/FT/AIAK-Training-Omni/tests/baseline/optional/internvl2.5_8b.json"

# 需要提取的字段
iter_pattern = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\): ([\d\.]+) \| throughput \(token/sec/GPU\): ([\d\.]+).*?lm loss: ([\d\.E\+\-]+).*?grad norm: ([\d\.E\+\-]+).*?mem-allocated-bytes-avg\(MB\): ([\d\.]+) \| mem-max-allocated-bytes-avg\(MB\): ([\d\.]+)",
    re.DOTALL
)
phase_pattern = re.compile(r"training_phase\s*\.*\s*(\w+)")

with open(log_path, "r") as f:
    content = f.read()

# 提取 training_phase
phase_match = phase_pattern.search(content)
phase = phase_match.group(1) if phase_match else "unknown"

# 提取每个 iteration 的信息
results = []
for m in iter_pattern.finditer(content):
    results.append({
        "iteration": int(m.group(1)),
        "elapsed_time_ms": float(m.group(3)),
        "throughput": float(m.group(4)),
        "lm_loss": float(m.group(5)),
        "grad_norm": float(m.group(6)),
        "mem_allocated_avg_MB": float(m.group(7)),
        "mem_max_allocated_avg_MB": float(m.group(8)),
    })

# 组织为目标格式
final = {phase: results}

with open(output_path, "w") as f:
    json.dump(final, f, indent=2)