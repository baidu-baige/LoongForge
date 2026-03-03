#!/bin/bash

# Quick Start Script for HF Checkpoint Roundtrip Test
# This script demonstrates how to run the roundtrip test

set -e

echo "=================================================="
echo "HF Checkpoint Roundtrip Test - Quick Start"
echo "=================================================="

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
export AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
export PYTHONPATH=$AIAK_TRAINING_PATH:$MEGATRON_PATH:$PYTHONPATH

# Configuration
HF_CHECKPOINT=/workspace/aiak-ckpt/Qwen2.5-0.5B-Instruct
YAML_CONFIG=$AIAK_TRAINING_PATH/tools/dist_checkpoint/demo/llm_demo.yaml
OUTPUT_DIR=/workspace/aiak-ckpt/qwen2.5-0.5b-hf-bridge-syh
NUM_PROCS=4

torchrun --nproc_per_node=$NUM_PROCS \
    tools/dist_checkpoint/checkpoint/hf_roundtrip_test.py \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --yaml-file "$YAML_CONFIG" \
    --output-dir "$OUTPUT_DIR"

# Check results
if [ -f "$OUTPUT_DIR/comparison_report.json" ]; then
    echo ""
    echo "=================================================="
    echo "Test Results"
    echo "=================================================="
    cat "$OUTPUT_DIR/comparison_report.json"
    echo ""
    echo "Report saved to: $OUTPUT_DIR/comparison_report.json"
else
    echo "❌ Roundtrip test failed (no report generated)"
    exit 1
fi

echo ""
echo "✅ Roundtrip test completed successfully!"