#!/bin/bash
# Quick nsys profiling for vLLM
# Usage: ./nsys_quick_profile.sh [model_name] [output_dir]

set -e

MODEL="${1:-meta-llama/Llama-2-7b-hf}"
OUTPUT_DIR="${2:-./profiling_results}"
PROFILE_NAME="vllm_quick_profile"

mkdir -p "$OUTPUT_DIR"

echo "=== vLLM Quick Profiling with Nsight Systems ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo

# Enable NVTX markers
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

# Start profiling
echo "Starting nsys profile..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --delay=30 \
    --duration=60 \
    --output="$OUTPUT_DIR/$PROFILE_NAME" \
    --export=sqlite,text \
    --force-overwrite=true \
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --max-num-seqs 256 \
        --max-model-len 2048 \
        --port 8000 &

VLLM_PID=$!

echo "vLLM server starting (PID: $VLLM_PID)..."
sleep 35

# Send test requests
echo "Sending test requests..."
python -c "
import torch
import requests
import time

# Wait for server
for i in range(30):
    try:
        requests.get('http://localhost:8000/health')
        break
    except:
        time.sleep(1)
else:
    print('Server failed to start')
    exit(1)

# Trigger profiling
torch.cuda.cudart().cudaProfilerStart()

# Send requests
for i in range(50):
    requests.post(
        'http://localhost:8000/v1/completions',
        json={
            'model': '$MODEL',
            'prompt': 'The quick brown fox jumps over the lazy dog. ' * 10,
            'max_tokens': 50,
        }
    )

# Stop profiling
torch.cuda.cudart().cudaProfilerStop()
time.sleep(2)
"

echo "Profiling complete. Stopping server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo
echo "=== Parsing Results ==="
python "$(dirname "$0")/parse_nsys_profile.py" \
    "$OUTPUT_DIR/$PROFILE_NAME.sqlite" \
    --output-dir "$OUTPUT_DIR/parsed"

echo
echo "=== Profile Complete ==="
echo "Nsys report: $OUTPUT_DIR/$PROFILE_NAME.nsys-rep"
echo "Parsed results: $OUTPUT_DIR/parsed/"
echo
echo "View in GUI: nsys-ui $OUTPUT_DIR/$PROFILE_NAME.nsys-rep"
