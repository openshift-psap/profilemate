#!/bin/bash
# Detailed NCU kernel profiling for vLLM
# Usage: ./ncu_detailed_profile.sh [model_name] [output_dir]

set -e

MODEL="${1:-meta-llama/Llama-2-7b-hf}"
OUTPUT_DIR="${2:-./profiling_results}"
PROFILE_NAME="vllm_ncu_profile"

mkdir -p "$OUTPUT_DIR"

echo "=== vLLM Detailed Kernel Profiling with Nsight Compute ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo
echo "WARNING: NCU profiling has 10-100x overhead. This will be slow."
echo

# Profile with NCU
echo "Starting ncu profile..."
ncu \
    --set full \
    --target-processes all \
    --kernel-name regex:"gemm|attention|moe|flash|fused" \
    --launch-skip 100 \
    --launch-count 50 \
    --output "$OUTPUT_DIR/$PROFILE_NAME" \
    --csv \
    --page raw \
    --force-overwrite \
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --max-num-seqs 32 \
        --max-model-len 1024 \
        --enforce-eager \
        --port 8001 &

NCU_PID=$!

# Send a few requests in background
sleep 40
python -c "
import requests
import time

# Wait for server
for i in range(60):
    try:
        requests.get('http://localhost:8001/health')
        break
    except:
        time.sleep(1)
else:
    print('Server failed to start')
    exit(1)

# Send a few requests (NCU is slow)
for i in range(5):
    try:
        requests.post(
            'http://localhost:8001/v1/completions',
            json={
                'model': '$MODEL',
                'prompt': 'Hello, how are you?',
                'max_tokens': 20,
            },
            timeout=120,
        )
        print(f'Request {i+1}/5 complete')
    except Exception as e:
        print(f'Request {i+1} failed: {e}')

print('Requests complete')
" &

# Wait for NCU to finish
wait $NCU_PID

echo
echo "=== Exporting to CSV ==="
ncu --csv --page raw "$OUTPUT_DIR/$PROFILE_NAME.ncu-rep" > "$OUTPUT_DIR/$PROFILE_NAME.csv"

echo "=== Parsing Results ==="
python "$(dirname "$0")/parse_ncu_profile.py" \
    "$OUTPUT_DIR/$PROFILE_NAME.csv" \
    --output-dir "$OUTPUT_DIR/ncu_parsed"

echo
echo "=== Profile Complete ==="
echo "NCU report: $OUTPUT_DIR/$PROFILE_NAME.ncu-rep"
echo "CSV export: $OUTPUT_DIR/$PROFILE_NAME.csv"
echo "Parsed results: $OUTPUT_DIR/ncu_parsed/"
echo
echo "View in GUI: ncu-ui $OUTPUT_DIR/$PROFILE_NAME.ncu-rep"
