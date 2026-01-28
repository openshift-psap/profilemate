#!/bin/bash
# Complete automated vLLM profiling pipeline
# Usage: ./profile_vllm.sh --model <model> --mode [quick|full|moe]

set -e

# Default values
MODEL="meta-llama/Llama-2-7b-hf"
MODE="quick"
OUTPUT_DIR="./profiling_results_$(date +%Y%m%d_%H%M%S)"
RUN_NCU="false"
GENERATE_REPORT="true"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --with-ncu)
            RUN_NCU="true"
            shift
            ;;
        --no-report)
            GENERATE_REPORT="false"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --model MODEL          Model to profile (default: Llama-2-7b-hf)"
            echo "  --mode MODE            quick|full|moe (default: quick)"
            echo "  --output-dir DIR       Output directory (default: ./profiling_results_TIMESTAMP)"
            echo "  --with-ncu             Also run NCU profiling (slow!)"
            echo "  --no-report            Skip HTML report generation"
            echo "  --help                 Show this help"
            echo
            echo "Modes:"
            echo "  quick  - Fast nsys profiling only (~5 min)"
            echo "  full   - Nsys + NCU profiling (~60 min)"
            echo "  moe    - MoE-specific profiling with expert tracking"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

echo "============================================"
echo "vLLM Automated Profiling Pipeline"
echo "============================================"
echo "Model: $MODEL"
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo "NCU profiling: $RUN_NCU"
echo "============================================"
echo

# Step 1: Nsight Systems profiling
echo "=== Step 1/4: Nsight Systems Profiling ==="
bash "$SCRIPT_DIR/nsys_quick_profile.sh" "$MODEL" "$OUTPUT_DIR"

# Step 2: NCU profiling (if requested or mode=full)
if [[ "$RUN_NCU" == "true" ]] || [[ "$MODE" == "full" ]]; then
    echo
    echo "=== Step 2/4: Nsight Compute Profiling ==="
    bash "$SCRIPT_DIR/ncu_detailed_profile.sh" "$MODEL" "$OUTPUT_DIR"
    NCU_DIR="$OUTPUT_DIR/ncu_parsed"
else
    echo
    echo "=== Step 2/4: Skipping NCU profiling ==="
    NCU_DIR=""
fi

# Step 3: MoE expert tracking (if mode=moe)
if [[ "$MODE" == "moe" ]]; then
    echo
    echo "=== Step 3/4: MoE Expert Tracking ==="
    export PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH"

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --max-num-seqs 128 \
        --port 8002 &

    VLLM_PID=$!
    sleep 30

    # Send requests
    python -c "
import requests
import time

for i in range(20):
    try:
        requests.get('http://localhost:8002/health')
        break
    except:
        time.sleep(1)

for i in range(100):
    requests.post(
        'http://localhost:8002/v1/completions',
        json={
            'model': '$MODEL',
            'prompt': 'Explain quantum computing in simple terms. ' * 5,
            'max_tokens': 100,
        }
    )

print('MoE profiling requests complete')
"

    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true

    # Copy MoE results to output dir
    MOE_SESSION=$(ls -t /tmp/vllm_profiling/ | head -1)
    if [[ -n "$MOE_SESSION" ]]; then
        cp -r "/tmp/vllm_profiling/$MOE_SESSION" "$OUTPUT_DIR/moe_expert_tracking"
        echo "MoE expert tracking results: $OUTPUT_DIR/moe_expert_tracking"
    fi
else
    echo
    echo "=== Step 3/4: Skipping MoE tracking ==="
fi

# Step 4: Generate comprehensive report
if [[ "$GENERATE_REPORT" == "true" ]]; then
    echo
    echo "=== Step 4/4: Generating Report ==="

    REPORT_ARGS="--nsys-results $OUTPUT_DIR/parsed --output $OUTPUT_DIR/profile_report.html"
    if [[ -n "$NCU_DIR" ]]; then
        REPORT_ARGS="$REPORT_ARGS --ncu-results $NCU_DIR"
    fi

    python "$SCRIPT_DIR/generate_profile_report.py" $REPORT_ARGS

    echo
    echo "============================================"
    echo "Profiling Complete!"
    echo "============================================"
    echo "Results directory: $OUTPUT_DIR"
    echo
    echo "Key files:"
    echo "  - HTML Report: $OUTPUT_DIR/profile_report.html"
    echo "  - Nsys GUI: nsys-ui $OUTPUT_DIR/vllm_quick_profile.nsys-rep"
    if [[ -n "$NCU_DIR" ]]; then
        echo "  - NCU GUI: ncu-ui $OUTPUT_DIR/vllm_ncu_profile.ncu-rep"
    fi
    if [[ "$MODE" == "moe" ]]; then
        echo "  - MoE tracking: $OUTPUT_DIR/moe_expert_tracking/"
    fi
    echo
    echo "Open report: xdg-open $OUTPUT_DIR/profile_report.html"
else
    echo
    echo "=== Step 4/4: Skipping report generation ==="
fi

echo
echo "Done!"
