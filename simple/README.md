# Simple vLLM Forward Pass Profiler

A simplified, easy-to-understand profiler that tracks forward pass timing in vLLM.

## Features

- ✅ **Simple**: Only tracks forward pass timing (no complex features)
- ✅ **Immediate Patching**: Patches vLLM modules as soon as they load
- ✅ **No Import Hooks Complexity**: Uses simple import interception
- ✅ **Easy to Understand**: Clean, readable code (~200 lines)
- ✅ **CSV Output**: Simple CSV files for analysis

## Quick Start

```bash
# 1. Set PYTHONPATH to include the simple directory
export PYTHONPATH="/mnt/data/nmiriyal/profilemate/simple:$PYTHONPATH"

# 2. (Optional) Force enable profiling
export VLLM_ENABLE_PROFILING=1

# 3. (Optional) Enable verbose logging
export VLLM_PROFILING_VERBOSE=1

# 4. Run vLLM
python -m vllm.entrypoints.openai.api_server --model <your-model>
```

## How It Works

1. **Immediate Patching**: When `sitecustomize.py` loads, it tries to patch `GPUModelRunner.execute_model()` immediately
2. **Import Hook**: If vLLM isn't loaded yet, it sets up a simple import hook to patch when the module loads
3. **Timing**: Wraps the forward pass execution and measures time
4. **Data Collection**: Records phase (prefill/decode), batch size, tokens, and duration
5. **Save on Exit**: Automatically saves data to CSV when the process exits

## Output

Profiling data is saved to:
```
/tmp/vllm_profiling/session_<timestamp>/
├── forward_pass_timing.csv
└── summary.json
```

### forward_pass_timing.csv

Columns:
- `timestamp_sec`: Time since start (seconds)
- `phase`: "prefill" or "decode"
- `batch_size`: Number of requests in batch
- `num_tokens`: Total tokens in batch
- `duration_ms`: Forward pass duration (milliseconds)
- `throughput_tokens_per_sec`: Tokens per second

### summary.json

Statistics grouped by phase:
```json
{
  "prefill": {
    "count": 100,
    "mean_ms": 45.2,
    "min_ms": 32.1,
    "max_ms": 78.5,
    "p50_ms": 44.0,
    "p95_ms": 65.3,
    "p99_ms": 72.1
  },
  "decode": {
    "count": 1000,
    "mean_ms": 12.5,
    ...
  }
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ENABLE_PROFILING` | Auto-detect | Set to `1` to force enable, `0` to disable |
| `VLLM_PROFILING_DIR` | `/tmp/vllm_profiling` | Output directory |
| `VLLM_PROFILING_VERBOSE` | `0` | Set to `1` for verbose logging |

### Auto-Detection

If `VLLM_ENABLE_PROFILING` is not set, the profiler auto-detects vLLM processes by checking command line arguments for:
- `vllm.entrypoints`
- `vllm.engine`
- `vllm`
- `--model`

## Example Output

### Console Output

```
[Simple Profiler] Forward pass timing enabled
  Session ID: 20240101_120000
  Output directory: /tmp/vllm_profiling/session_20240101_120000
[Simple Profiler] ✅ Successfully patched GPUModelRunner.execute_model
[Forward Pass] prefill: 45.23ms (batch=1, tokens=128)
[Forward Pass] decode: 12.45ms (batch=1, tokens=1)
...
[Simple Profiler] Saved statistics to /tmp/vllm_profiling/session_20240101_120000/
  - Total forward passes: 1100
  - prefill: mean=45.20ms, count=100
  - decode: mean=12.50ms, count=1000
```

## Comparison with Full Version

| Feature | Simple Version | Full Version |
|---------|---------------|--------------|
| Forward Pass Timing | ✅ | ✅ |
| CUDA Graph Tracking | ❌ | ✅ |
| KV Cache Tracking | ❌ | ✅ |
| MoE Expert Tracking | ❌ | ✅ |
| Batch Utilization | ❌ | ✅ |
| Preemption Tracking | ❌ | ✅ |
| Code Complexity | ~200 lines | ~1600 lines |
| Import Hooks | Simple | Complex |

## Troubleshooting

### No Output Files

1. Check if profiling is enabled:
   ```bash
   echo $VLLM_ENABLE_PROFILING
   ```

2. Check if you see the startup message:
   ```
   [Simple Profiler] Forward pass timing enabled
   ```

3. Check if patch was applied:
   ```
   [Simple Profiler] ✅ Successfully patched GPUModelRunner.execute_model
   ```

### Patch Not Applied

If you don't see the "Successfully patched" message:
- vLLM might not be installed
- Module path might be different (check vLLM version)
- Try setting `VLLM_PROFILING_VERBOSE=1` for more info

### No Data in CSV

- Make sure vLLM is actually processing requests
- Check that forward passes are happening (not just scheduling)
- Enable verbose mode to see timing messages

## Next Steps

Once you understand how this simple version works, you can:
1. Add CUDA graph tracking
2. Add KV cache metrics
3. Add more detailed timing breakdowns
4. Extend to other vLLM components

## Code Structure

```
sitecustomize.py
├── Config: Configuration class
├── ForwardPassProfiler: Data collection and saving
├── patch_gpu_model_runner(): Patching function
├── apply_patches(): Immediate patching + import hook
└── save_stats(): Exit handler
```

The code is intentionally simple and well-commented for easy understanding and modification.
