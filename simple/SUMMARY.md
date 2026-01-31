# Simple Profiler Summary

## What We Created

A **simplified, easy-to-understand** profiler that tracks forward pass timing in vLLM.

### Key Differences from Full Version

| Aspect | Simple Version | Full Version |
|--------|---------------|--------------|
| **Complexity** | ~270 lines | ~1600 lines |
| **Features** | Forward pass timing only | CUDA graphs, KV cache, MoE, etc. |
| **Patching** | Immediate + simple import hook | Complex import hook system |
| **Dependencies** | Minimal (stdlib only) | Same |
| **Learning Curve** | Easy to understand | Complex |
| **Maintenance** | Easy to modify | More complex |

## Architecture

### 1. Immediate Patching
- Tries to patch `GPUModelRunner.execute_model()` as soon as `sitecustomize.py` loads
- If vLLM isn't imported yet, sets up a simple import hook

### 2. Simple Import Hook
- Uses modern `find_spec()` API (Python 3.4+)
- Only intercepts `vllm.v1.worker.gpu_model_runner`
- Patches immediately after module loads

### 3. Timing Collection
- Wraps `execute_model()` method
- Measures time with `time.perf_counter()`
- Synchronizes GPU for accurate timing
- Records: phase, batch_size, num_tokens, duration

### 4. Data Storage
- CSV file for timeline data
- JSON file for summary statistics
- Auto-saves on process exit

## Code Structure

```
sitecustomize.py (270 lines)
│
├── Config (lines 34-47)
│   └── Simple configuration with auto-detection
│
├── ForwardPassProfiler (lines 52-108)
│   ├── record() - Record timing data
│   ├── save() - Save to CSV/JSON
│   └── _compute_summary() - Calculate statistics
│
├── Global Setup (lines 113-123)
│   └── Create profiler instance if enabled
│
├── patch_gpu_model_runner() (lines 128-195)
│   └── Patches GPUModelRunner.execute_model()
│       ├── Determines phase (prefill/decode)
│       ├── Measures timing
│       └── Records data
│
└── apply_patches() (lines 200-235)
    ├── Try immediate patching
    └── Set up import hook for later
```

## Why This Approach?

### ✅ Advantages

1. **Simplicity**: Easy to understand and modify
2. **Immediate**: Patches as soon as possible
3. **Robust**: Handles different vLLM versions gracefully
4. **Focused**: Does one thing well (forward pass timing)
5. **Modern**: Uses `find_spec()` instead of deprecated APIs

### ⚠️ Limitations

1. **Single Feature**: Only tracks forward pass timing
2. **No CUDA Graph Details**: Doesn't track capture/replay
3. **No KV Cache Metrics**: Doesn't track cache usage
4. **No MoE Tracking**: Doesn't track expert activations

## Next Steps

### Option 1: Add CUDA Graph Tracking
Add a function to patch `CUDAGraphWrapper.__call__()`:
```python
def patch_cuda_graph_wrapper():
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    # Patch to track captures/replays
```

### Option 2: Add More Timing Breakdowns
Break down forward pass into:
- Preprocessing time
- Model forward time
- Postprocessing time

### Option 3: Add Batch Statistics
Track:
- Batch utilization
- Token utilization
- Queue lengths

## Testing

To test the simple profiler:

```bash
# 1. Set up
export PYTHONPATH="/mnt/data/nmiriyal/profilemate/simple:$PYTHONPATH"
export VLLM_ENABLE_PROFILING=1
export VLLM_PROFILING_VERBOSE=1

# 2. Run vLLM
python -m vllm.entrypoints.openai.api_server --model <model> --port 8000

# 3. Send some requests
curl http://localhost:8000/v1/completions ...

# 4. Check output
ls -la /tmp/vllm_profiling/session_*/
cat /tmp/vllm_profiling/session_*/forward_pass_timing.csv
cat /tmp/vllm_profiling/session_*/summary.json
```

## Expected Output

### Console
```
[Simple Profiler] Forward pass timing enabled
  Session ID: 20240101_120000
  Output directory: /tmp/vllm_profiling/session_20240101_120000
[Simple Profiler] ✅ Successfully patched GPUModelRunner.execute_model
[Forward Pass] prefill: 45.23ms (batch=1, tokens=128)
[Forward Pass] decode: 12.45ms (batch=1, tokens=1)
...
```

### CSV File
```csv
timestamp_sec,phase,batch_size,num_tokens,duration_ms,throughput_tokens_per_sec
0.123,prefill,1,128,45.230,2830.5
0.456,decode,1,1,12.450,80.3
...
```

### JSON Summary
```json
{
  "prefill": {
    "count": 100,
    "mean_ms": 45.2,
    "p50_ms": 44.0,
    "p95_ms": 65.3
  },
  "decode": {
    "count": 1000,
    "mean_ms": 12.5,
    "p50_ms": 12.0,
    "p95_ms": 18.2
  }
}
```

## Comparison: Simple vs Full

### When to Use Simple Version

✅ Use simple version when:
- You only need forward pass timing
- You want to understand how profiling works
- You want to extend it yourself
- You need something easy to maintain

### When to Use Full Version

✅ Use full version when:
- You need CUDA graph tracking
- You need KV cache metrics
- You need MoE expert tracking
- You need comprehensive profiling

## Code Quality

- ✅ Clean, readable code
- ✅ Well-commented
- ✅ Error handling
- ✅ Version compatibility checks
- ✅ Modern Python APIs
- ✅ No deprecated code

This simple version is a great starting point for understanding vLLM profiling and can be easily extended with additional features!
