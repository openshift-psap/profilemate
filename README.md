# ProfileMate: vLLM Runtime Profiling Tool

Comprehensive runtime instrumentation for vLLM servers to capture CUDA graph and KV cache metrics.

## Features

- ✅ **CUDA Graph Tracking**: Capture unique graphs, replay counts, and latencies
- ✅ **KV Cache Profiling**: Block allocations, usage patterns, and eviction metrics
- ✅ **Full BatchDescriptor Tracking**: See exact graph configurations
- ✅ **Automatic CSV Export**: Easy analysis with pandas/excel
- ✅ **Zero Code Changes**: Works via Python import hooks
- ✅ **Production-Ready**: Minimal overhead (<1%)

## Quick Start

### 1. Installation

```bash
cd profilemate
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### 2. Run vLLM with Profiling

```bash
python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --port 9999
```

### 3. Check Results

```bash
ls /tmp/vllm_profiling/session_*/
```

## Output Files

After running, you'll find:

```
/tmp/vllm_profiling/session_20260124_123456/
├── metadata.json                  # Session info
│
├── CUDA Graph Files:
│   ├── cuda_graph_captures.csv   # Unique graphs captured
│   ├── cuda_graph_usage.csv      # Replay frequency per graph
│   └── cuda_graph_timeline.csv   # Detailed replay timeline
│
└── KV Cache Files:
    ├── kv_cache_usage.csv         # Usage over time
    ├── kv_cache_evictions.csv     # Eviction events
    └── kv_cache_summary.txt       # Summary statistics
```

## Configuration

### Environment Variables

```bash
# Change output location (default: /tmp/vllm_profiling)
export VLLM_PROFILING_DIR="/custom/path"

# Enable verbose logging
export VLLM_PROFILING_VERBOSE=1

# Adjust logging interval (default: 100 operations)
export VLLM_PROFILING_LOG_INTERVAL=50
```

### Selective Tracking

Edit `sitecustomize.py`:

```python
class ProfilingConfig:
    ENABLE_CUDA_GRAPH_TRACKING = True   # Set to False to disable
    ENABLE_KV_CACHE_TRACKING = True     # Set to False to disable
```

## Understanding the Output

### CUDA Graph Captures

**cuda_graph_captures.csv**:
```csv
runtime_mode,num_tokens,num_reqs,uniform,has_lora,capture_time_sec
FULL,256,128,True,False,2.345
FULL,512,256,True,False,3.456
PIECEWISE,1024,None,False,False,4.567
```

**What it tells you**:
- Which unique CUDA graphs were created
- When each graph was captured (relative to start)
- Full BatchDescriptor configuration

### CUDA Graph Usage

**cuda_graph_usage.csv**:
```csv
runtime_mode,num_tokens,num_reqs,uniform,has_lora,replay_count
FULL,256,128,True,False,5432
FULL,512,256,True,False,3210
PIECEWISE,1024,None,False,False,876
```

**What it tells you**:
- How often each unique graph was replayed
- Which graphs are "hot" (frequently used)
- Distribution of workload across graphs

**Key insights**:
- If one graph dominates → Workload is uniform
- If many graphs used → Workload is diverse
- Compare with `--cudagraph-metrics` aggregated stats

### KV Cache Usage

**kv_cache_usage.csv**:
```csv
timestamp_sec,usage_pct,num_blocks,total_blocks
0.123,15.30,625,4096
0.456,32.45,1329,4096
0.789,58.12,2381,4096
```

**What it tells you**:
- Cache utilization over time
- Peak usage patterns
- Whether you're over/under-provisioned

**Optimal ranges**:
- 60-80%: Good balance
- <40%: Over-provisioned, reduce `max_model_len`
- >90%: Under-provisioned, increase `gpu_memory_utilization`

### KV Cache Evictions

**kv_cache_evictions.csv**:
```csv
timestamp_sec,lifetime_sec,idle_sec
1.234,12.34,3.45
2.345,8.76,2.10
```

**What it tells you**:
- How long blocks lived before eviction
- How long blocks sat idle
- Cache churn rate

**Healthy metrics**:
- `lifetime_sec > 10`: Blocks are well-utilized
- `idle_sec < 5`: Efficient eviction
- Few evictions overall: Good cache sizing

## Analysis Examples

### Python Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CUDA graph usage
graphs = pd.read_csv('cuda_graph_usage.csv')

# Find most-used graphs
top_graphs = graphs.nlargest(10, 'replay_count')
print("Top 10 CUDA Graphs:")
print(top_graphs[['num_tokens', 'replay_count']])

# Load KV cache usage
kv_cache = pd.read_csv('kv_cache_usage.csv')

# Plot cache usage over time
plt.figure(figsize=(12, 6))
plt.plot(kv_cache['timestamp_sec'], kv_cache['usage_pct'])
plt.xlabel('Time (seconds)')
plt.ylabel('KV Cache Usage (%)')
plt.title('KV Cache Usage Over Time')
plt.grid(True)
plt.savefig('kv_cache_usage.png')
```

### Command-Line Analysis

```bash
# Count unique CUDA graphs
wc -l cuda_graph_captures.csv

# Find most-used graph
sort -t',' -k6 -rn cuda_graph_usage.csv | head -5

# Calculate average KV cache usage
awk -F',' 'NR>1 {sum+=$2; count++} END {print sum/count}' kv_cache_usage.csv

# Check peak usage
sort -t',' -k2 -rn kv_cache_usage.csv | head -1
```

## Comparison with Built-in Metrics

| Feature | sitecustomize.py | --cudagraph-metrics | --kv-cache-metrics |
|---------|------------------|---------------------|-------------------|
| Unique CUDA graphs | ✅ Full details | ❌ Aggregated | N/A |
| Graph replay counts | ✅ Per graph | ❌ Aggregated | N/A |
| BatchDescriptor details | ✅ Complete | ❌ Partial | N/A |
| KV cache usage | ✅ Timeline | N/A | ✅ Sampled |
| Block allocations | ✅ Total count | N/A | ✅ Sampled |
| Block evictions | ✅ All events | N/A | ✅ Sampled |
| Output format | CSV (easy analysis) | Logs | Prometheus |
| Overhead | <1% | <0.1% | <1% |

**Recommendation**: Use **both** for comprehensive profiling:
```bash
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --cudagraph-metrics \
    --kv-cache-metrics \
    --port 9999
```

## Troubleshooting

### Issue: No output files generated

**Check**:
1. Verify `PYTHONPATH` is set correctly:
   ```bash
   echo $PYTHONPATH
   ```

2. Check if sitecustomize loaded:
   ```bash
   python -c "import sys; print('sitecustomize' in sys.modules)"
   ```

3. Look for startup message:
   ```
   [sitecustomize] vLLM Comprehensive Instrumentation Loaded
   ```

### Issue: Import errors

**Solution**: Ensure vLLM is installed:
```bash
pip list | grep vllm
```

### Issue: Permission denied on output directory

**Solution**: Change output directory:
```bash
export VLLM_PROFILING_DIR="$HOME/vllm_profiling"
```

### Issue: High overhead

**Solution**: Disable detailed tracking:
```python
# Edit sitecustomize.py
class ProfilingConfig:
    LOG_INTERVAL = 1000  # Log less frequently
```

## Use Cases

### 1. Understanding Workload Patterns

**Goal**: See which batch sizes are actually used

```bash
# Run with profiling
python -m vllm.entrypoints.openai.api_server --model gpt2

# Analyze results
cat cuda_graph_usage.csv | cut -d',' -f2 | sort | uniq -c
```

### 2. Optimizing KV Cache Size

**Goal**: Determine if `max_model_len` is too large

```bash
# Run with conservative setting
--max-model-len 2048

# Check peak usage
sort -t',' -k2 -rn kv_cache_usage.csv | head -1

# If peak < 60%, reduce max_model_len
# If peak > 90%, increase gpu_memory_utilization
```

### 3. CUDA Graph Coverage Analysis

**Goal**: Ensure most requests use CUDA graphs

```bash
# Compare CUDA graph replays vs total requests
total_replays=$(awk -F',' 'NR>1 {sum+=$6} END {print sum}' cuda_graph_usage.csv)
echo "Total CUDA graph replays: $total_replays"

# High number → Good CUDA graph coverage
```

### 4. Prefix Caching Validation

**Goal**: Verify prefix caching is effective

```bash
# Look for block reuse in eviction data
# Short lifetime + many accesses = good sharing
awk -F',' 'NR>1 && $2<10 {count++} END {print count " blocks evicted quickly"}' \
    kv_cache_evictions.csv
```

## Advanced Usage

### Custom Profiling Hooks

Edit `sitecustomize.py` to add custom tracking:

```python
def patch_custom_component():
    """Add your own instrumentation"""
    try:
        from vllm.custom.module import CustomClass
        original_method = CustomClass.method

        def instrumented_method(self, *args, **kwargs):
            # Your tracking code here
            start = time.time()
            result = original_method(self, *args, **kwargs)
            duration = time.time() - start

            # Log or save metrics
            print(f"Custom metric: {duration}")

            return result

        CustomClass.method = instrumented_method
    except ImportError:
        pass
```

### Integrating with Monitoring Systems

Export to Prometheus format:

```python
# convert_to_prometheus.py
import pandas as pd

kv_cache = pd.read_csv('kv_cache_usage.csv')

# Generate Prometheus metrics
for _, row in kv_cache.iterrows():
    print(f'vllm_kv_cache_usage{{timestamp="{row["timestamp_sec"]}"}} {row["usage_pct"]}')
```

## Documentation

### Guides

- **[Advanced Profiling Guide](docs/ADVANCED_PROFILING_GUIDE.md)** - **NEW!** Comprehensive guide covering:
  - CUDA Graph modes (FULL, PIECEWISE, NONE, etc.) explained
  - Forward pass timing with minimal overhead
  - Scheduling efficiency metrics
  - Prefill/decode breakup analysis
  - GPU bandwidth estimation
  - Impact of max_model_len on performance
- **[KV Cache Guide](docs/KV_CACHE_GUIDE.md)**: Deep dive into KV cache architecture
- **[CUDA Graphs Guide](docs/CUDA_GRAPHS.md)**: CUDA graph metrics and tracking

### Source Code

- **[sitecustomize.py](sitecustomize.py)**: Source code with inline comments

## Contributing

To add new metrics or improve tracking:

1. Edit `sitecustomize.py`
2. Add profiler class (see `CUDAGraphProfiler` or `KVCacheProfiler`)
3. Create patch function
4. Register in `install_import_hook()`
5. Test and document

## License

Same as vLLM project (Apache 2.0)

## Credits

Built for analyzing vLLM performance in MLPerf inference benchmarks.
