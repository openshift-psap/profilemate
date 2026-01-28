# ProfileMate: vLLM Runtime Profiling Tool

Comprehensive runtime instrumentation for vLLM servers to capture CUDA graph and KV cache metrics.

## Features

### Core Profiling
- ✅ **CUDA Graph Tracking**: Capture unique graphs, replay counts, and latencies
- ✅ **KV Cache Profiling**: Block allocations, usage patterns, and eviction metrics
- ✅ **MoE Expert Tracking**: Expert activations, co-selection, load balancing
- ✅ **Full BatchDescriptor Tracking**: See exact graph configurations

### NEW: Performance Profiling (V1 Scheduler)
- ✅ **Forward Pass Timing**: Accurate GPU timing with CUDA Events (prefill/decode)
- ✅ **CPU Operation Breakdown**: Scheduling, batch prep, sampling overhead
- ✅ **Batch Utilization Tracking**: Scheduling efficiency and queue analysis
- ✅ **Preemption Tracking**: Request lifecycle and preemption reasons
- ✅ **Encoder-Decoder Timing**: Generic support for Whisper, Qwen3-VL, GPT, etc.

### General
- ✅ **Automatic CSV Export**: Easy analysis with pandas/excel
- ✅ **Zero Code Changes**: Works via Python import hooks
- ✅ **Production-Ready**: Minimal overhead (<3%)
- ✅ **Configurable**: Enable/disable individual profilers

## Quick Start

### Option 1: Automated Profiling with nsys/ncu (Recommended)

**Complete automation with performance report**:

```bash
cd profilemate

# Quick profiling (~5 min)
./scripts/profile_vllm.sh --model meta-llama/Llama-2-7b-hf --mode quick

# Open HTML report
xdg-open ./profiling_results_*/profile_report.html
```

**Outputs**:
- ✅ Prefill/decode breakup
- ✅ Attention vs FFN vs MoE timing
- ✅ Kernel-level bandwidth analysis
- ✅ CUDA graph coverage
- ✅ Performance recommendations

**Modes**:
- `--mode quick`: nsys profiling only (~5 min)
- `--mode full --with-ncu`: nsys + ncu (~60 min)
- `--mode moe`: MoE expert tracking

See: [Nsight Automated Profiling Guide](docs/NSIGHT_AUTOMATED_PROFILING_GUIDE.md)

### Option 2: Runtime Instrumentation (sitecustomize)

**Continuous monitoring with minimal overhead**:

```bash
cd profilemate
export PYTHONPATH="$(pwd):$PYTHONPATH"

python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --port 9999

# Check results
ls /tmp/vllm_profiling/session_*/
```

**Outputs**:
- CUDA graph captures/replays
- KV cache usage over time
- Block eviction patterns
- (Optional) MoE expert activations

## Output Files

After running, you'll find:

```
/tmp/vllm_profiling/session_20260124_123456/
├── metadata.json                         # Session info
│
├── CUDA Graph Files:
│   ├── cuda_graph_captures.csv          # Unique graphs captured
│   ├── cuda_graph_usage.csv             # Replay frequency per graph
│   └── cuda_graph_timeline.csv          # Detailed replay timeline
│
├── KV Cache Files:
│   ├── kv_cache_usage.csv               # Usage over time
│   ├── kv_cache_evictions.csv           # Eviction events
│   └── kv_cache_summary.txt             # Summary statistics
│
├── MoE Expert Tracking:
│   ├── moe_expert_activations.csv       # Expert activation counts per layer
│   ├── moe_expert_coselection.csv       # Which experts are selected together
│   ├── moe_routing_weights_hist.csv     # Routing weight distributions
│   ├── moe_load_imbalance.csv           # Load balancing metrics over time
│   └── moe_summary.json                 # Aggregated statistics
│
├── Forward Pass Timing (NEW):
│   ├── forward_pass_timing.csv          # Prefill/decode GPU timing
│   └── forward_pass_summary.json        # Latency percentiles (P50/P95/P99)
│
├── CPU Operations (NEW):
│   ├── cpu_operations_timing.csv        # Scheduling, sampling, batch prep
│   └── cpu_timing_summary.json          # Per-operation breakdown
│
├── Batch Utilization (NEW):
│   ├── batch_utilization.csv            # Token/seq utilization over time
│   └── batch_utilization_summary.json   # Mean utilization, underutil events
│
├── Preemption Tracking (NEW):
│   ├── preemption_events.csv            # Preemption/resume events
│   ├── request_lifecycle.csv            # Full request timeline
│   └── preemption_summary.json          # Preemption rate, reasons, delays
│
└── Encoder-Decoder Timing (NEW):
    ├── encoder_decoder_timing.csv       # Component timing (encoder/decoder)
    └── encoder_decoder_summary.json     # Time distribution percentages
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
    # Core profilers
    ENABLE_CUDA_GRAPH_TRACKING = True
    ENABLE_KV_CACHE_TRACKING = True
    ENABLE_MOE_EXPERT_TRACKING = True

    # NEW: Performance profilers (V1 scheduler)
    ENABLE_FORWARD_PASS_TIMING = True
    ENABLE_CPU_TIMING = True
    ENABLE_BATCH_UTILIZATION_TRACKING = True
    ENABLE_PREEMPTION_TRACKING = True
    ENABLE_ENCODER_DECODER_TIMING = True

    # CUDA timing mode (for GPU profiling)
    USE_CUDA_EVENTS = True   # True = perfect accuracy (0.5% overhead)
                              # False = good accuracy (0.1% overhead)
    CUDA_EVENT_BATCH_SIZE = 100  # Sync every N iterations
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

### MoE Expert Activations

**moe_expert_activations.csv**:
```csv
layer_idx,expert_id,activation_count,percentage
0,0,15234,12.45
0,1,18932,15.47
0,2,14521,11.87
```

**What it tells you**:
- Which experts are being utilized in each layer
- Distribution of workload across experts
- Expert activation patterns and coverage

**Healthy patterns**:
- Balanced activation percentages (8-12% for 8 experts with top-2)
- All experts activated (100% coverage)
- Load balance ratio < 2.0 (max/min activations)

### MoE Expert Co-Selection

**moe_expert_coselection.csv**:
```csv
layer_idx,expert_id_1,expert_id_2,coselection_count
0,0,1,5432
0,0,2,4821
0,1,2,6123
```

**What it tells you**:
- Which pairs of experts are frequently selected together
- Routing patterns and expert specialization
- Potential expert redundancy

**Key insights**:
- Frequent co-selection → Experts handle related tasks
- Uniform co-selection → Good load distribution
- Skewed patterns → Some expert pairs dominate

### MoE Load Imbalance

**moe_load_imbalance.csv**:
```csv
layer_idx,timestamp_sec,std_dev,max_min_ratio
0,1.234,234.5,1.85
0,2.345,189.2,1.62
```

**What it tells you**:
- Load balancing quality over time
- Expert utilization variance
- Routing efficiency

**Healthy metrics**:
- `max_min_ratio < 2.0`: Good load balance
- `std_dev < mean * 0.3`: Reasonable variance
- Stable over time: Consistent routing

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

# Load MoE expert activations (if using MoE model)
moe_activations = pd.read_csv('moe_expert_tracking/moe_expert_activations.csv')

# Analyze expert load balance per layer
for layer_idx in moe_activations['layer_idx'].unique():
    layer_data = moe_activations[moe_activations['layer_idx'] == layer_idx]
    print(f"\nLayer {layer_idx}:")
    print(f"  Expert coverage: {len(layer_data)}/{layer_data['expert_id'].max()+1}")
    print(f"  Activation range: {layer_data['percentage'].min():.2f}% - {layer_data['percentage'].max():.2f}%")
    print(f"  Load balance ratio: {layer_data['activation_count'].max() / layer_data['activation_count'].min():.2f}")

# Plot expert activation distribution
layer_0 = moe_activations[moe_activations['layer_idx'] == 0]
plt.figure(figsize=(10, 6))
plt.bar(layer_0['expert_id'], layer_0['percentage'])
plt.xlabel('Expert ID')
plt.ylabel('Activation Percentage (%)')
plt.title('Expert Activation Distribution - Layer 0')
plt.grid(True, axis='y')
plt.savefig('expert_activations.png')
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

| Feature | sitecustomize.py | --cudagraph-metrics | --kv-cache-metrics | EPLB (MoE) |
|---------|------------------|---------------------|-------------------|-------------------|
| Unique CUDA graphs | ✅ Full details | ❌ Aggregated | N/A | N/A |
| Graph replay counts | ✅ Per graph | ❌ Aggregated | N/A | N/A |
| BatchDescriptor details | ✅ Complete | ❌ Partial | N/A | N/A |
| KV cache usage | ✅ Timeline | N/A | ✅ Sampled | N/A |
| Block allocations | ✅ Total count | N/A | ✅ Sampled | N/A |
| Block evictions | ✅ All events | N/A | ✅ Sampled | N/A |
| Expert activations | ✅ Per expert/layer | N/A | N/A | ❌ Aggregated only |
| Expert co-selection | ✅ Full patterns | N/A | N/A | ❌ Not tracked |
| Routing weights | ✅ Distributions | N/A | N/A | ❌ Not tracked |
| Load imbalance | ✅ Timeline | N/A | N/A | ✅ Balancedness only |
| Output format | CSV (easy analysis) | Logs | Prometheus | Logs |
| Overhead | <1% | <0.1% | <1% | <0.5% |

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

### 5. MoE Expert Load Balancing

**Goal**: Verify expert utilization is balanced (for MoE models)

```bash
# Check expert coverage per layer
cd moe_expert_tracking/

# Count unique experts activated
awk -F',' 'NR>1 {print $1}' moe_expert_activations.csv | sort -u | wc -l

# Find load imbalance per layer
python -c "
import pandas as pd
df = pd.read_csv('moe_expert_activations.csv')
for layer in df['layer_idx'].unique():
    layer_df = df[df['layer_idx'] == layer]
    ratio = layer_df['activation_count'].max() / layer_df['activation_count'].min()
    print(f'Layer {layer}: load balance ratio = {ratio:.2f}')
"
```

### 6. Expert Specialization Analysis

**Goal**: Understand which experts work together (co-selection patterns)

```bash
cd moe_expert_tracking/

# Find most common expert pairs
head -20 moe_expert_coselection.csv

# Analyze routing weight distribution
python -c "
import pandas as pd
import numpy as np
df = pd.read_csv('moe_routing_weights_hist.csv')
print('Routing weight statistics:')
print(f'  Mean: {df[\"weight\"].mean():.4f}')
print(f'  Std: {df[\"weight\"].std():.4f}')
print(f'  Min: {df[\"weight\"].min():.4f}')
print(f'  Max: {df[\"weight\"].max():.4f}')
"
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

### Quick Start Guides

- **[New Profilers Guide](docs/NEW_PROFILERS_GUIDE.md)** - **NEW!** Complete guide for the 5 new profilers:
  - ForwardPassProfiler: Accurate GPU timing with CUDA Events
  - CPUTimingProfiler: CPU operation breakdown
  - BatchUtilizationProfiler: Scheduling efficiency
  - PreemptionProfiler: Request lifecycle tracking
  - EncoderDecoderProfiler: Whisper, Qwen3-VL, GPT support
  - Complete analysis examples and optimization insights
  - **Total overhead: <3%** for all profilers combined
- **[CUDA Sync and GPU Timing Explained](docs/CUDA_SYNC_AND_GPU_TIMING.md)** - **NEW!** Why CUDA sync is necessary:
  - How asynchronous CUDA execution works
  - Why timing without sync is wrong (100-1000x error)
  - CUDA Events vs torch.cuda.synchronize()
  - Two profiling modes: perfect accuracy (0.5%) vs lightweight (0.1%)
  - Real-world examples and performance analysis

### Advanced Guides

- **[Nsight Automated Profiling Guide](docs/NSIGHT_AUTOMATED_PROFILING_GUIDE.md)** - Complete automation with nsys/ncu:
  - Automated profiling with Nsight Systems (timeline, NVTX markers)
  - Kernel-level analysis with Nsight Compute (bandwidth, roofline)
  - Python scripts to parse SQLite/CSV outputs automatically
  - Prefill/decode breakup extraction
  - Component timing (attention vs FFN vs MoE)
  - All-in-one automation script: `./scripts/profile_vllm.sh`
  - HTML report generation with performance recommendations
  - [Part 2: Integration & CI/CD](docs/NSIGHT_AUTOMATED_PROFILING_GUIDE_PART2.md)
- **[Advanced Profiling Guide](docs/ADVANCED_PROFILING_GUIDE.md)** - Comprehensive guide covering:
  - CUDA Graph modes (FULL, PIECEWISE, NONE, etc.) explained
  - Forward pass timing with minimal overhead
  - Scheduling efficiency metrics
  - Prefill/decode breakup analysis
  - GPU bandwidth estimation
  - Impact of max_model_len on performance
- **[MoE Expert Tracking & MFU Guide](docs/MOE_EXPERT_TRACKING_AND_MFU_GUIDE.md)** - MoE profiling and metrics reliability:
  - MoE expert activation tracking (existing support + custom implementation)
  - MFU (Model FLOPs Utilization) metrics reliability analysis
  - How analytical metrics work and their accuracy (95% for dense, 70-90% for MoE)
  - sitecustomize.py implementation for expert tracking
  - Load balancing analysis and expert utilization patterns
- **[MoE Expert Profiler Implementation](docs/MOE_EXPERT_PROFILER_IMPLEMENTATION.md)** - **NEW!** Complete MoEExpertProfiler implementation:
  - ✅ Production-ready implementation in sitecustomize.py
  - Automatic instrumentation of FusedMoE layer for Expert Parallelism
  - Per-layer expert activation tracking, co-selection patterns, and load balancing
  - CSV outputs with analysis examples and visualization scripts
  - <3% overhead with configurable sampling
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
