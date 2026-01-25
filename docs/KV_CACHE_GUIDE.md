# vLLM KV Cache Architecture and Profiling Guide

## Table of Contents

- [Overview](#overview)
- [KV Cache Fundamentals](#kv-cache-fundamentals)
- [How max-model-len Determines KV Cache Size](#how-max-model-len-determines-kv-cache-size)
- [KV Cache Layout and Block Organization](#kv-cache-layout-and-block-organization)
- [Available KV Cache Metrics](#available-kv-cache-metrics)
- [Using sitecustomize.py for KV Cache Profiling](#using-sitecustomizepy-for-kv-cache-profiling)
- [Configuration Options](#configuration-options)
- [Interpreting Results](#interpreting-results)

---

## Overview

vLLM uses a **block-based KV cache** system called PagedAttention, which organizes key-value cache data into fixed-size blocks for efficient memory management and sharing.

### Key Concepts

1. **Block Size**: Number of tokens stored in each KV cache block (default: 16 tokens)
2. **Block Pool**: Pre-allocated pool of KV cache blocks
3. **max_model_len**: Maximum sequence length the model can handle
4. **gpu_memory_utilization**: Fraction of GPU memory allocated for KV cache (default: 0.9)

---

## KV Cache Fundamentals

### What is KV Cache?

In transformer models, the Key-Value (KV) cache stores:
- **Keys (K)**: `[batch, num_heads, seq_len, head_dim]`
- **Values (V)**: `[batch, num_heads, seq_len, head_dim]`

For each layer in the model.

### Memory Requirements

For a single token in one layer:
```
kv_memory_per_token = 2 * num_heads * head_dim * dtype_size
```

For the entire model:
```
total_kv_memory = kv_memory_per_token * num_layers * max_model_len
```

### Example Calculation (GPT-2 Small)

```python
# Model specs
num_layers = 12
num_heads = 12
head_dim = 64
dtype_size = 2  # fp16/bf16

# For max_model_len = 2048
kv_per_token = 2 * 12 * 64 * 2 = 3,072 bytes
total_kv = 3,072 * 12 * 2048 = 75,497,472 bytes ≈ 72 MB
```

---

## How max-model-len Determines KV Cache Size

### 1. **Direct Impact on Block Count**

`max_model_len` directly determines the **maximum number of blocks** needed:

```python
max_blocks_per_sequence = ceil(max_model_len / block_size)
```

**Example** (block_size=16):
- `max_model_len=2048` → max 128 blocks per sequence
- `max_model_len=4096` → max 256 blocks per sequence
- `max_model_len=8192` → max 512 blocks per sequence

### 2. **GPU Memory Allocation Formula**

vLLM determines the total number of blocks during **memory profiling**:

```python
# Simplified formula
available_memory = total_gpu_memory * gpu_memory_utilization
model_memory = <loaded model weights + activations>
kv_cache_memory = available_memory - model_memory

num_gpu_blocks = kv_cache_memory / memory_per_block
```

Where:
```python
memory_per_block = block_size * num_layers * num_kv_heads * head_dim * dtype_size * 2
#                  ↑           ↑            ↑              ↑           ↑          ↑
#                  tokens      layers       attention      key/value  bytes/     K+V
#                  per block                heads          dim        element
```

### 3. **Configuration Files**

**Source**: `vllm/config/cache.py:39-149`

```python
@dataclass
class CacheConfig:
    block_size: BlockSize = None  # 1, 8, 16, 32, 64, 128, or 256
    gpu_memory_utilization: float = 0.9  # 90% of GPU memory
    num_gpu_blocks: int | None = None  # Set after profiling
    num_cpu_blocks: int | None = None
    kv_cache_memory_bytes: int | None = None  # Manual override
```

### 4. **Relationship Between Parameters**

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Memory Budget                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │          Model Weights & Activations                   │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │          KV Cache Memory                               │ │
│  │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐  │ │
│  │  │Blk1│Blk2│Blk3│... │                                │ │
│  │  │ 16 │ 16 │ 16 │    │    num_gpu_blocks              │ │
│  │  │toks│toks│toks│    │                                │ │
│  │  └────┴────┴────┴────┴────────────────────────────────┘ │ │
│  │          ↑                                               │ │
│  │          block_size determines tokens per block         │ │
│  └────────────────────────────────────────────────────────┘ │
│            ↑                                                 │
│            max_model_len limits max sequence length         │
└─────────────────────────────────────────────────────────────┘
```

### 5. **Impact on Performance**

| max_model_len | Memory Per Seq | Max Concurrent | Latency Impact |
|---------------|----------------|----------------|----------------|
| 512 | Low | High | Low |
| 2048 | Medium | Medium | Medium |
| 4096 | High | Low | High |
| 8192+ | Very High | Very Low | Very High |

**Trade-off**: Larger `max_model_len` → More memory per sequence → Fewer concurrent requests

---

## KV Cache Layout and Block Organization

### Block-Based Storage

vLLM uses **block tables** to map logical token positions to physical blocks:

```
Request Sequence: [T0, T1, T2, ..., T31]  (32 tokens)
Block Size: 16

Logical View:
┌────────────────┬────────────────┐
│   Tokens 0-15  │  Tokens 16-31  │
└────────────────┴────────────────┘

Physical Blocks:
┌────────────────┐
│   Block #42    │ ← Tokens 0-15
└────────────────┘
┌────────────────┐
│   Block #17    │ ← Tokens 16-31
└────────────────┘

Block Table: [42, 17]
```

**Source**: `vllm/v1/core/kv_cache_utils.py:30-66`

### Multi-Layer Storage

Each block stores KV data for **all layers**:

```python
@dataclass
class KVCacheBlock:
    block_id: int          # Physical block ID
    block_hash: bytes      # Hash for prefix caching
    ref_cnt: int           # Reference count for sharing
```

**Memory Layout per Block**:
```
Block #42:
├── Layer 0
│   ├── Keys:   [num_heads, block_size, head_dim]
│   └── Values: [num_heads, block_size, head_dim]
├── Layer 1
│   ├── Keys:   [num_heads, block_size, head_dim]
│   └── Values: [num_heads, block_size, head_dim]
...
└── Layer N
    ├── Keys:   [num_heads, block_size, head_dim]
    └── Values: [num_heads, block_size, head_dim]
```

### Prefix Caching (Block Sharing)

Blocks with identical content can be **shared** across requests:

```
Request 1: "The quick brown fox jumps"
Request 2: "The quick brown fox runs"

Shared Prefix: "The quick brown"
┌────────────────┐
│   Block #42    │ ← Shared by both requests
│ "The quick..."│
└────────────────┘
          ↑
    ref_cnt = 2
```

**Source**: `vllm/v1/core/block_pool.py:31-52`

### Impact of max_model_len on Layout

1. **Block Table Size**: Larger `max_model_len` → Larger block tables → More metadata overhead

2. **Fragmentation**: With large `max_model_len`, partially-filled blocks waste memory:
   ```
   Sequence length: 2050 tokens
   Block size: 16
   Blocks needed: ceil(2050/16) = 129 blocks
   Last block utilization: 2050 % 16 = 2 tokens (only 12.5% utilized)
   ```

3. **Optimal Block Size**:
   - Small block size (8-16): Better for variable-length requests, less waste
   - Large block size (32-64): Better for long sequences, less metadata overhead

---

## Available KV Cache Metrics

### 1. Built-in Metrics (--kv-cache-metrics)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --kv-cache-metrics \
    --kv-cache-metrics-sample 0.01
```

**Tracks**:
- Block lifetime (allocation to eviction)
- Idle time before eviction
- Reuse gaps (time between accesses)

**Source**: `vllm/v1/core/kv_cache_metrics.py:46-97`

```python
class KVCacheMetricsCollector:
    def on_block_allocated(self, block) -> None
    def on_block_accessed(self, block) -> None
    def on_block_evicted(self, block) -> None
```

**Output** (in Prometheus metrics):
```
vllm:kv_block_lifetime_seconds
vllm:kv_block_idle_before_evict_seconds
vllm:kv_block_reuse_gap_seconds
```

### 2. Scheduler Stats

Available at every logging interval:

**Source**: `vllm/v1/metrics/stats.py:165-191`

```python
@dataclass
class SchedulerStats:
    kv_cache_usage: float  # 0.0 to 1.0
    prefix_cache_stats: PrefixCacheStats
    kv_cache_eviction_events: list[KVCacheEvictionEvent]
```

**Logged as**:
```
INFO: GPU KV cache usage: 75.3%
INFO: Prefix cache hit rate: 45.2%
```

### 3. Cache Config Info

Prometheus metric with configuration:

```
vllm:cache_config_info{
    block_size="16",
    num_gpu_blocks="4096",
    num_cpu_blocks="512",
    cache_dtype="auto",
    gpu_memory_utilization="0.9"
} = 1
```

### 4. Direct Access via KVCacheManager

**Source**: `vllm/v1/core/kv_cache_manager.py:93-149`

```python
class KVCacheManager:
    @property
    def usage(self) -> float:
        """Get KV cache usage (0.0 to 1.0)"""
        return self.block_pool.get_usage()

    max_model_len: int
    num_kv_cache_groups: int
    block_pool: BlockPool
```

---

## Using sitecustomize.py for KV Cache Profiling

### Installation

```bash
cd /path/to/profilemate
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Optional: Configure output location
export VLLM_PROFILING_DIR="/custom/path"
export VLLM_PROFILING_VERBOSE=1
```

### Run vLLM with Instrumentation

```bash
python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --block-size 16 \
    --gpu-memory-utilization 0.9 \
    --port 9999
```

### Output Files

After running, check `/tmp/vllm_profiling/session_<timestamp>/`:

```
session_20260124_123456/
├── metadata.json                  # Session configuration
├── cuda_graph_captures.csv        # Unique CUDA graphs captured
├── cuda_graph_usage.csv           # Graph replay frequencies
├── cuda_graph_timeline.csv        # Detailed replay timeline
├── kv_cache_usage.csv             # KV cache usage over time
├── kv_cache_evictions.csv         # Block eviction events
└── kv_cache_summary.txt           # Summary statistics
```

### Example: kv_cache_usage.csv

```csv
timestamp_sec,usage_pct,num_blocks,total_blocks
0.123,15.30,625,4096
0.456,32.45,1329,4096
0.789,58.12,2381,4096
1.234,75.67,3099,4096
```

### Example: kv_cache_summary.txt

```
KV Cache Statistics
==================================================

Total blocks available: 4096
Total allocations: 15234
Total deallocations: 8901
Peak blocks used: 3456
Peak usage: 84.38%
Total evictions: 2134
Avg block lifetime: 12.34s
Avg idle before eviction: 3.45s
```

---

## Configuration Options

### 1. max-model-len

```bash
--max-model-len 4096
```

**Impact**:
- Limits maximum sequence length
- Affects number of blocks allocated
- Determines memory budget per sequence

**Guidelines**:
- Use smallest value that satisfies your workload
- Check actual request lengths in production
- Consider using `--max-num-seqs` to limit concurrent requests

### 2. block-size

```bash
--block-size 16  # Options: 1, 8, 16, 32, 64, 128, 256
```

**Trade-offs**:

| Block Size | Pros | Cons |
|------------|------|------|
| 8-16 | Less waste, better for variable lengths | More metadata overhead |
| 32-64 | Less overhead, better for long sequences | More waste on short sequences |
| 128+ | Minimal overhead | Significant waste |

**Recommendation**: Use 16 for general workloads, 32 for long-context scenarios

### 3. gpu-memory-utilization

```bash
--gpu-memory-utilization 0.9  # Range: 0.0 to 1.0
```

**Impact**:
```
Higher value → More KV cache → More concurrent requests
Lower value → Less memory pressure → More stable
```

**Guidelines**:
- Start with 0.9 (default)
- Reduce to 0.7-0.8 if seeing OOM errors
- Monitor with: `nvidia-smi dmon -s mu`

### 4. num-gpu-blocks-override

```bash
--num-gpu-blocks-override 2048
```

**Use cases**:
- Testing preemption behavior
- Simulating memory constraints
- Profiling with limited cache

### 5. enable-prefix-caching

```bash
--enable-prefix-caching  # Default: enabled
```

**Impact**:
- Enables block sharing for common prefixes
- Increases hit rate for similar requests
- Adds hashing overhead

**Metrics**:
```
INFO: Prefix cache hit rate: 45.2%
INFO: Prefix cache queries: 12345 tokens
INFO: Prefix cache hits: 5678 tokens
```

### 6. kv-cache-memory-bytes

```bash
--kv-cache-memory-bytes $((10 * 1024 * 1024 * 1024))  # 10 GB
```

**When to use**:
- Fine-grained control over KV cache size
- Overrides `gpu_memory_utilization`
- Useful for multi-tenant setups

---

## Interpreting Results

### Optimal KV Cache Usage

**Target**: 60-80% average usage

```
< 40%: Under-utilized → Reduce max_model_len or increase max_num_seqs
60-80%: Optimal → Good balance
> 90%: Over-utilized → Increase gpu_memory_utilization or reduce load
```

### Block Lifetime Analysis

**From kv_cache_evictions.csv**:

```python
import pandas as pd
df = pd.read_csv('kv_cache_evictions.csv')

# Healthy cache churn
avg_lifetime = df['lifetime_sec'].mean()
print(f"Average block lifetime: {avg_lifetime:.2f}s")

# If avg_lifetime < 5s → Too aggressive eviction, increase cache size
# If avg_lifetime > 60s → Blocks sitting idle, may have excess capacity
```

### Prefix Cache Hit Rate

**From logs**:
```
INFO: Prefix cache hit rate: 45.2%
```

**Interpretation**:
- `< 10%`: Low sharing, consider different workload or disable prefix caching
- `10-30%`: Moderate sharing, typical for diverse requests
- `> 50%`: High sharing, excellent for repetitive patterns

### CUDA Graph vs KV Cache Correlation

**Analyze together**:
```python
import pandas as pd

# Load both datasets
graphs = pd.read_csv('cuda_graph_usage.csv')
kv_cache = pd.read_csv('kv_cache_usage.csv')

# Find correlation
# High num_tokens in graphs → Higher KV cache usage
# Verify cache can support max graph batch sizes
```

---

## Best Practices

### 1. Start with Profiling

```bash
# Profile with realistic workload
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --max-model-len 2048 \  # Start conservative
    --block-size 16 \
    --gpu-memory-utilization 0.9 \
    --kv-cache-metrics
```

### 2. Monitor Key Metrics

- KV cache usage percentage
- Prefix cache hit rate
- Block eviction frequency
- Peak memory usage

### 3. Tune Based on Results

**If cache usage consistently < 50%**:
```bash
# Increase capacity or concurrent requests
--max-num-seqs 512
# or reduce max-model-len
--max-model-len 1024
```

**If seeing frequent evictions**:
```bash
# Increase cache size
--gpu-memory-utilization 0.95
# or reduce concurrent requests
--max-num-seqs 128
```

### 4. Validate with Load Testing

```bash
# Generate realistic load
python benchmark_serving.py \
    --model <model> \
    --dataset-path requests.json \
    --request-rate 10
```

---

## Troubleshooting

### Issue: OOM Errors

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce `--max-model-len`
2. Reduce `--gpu-memory-utilization` to 0.7-0.8
3. Reduce `--max-num-seqs`
4. Use quantization (`--quantization fp8`)

### Issue: Low Throughput

**Symptoms**:
```
Low tokens/sec, high latency
```

**Check**:
1. KV cache usage (should be 60-80%)
2. CUDA graph coverage (should be > 90%)
3. Prefix cache hit rate

**Solutions**:
1. Increase `--max-num-seqs`
2. Enable prefix caching
3. Optimize batch sizes

### Issue: High Memory Fragmentation

**Symptoms**:
```
Available blocks but allocation failures
```

**Solutions**:
1. Use smaller `--block-size` (e.g., 8 or 16)
2. Enable prefix caching for better sharing
3. Implement request batching

---

## Summary

### Key Takeaways

1. **max_model_len** directly controls:
   - Maximum sequence length
   - Memory budget per sequence
   - Number of concurrent requests possible

2. **Block size** affects:
   - Memory efficiency
   - Metadata overhead
   - Fragmentation

3. **Monitoring is essential**:
   - Use `--kv-cache-metrics` for detailed stats
   - Use sitecustomize.py for comprehensive profiling
   - Correlate with CUDA graph usage

4. **Optimal configuration** depends on:
   - Workload characteristics (short vs long contexts)
   - Request patterns (diverse vs repetitive)
   - Hardware constraints (GPU memory)

### Quick Reference

```bash
# Minimal instrumentation
--kv-cache-metrics

# Full profiling
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --max-model-len 2048 \
    --block-size 16 \
    --gpu-memory-utilization 0.9 \
    --kv-cache-metrics \
    --cudagraph-metrics
```

### Further Reading

- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [KV Cache Config](../vllm/vllm/config/cache.py)
- [Block Pool Implementation](../vllm/vllm/v1/core/block_pool.py)
- [CUDA Graphs Guide](CUDA_GRAPHS.md)
