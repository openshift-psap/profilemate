# Quick Answers to Your Profiling Questions

This document provides direct answers to common profiling questions. For detailed explanations, see [ADVANCED_PROFILING_GUIDE.md](docs/ADVANCED_PROFILING_GUIDE.md).

---

## 1. What are the different CUDA Graph modes?

### Quick Summary Table

| Mode | Performance | Memory | Use When |
|------|------------|--------|----------|
| **NONE** | Slowest | Lowest | Debugging, NVTX layerwise tracing |
| **PIECEWISE** | Good | Moderate | Attention backend requires it |
| **FULL** | Best* | Highest | Small models, simple workloads (*if compatible) |
| **FULL_DECODE_ONLY** | Excellent | Moderate | Decode instance in P/D split |
| **FULL_AND_PIECEWISE** | Best overall | High | **Production (DEFAULT)** |

### Mode Explanations

**NONE**: No CUDA graphs at all. Use `--enforce-eager`.

**PIECEWISE**: Captures graphs for everything EXCEPT attention operations. Attention runs in eager mode. Most compatible.

**FULL**: Everything in one big CUDA graph, including attention. Requires compatible attention backend. Best performance but not always supported.

**FULL_DECODE_ONLY**:
- Decode-only batches (1 token/request) → Full CUDA graph
- Mixed prefill-decode → Eager mode
- Good for dedicated decode instances

**FULL_AND_PIECEWISE** (V1 Default):
- Decode-only batches → Full CUDA graph (best performance)
- Prefill or mixed batches → Piecewise CUDA graph (good performance)
- **Best for production**

**Code location**: `vllm/vllm/config/compilation.py:55-99`

---

## 2. Can we add timers for each forward pass?

**Yes!** Multiple options with different overhead levels:

### Option 1: Built-in MFU Metrics (Recommended)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-mfu-metrics
```

**Output:**
```
Engine 000: MFU: 245.3 TF/s/GPU 892.1 GB/s/GPU
```

**Overhead**: 0.5-1%

**With detailed breakdown:**
```bash
export VLLM_DEBUG_MFU_METRICS=1
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-mfu-metrics
```

This gives you per-component breakdowns (attention, FFN, etc.) with prefill vs decode separation.

### Option 2: Custom Timer (Low Overhead)

**Add to**: `vllm/vllm/v1/worker/gpu_model_runner.py` (around line 3000)

```python
import time

# Before forward pass
start_time = time.perf_counter()

# Forward pass happens here
output = self.model.forward(...)

# After forward pass
forward_time = time.perf_counter() - start_time
```

**Overhead**: ~0.1%

**Important**: Do NOT use `torch.cuda.synchronize()` - it adds 5-10% overhead!

### Where to Place Timers

**Good locations** (low overhead):
- ✓ `vllm/v1/worker/gpu_model_runner.py:execute_model()` - Overall execution
- ✓ `vllm/v1/worker/gpu_model_runner.py:~line 3000` - Model forward only
- ✓ `vllm/v1/engine/core.py:step()` - Full iteration

**Avoid**:
- ✗ Inside attention kernels (too frequent)
- ✗ Per-layer timing (use NVTX + Nsight instead)
- ✗ Anywhere requiring `torch.cuda.synchronize()` (huge overhead)

---

## 3. Does debug log print scheduling steps?

**Yes!** Multiple ways to see scheduling details:

### Option 1: Standard Logs (Always Available)

```
Engine 000: Avg prompt throughput: 1250.3 tokens/s, Avg generation throughput: 892.7 tokens/s,
Running: 45 reqs, Waiting: 12 reqs, Preemptions: 3, GPU KV cache usage: 72.5%
```

Shows:
- Number of running requests
- Number of waiting (queued) requests
- Preemptions (scheduling pressure)
- KV cache utilization

### Option 2: Detailed Iteration Logging

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-logging-iteration-details
```

**Code location**: `vllm/vllm/config/observability.py:78-83`

Logs:
- Number of context (prefill) requests
- Number of generation (decode) requests
- Number of tokens processed
- Elapsed CPU time for iteration

### Option 3: Debug Logging

```bash
VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-logging-iteration-details
```

Shows detailed scheduling decisions including:
- `scheduled_new_reqs` - New requests scheduled
- `scheduled_cached_reqs` - Continuing requests scheduled
- `num_batched_tokens` - Total tokens in batch
- `max_num_batched_tokens` - Maximum allowed

### What You Can Measure

**Batch utilization:**
```
utilization = num_batched_tokens / max_num_batched_tokens
```

**Preemption rate:**
```
preemption_rate = num_preemptions / total_iterations
```

**Queue pressure:**
```
queue_pressure = num_waiting_reqs / num_running_reqs
```

---

## 4. Can we understand prefill/decode breakup?

**Yes!** Built-in tracking with MFU metrics.

### Enable Prefill/Decode Tracking

```bash
export VLLM_DEBUG_MFU_METRICS=1

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-mfu-metrics
```

### Sample Output

```json
{
  "prefill_reqs": 10,
  "decode_reqs": 50,
  "num_batches": 60,
  "context_breakdown": {
    "num_prefill_requests": 10,
    "prefill_num_tokens": 20480,
    "prefill_context_len": 20480,
    "num_decode_requests": 50,
    "decode_num_tokens": 50,
    "decode_context_len": 409600
  }
}
```

### Prometheus Metrics (Per-Request)

```bash
curl http://localhost:8000/metrics | grep -E "(prefill|decode)"
```

Shows:
- `vllm:request_prefill_time_seconds` - Time in prefill phase
- `vllm:request_decode_time_seconds` - Time in decode phase
- `vllm:request_prefill_kv_computed_tokens` - KV tokens computed during prefill

### Calculate Ratios

```python
import json

with open('vllm_mfu_debug.log') as f:
    for line in f:
        if 'context_breakdown' in line:
            data = json.loads(line.split('MFU details: ')[1])

            prefill_tokens = data['context_breakdown']['prefill_num_tokens']
            decode_tokens = data['context_breakdown']['decode_num_tokens']

            total = prefill_tokens + decode_tokens
            prefill_pct = prefill_tokens / total * 100
            decode_pct = decode_tokens / total * 100

            print(f"Prefill: {prefill_pct:.1f}% ({prefill_tokens} tokens)")
            print(f"Decode: {decode_pct:.1f}% ({decode_tokens} tokens)")
```

---

## 5. Can we calculate GPU bandwidth for kernels?

**Yes!** Two approaches:

### Option 1: vLLM Built-in Bandwidth (Analytical)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-mfu-metrics
```

**Output:**
```
Engine 000: MFU: 245.3 TF/s/GPU 892.1 GB/s/GPU
```

Where:
- **245.3 TF/s/GPU** = Compute throughput
- **892.1 GB/s/GPU** = **Memory bandwidth** (read + write)

**How it's calculated**:
```python
bandwidth = (total_read_bytes + total_write_bytes) / elapsed_time / 1e9
```

**With breakdown** (set `VLLM_DEBUG_MFU_METRICS=1`):
```json
{
  "num_read_bytes_breakdown": {
    "attn.qkv_weight": "12.3GB",
    "attn.attn_input": "45.6GB",
    "ffn.dense_up_gate_weights": "78.9GB"
  },
  "num_write_bytes_breakdown": {
    "attn.kv_cache": "6.2GB",
    "ffn.dense_up_gate_output": "9.8GB"
  }
}
```

### Option 2: Per-Kernel Bandwidth (NVIDIA Nsight Compute)

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

ncu --set full --target-processes all \
    --kernel-name regex:"attention|gemm|flash" \
    -o vllm_bandwidth_profile \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-hf

# View report
ncu-ui vllm_bandwidth_profile.ncu-rep
```

**Metrics in Nsight Compute:**
- `dram__bytes_read.sum` - Total DRAM reads per kernel
- `dram__bytes_write.sum` - Total DRAM writes per kernel
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` - % of peak bandwidth
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` - Load efficiency
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` - Store efficiency

### Compare Against Hardware Specs

```python
# Example: A100-80GB
HW_PEAK_BANDWIDTH = 2000  # GB/s (HBM2e)
measured_bandwidth = 892.1  # From vLLM metrics

utilization = (measured_bandwidth / HW_PEAK_BANDWIDTH) * 100
print(f"Bandwidth utilization: {utilization:.1f}%")

if utilization > 70:
    print("Memory-bound workload")
else:
    print("Compute-bound workload")
```

### Rough Manual Calculation

```python
# Model: Llama-2-7B
model_size_gb = 14  # 7B params * 2 bytes (FP16)
kv_cache_gb = 8     # Typical for batch
activations_gb = 2  # Intermediate tensors

# Per forward pass
total_bytes = (
    model_size_gb +      # Read model weights
    kv_cache_gb * 2 +    # Read + Write KV cache
    activations_gb * 2   # Read + Write activations
)

# If forward pass takes 10ms
forward_time_s = 0.010
bandwidth_gb_s = total_bytes / forward_time_s

print(f"Estimated bandwidth: {bandwidth_gb_s:.1f} GB/s")
# Output: ~360 GB/s
```

---

## 6. How to capture impact of max_model_len on performance?

### Quick Experiment

```bash
# Baseline: max_model_len = 2048
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-model-len 2048 \
    --enable-mfu-metrics \
    --cudagraph-metrics \
    2>&1 | tee vllm_2k.log

# Test: max_model_len = 4096
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-model-len 4096 \
    --enable-mfu-metrics \
    --cudagraph-metrics \
    2>&1 | tee vllm_4k.log
```

### Compare Metrics

```bash
# 1. Memory usage
grep "GPU KV cache usage" vllm_*.log

# 2. Throughput
grep "Avg generation throughput" vllm_*.log

# 3. Bandwidth
grep "MFU:" vllm_*.log | grep "GB/s"

# 4. CUDA graph stats
grep "CUDAGraph Stats" -A 10 vllm_*.log
```

### What Changes with max_model_len

**Doubling `max_model_len` causes:**

| Aspect | Impact | Formula |
|--------|--------|---------|
| **KV Cache Memory** | 2x | `mem = max_model_len * layers * heads * dim * 2 * dtype * batch` |
| **Decode Bandwidth** | 2x | More KV cache to read per token |
| **Prefill FLOPs** | 4x | Attention is O(n²) for prefill |
| **Throughput (Decode)** | -10-20% | More bandwidth required |
| **Throughput (Prefill)** | -50-75% | Quadratic complexity |

### Expected Performance Table

| max_model_len | KV Cache (7B) | Decode Throughput | Prefill Throughput |
|---------------|---------------|-------------------|-------------------|
| 512           | ~4 GB         | 100% (baseline)   | 100% (baseline)   |
| 2048          | ~16 GB        | ~90%              | ~25%              |
| 4096          | ~32 GB        | ~80%              | ~6%               |
| 8192          | ~64 GB        | ~65%              | ~2%               |

### Analysis Script

```python
import re

def parse_log(filename):
    with open(filename) as f:
        content = f.read()

    # Extract KV cache usage
    kv_match = re.search(r'GPU KV cache usage: ([\d.]+)%', content)
    kv_usage = float(kv_match.group(1)) if kv_match else 0

    # Extract throughput
    throughput_match = re.search(r'Avg generation throughput: ([\d.]+) tokens/s', content)
    throughput = float(throughput_match.group(1)) if throughput_match else 0

    # Extract bandwidth
    bandwidth_match = re.search(r'MFU: [\d.]+ TF/s/GPU ([\d.]+) GB/s/GPU', content)
    bandwidth = float(bandwidth_match.group(1)) if bandwidth_match else 0

    return {
        'kv_usage': kv_usage,
        'throughput': throughput,
        'bandwidth': bandwidth
    }

baseline = parse_log('vllm_2k.log')
test = parse_log('vllm_4k.log')

print(f"KV Cache Usage: {baseline['kv_usage']:.1f}% → {test['kv_usage']:.1f}%")
print(f"Throughput: {baseline['throughput']:.1f} → {test['throughput']:.1f} tokens/s " +
      f"({(test['throughput']/baseline['throughput']-1)*100:+.1f}%)")
print(f"Bandwidth: {baseline['bandwidth']:.1f} → {test['bandwidth']:.1f} GB/s " +
      f"({(test['bandwidth']/baseline['bandwidth']-1)*100:+.1f}%)")
```

### Optimization Strategies

**If max_model_len is too large:**

1. **Use appropriate value**: Don't set 8K if you only need 2K
   ```bash
   --max-model-len 2048  # Match your actual needs
   ```

2. **Enable prefix caching**: Reduces effective context
   ```bash
   --enable-prefix-caching
   ```

3. **Use quantization**: FP8 KV cache = half the memory
   ```bash
   --kv-cache-dtype fp8
   ```

4. **Chunked prefill**: Break long prefills into chunks
   ```bash
   --max-num-batched-tokens 4096  # Smaller than max_model_len
   ```

---

## 7. MoE Expert Tracking and MFU Metrics Reliability

### Can we track expert activation in MoE models?

**Partial built-in support**, but custom tracking recommended for detailed analysis.

#### Built-in Option: EPLB Load Metrics

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-MoE-72B \
    --tensor-parallel-size 4 \
    --enable-eplb \
    --eplb-config '{"log_balancedness": true}'
```

**Output:**
```
[EPLB] Balancedness: 0.85 (avg tokens per expert ÷ max tokens per expert)
```

**What it tracks:**
- ✓ Load balancing metric (aggregate)
- ✗ Per-expert activation counts
- ✗ Router weight distribution
- ✗ Expert selection patterns

**Code location**: `vllm/vllm/distributed/eplb/eplb_state.py:118-138`

#### Custom Tracking: sitecustomize.py (Recommended)

**Add to profilemate/sitecustomize.py**:

```python
class MoEExpertProfiler:
    """Tracks MoE expert activation patterns."""

    def record_expert_selection(self, layer_idx, topk_ids, topk_weights):
        # Count expert activations
        for expert_id in topk_ids.flatten().tolist():
            self.expert_activations[layer_idx][expert_id] += 1

        # Track co-selection patterns
        if topk_ids.size(1) == 2:
            for pair in topk_ids.tolist():
                key = f"{min(pair)},{max(pair)}"
                self.co_selection_patterns[key] += 1
```

**Outputs:**
- `moe_expert_activations.csv` - Per-expert activation counts
- `moe_expert_coselection.csv` - Which experts are selected together
- `moe_routing_weights_hist.csv` - Distribution of routing weights

**See**: [MOE_EXPERT_TRACKING_AND_MFU_GUIDE.md](docs/MOE_EXPERT_TRACKING_AND_MFU_GUIDE.md) for full implementation.

### How reliable are MFU metrics?

**MFU metrics are ANALYTICAL (calculated), not measured.**

#### How They Work

```
1. Parse model config → Extract dimensions (layers, heads, etc.)
2. Track scheduler output → Get num_tokens, context_len
3. Calculate FLOPs → Use analytical formulas
4. Calculate bandwidth → Model weight reads, KV cache access
5. Divide by time → Get TF/s and GB/s
```

**Code location**: `vllm/vllm/v1/metrics/perf.py`

#### Reliability by Model Type

| Model Type | FLOPs Accuracy | Bandwidth Accuracy | Notes |
|------------|----------------|-------------------|-------|
| **Dense (Llama, Qwen)** | **>95%** | **~90%** | Highly reliable |
| **MoE (balanced)** | **~90%** | **~85%** | Good if load is balanced |
| **MoE (skewed)** | **~70-90%** | **~70-85%** | Depends on expert imbalance |
| **Prefill phase** | **>95%** | **~90%** | Large matmuls are predictable |
| **Decode phase** | **~90%** | **~80-85%** | Cache effects hard to model |

#### Key Assumptions (May Not Hold)

**From `vllm/v1/metrics/perf.py:776-777`:**

```python
# FIXME: Assume perfect load balancing for now.
num_activated_experts = min(num_activated_tokens, num_experts)
```

**Reality**: Expert load is often skewed:
- Top 10% experts: 40% of tokens
- Middle 50%: 50% of tokens
- Bottom 40%: 10% of tokens

**Impact**: If load is skewed, MFU metrics can be 10-30% off.

#### What MFU Metrics DO NOT Include

❌ **Kernel efficiency** (assumes 100% theoretical peak)
❌ **Scheduling overhead** (CPU-GPU sync, kernel launches)
❌ **Communication costs** (TP all-reduce, EP all-to-all)
❌ **Cache effects** (hot vs cold experts)

#### Verification Methods

**Method 1: Compare with Nsight Compute**

```bash
ncu --set full -o profile.ncu-rep <vllm command>
```

**Typical deltas**:
- Compute (FLOPs): ±5-10%
- Memory bandwidth: ±10-15%

**Method 2: Roofline Analysis**

```python
arithmetic_intensity = total_flops / total_bytes

# A100: Ridge point = 312 TF/s / 2000 GB/s = 156 FLOPs/byte
# If your AI > 156 → Compute-bound (FLOPs metrics accurate)
# If your AI < 156 → Memory-bound (Bandwidth metrics more important)
```

#### When to Trust MFU Metrics

✅ **Highly reliable for:**
- Relative comparisons (comparing runs)
- Dense transformer models
- Identifying compute vs memory bottlenecks
- Capacity planning

⚠️ **Less reliable for:**
- Absolute performance claims
- MoE models with skewed expert usage
- Quantized models (hardware has specialized units)

❌ **Not accurate for:**
- Attention-variant architectures (sliding window, sparse, MLA)
- Models with custom kernels

#### Example: MoE Load Skew Impact

```python
# Scenario: Qwen2.5-MoE-72B with 64 experts, top-2 routing

# Theoretical (uniform): Each expert gets 2/64 = 3.125% of tokens
# Reality (from profiling):
#   - Top 10 experts: 5% each = 50% total
#   - Next 20 experts: 2% each = 40% total
#   - Bottom 34 experts: 0.3% each = 10% total

# Impact on MFU:
# - FLOPs: Still accurate (same total compute)
# - Bandwidth: Hot experts cached → 10-20% faster than analytical
# - Load balance: EPLB balancedness = 0.62 (vs 1.0 perfect)
```

#### Recommendations

**For production**:
```bash
# Monitor both MFU metrics AND expert load balance
python -m vllm.entrypoints.openai.api_server \
    --model <moe-model> \
    --enable-mfu-metrics \
    --enable-eplb \
    --eplb-config '{"log_balancedness": true}'
```

**For deep analysis**:
```bash
# Use sitecustomize for per-expert tracking
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
export VLLM_DEBUG_MFU_METRICS=1

python -m vllm.entrypoints.openai.api_server \
    --model <moe-model> \
    --enable-mfu-metrics
```

**Cross-check with Nsight**:
```bash
ncu --set full --target-processes all \
    -o verify.ncu-rep \
    <vllm command>
```

**See**: [MOE_EXPERT_TRACKING_AND_MFU_GUIDE.md](docs/MOE_EXPERT_TRACKING_AND_MFU_GUIDE.md) for complete details.

---

---

## 8. Automated Profiling with nsys and ncu

### How can we automate profiling and extract all breakdowns?

**Answer**: Complete automation pipeline with scripts provided!

### Quick Start: All-in-One Script

```bash
# Quick profiling (nsys only, ~5 min)
./profilemate/scripts/profile_vllm.sh \
    --model meta-llama/Llama-2-7b-hf \
    --mode quick

# Full profiling (nsys + ncu, ~60 min)
./profilemate/scripts/profile_vllm.sh \
    --model meta-llama/Llama-2-7b-hf \
    --mode full \
    --with-ncu

# MoE-specific profiling
./profilemate/scripts/profile_vllm.sh \
    --model Qwen/Qwen2.5-MoE-72B \
    --mode moe
```

**Outputs**:
- ✅ Nsys timeline analysis (prefill/decode, components)
- ✅ NCU kernel analysis (bandwidth, roofline)
- ✅ Parsed CSV files (automatically extracted)
- ✅ Comprehensive HTML report with recommendations
- ✅ MoE expert tracking (if mode=moe)

### What Gets Extracted Automatically

**From Nsight Systems** (`parse_nsys_profile.py`):
```python
# Prefill vs Decode breakdown
prefill_decode = {
    'prefill': {'count': 10, 'total_ms': 1234.5, 'avg_ms': 123.45},
    'decode': {'count': 50, 'total_ms': 567.8, 'avg_ms': 11.36}
}

# Component breakdown
components = {
    'attention': {'total_ms': 456.7, 'percent': 45.2},
    'ffn': {'total_ms': 345.6, 'percent': 34.2},
    'moe': {'total_ms': 123.4, 'percent': 12.2}
}

# Top kernels by time
top_kernels = [
    {'kernel_name': 'flash_attention_v2', 'total_time_ms': 234.5, ...},
    {'kernel_name': 'gemm_fp16', 'total_time_ms': 123.4, ...}
]

# CUDA graph coverage
cuda_graphs = {
    'total_kernel_launches': 5000,
    'graph_kernel_launches': 4500,
    'graph_coverage_pct': 90.0
}
```

**From Nsight Compute** (`parse_ncu_profile.py`):
```python
# Per-kernel bandwidth
bandwidth_metrics = [
    {
        'kernel_name': 'flash_attention_v2',
        'bandwidth_gbps': 892.1,
        'dram_throughput_pct': 85.2,
        'sm_throughput_pct': 45.3,
        'bottleneck': 'memory'  # or 'compute'
    },
    ...
]

# Roofline data
roofline_data = [
    {
        'kernel_name': 'gemm_fp16',
        'sm_utilization': 92.1,
        'memory_utilization': 43.5,
        'bottleneck': 'compute'
    },
    ...
]
```

### Manual Step-by-Step (If Needed)

#### Step 1: Nsight Systems Profiling

```bash
# Run nsys with NVTX markers
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

nsys profile \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    --delay=30 --duration=60 \
    --output=vllm_profile \
    --export=sqlite \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-hf
```

#### Step 2: Parse nsys SQLite

```bash
python ./profilemate/scripts/parse_nsys_profile.py \
    vllm_profile.sqlite \
    --output-dir ./results
```

**Outputs**:
- `./results/nvtx_ranges.csv` - All NVTX markers with timings
- `./results/kernel_stats.csv` - Per-kernel statistics
- `./results/summary.json` - Aggregated metrics

#### Step 3: Nsight Compute (Optional)

```bash
# Profile specific kernels
ncu --set full \
    --kernel-name regex:"attention|gemm|moe" \
    --launch-skip 100 --launch-count 50 \
    --output=vllm_kernels \
    --csv \
    python -m vllm.entrypoints.openai.api_server ...

# Export to CSV
ncu --csv --page raw vllm_kernels.ncu-rep > vllm_kernels.csv

# Parse
python ./profilemate/scripts/parse_ncu_profile.py \
    vllm_kernels.csv \
    --output-dir ./ncu_results
```

#### Step 4: Generate HTML Report

```bash
python ./profilemate/scripts/generate_profile_report.py \
    --nsys-results ./results \
    --ncu-results ./ncu_results \
    --output report.html
```

**Report includes**:
- Executive summary with key metrics
- Prefill/decode breakdown table
- Component time breakdown
- Top kernels by time
- Bandwidth analysis with bottleneck identification
- Performance recommendations

### Integration with Existing Tools

**Combine with sitecustomize profiling**:

```bash
# Enable all profiling layers
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

# Run with all metrics
./profilemate/scripts/profile_vllm.sh \
    --model <model> \
    --mode full \
    --with-ncu

# Results include:
# - sitecustomize data: /tmp/vllm_profiling/session_*/
# - nsys data: ./profiling_results/parsed/
# - ncu data: ./profiling_results/ncu_parsed/
# - Combined report: ./profiling_results/profile_report.html
```

### CI/CD Integration

**Automated regression detection**:

```bash
# Run baseline profiling
./profilemate/scripts/profile_vllm.sh --model <model> --mode quick
cp ./profiling_results_*/parsed/summary.json baseline_profile.json

# After changes, compare
./profilemate/scripts/profile_vllm.sh --model <model> --mode quick
python ./profilemate/scripts/check_regression.py \
    --current ./profiling_results_*/parsed/summary.json \
    --baseline baseline_profile.json \
    --threshold 10  # 10% regression threshold

# Exit code 1 if regressions detected
```

### Key Scripts Provided

| Script | Purpose | Time |
|--------|---------|------|
| `profile_vllm.sh` | All-in-one automation | 5-60 min |
| `nsys_quick_profile.sh` | Quick nsys profiling | ~5 min |
| `ncu_detailed_profile.sh` | Detailed kernel analysis | ~30-60 min |
| `parse_nsys_profile.py` | Extract nsys metrics | <1 min |
| `parse_ncu_profile.py` | Extract ncu metrics | <1 min |
| `generate_profile_report.py` | HTML report generator | <1 min |
| `check_regression.py` | Compare vs baseline | <1 min |
| `send_test_requests.py` | Send test traffic | Variable |

### Example Output Structure

```
profiling_results_20260127_123456/
├── vllm_quick_profile.nsys-rep         # View: nsys-ui file.nsys-rep
├── vllm_quick_profile.sqlite           # Parsed automatically
├── vllm_ncu_profile.ncu-rep            # View: ncu-ui file.ncu-rep
├── vllm_ncu_profile.csv                # Parsed automatically
├── parsed/                             # nsys results
│   ├── nvtx_ranges.csv                 # All NVTX markers
│   ├── kernel_stats.csv                # Per-kernel stats
│   ├── memcpy_stats.csv                # Memory transfers
│   └── summary.json                    # Aggregated metrics
├── ncu_parsed/                         # ncu results
│   ├── kernel_bandwidth_metrics.csv    # Bandwidth per kernel
│   ├── roofline_data.csv               # Roofline analysis
│   └── ncu_summary.json                # Aggregated metrics
├── moe_expert_tracking/                # MoE results (if mode=moe)
│   ├── moe_expert_activations.csv
│   ├── moe_expert_coselection.csv
│   └── moe_routing_weights_hist.csv
└── profile_report.html                 # Open in browser
```

### Quick Analysis Examples

**Find prefill/decode split**:
```bash
cat ./profiling_results_*/parsed/summary.json | \
    python -c "import sys, json; d=json.load(sys.stdin); \
    print(f\"Prefill: {d['prefill_decode']['prefill']['total_ms']:.1f}ms\"); \
    print(f\"Decode: {d['prefill_decode']['decode']['total_ms']:.1f}ms\")"
```

**Find top time-consuming component**:
```bash
cat ./profiling_results_*/parsed/summary.json | \
    python -c "import sys, json; d=json.load(sys.stdin); \
    comps=sorted(d['components'].items(), key=lambda x: x[1]['total_ms'], reverse=True); \
    print(f\"Top component: {comps[0][0]} ({comps[0][1]['percent']:.1f}%)\")"
```

**Check CUDA graph coverage**:
```bash
cat ./profiling_results_*/parsed/summary.json | \
    python -c "import sys, json; d=json.load(sys.stdin); \
    print(f\"CUDA graph coverage: {d['cuda_graphs']['graph_coverage_pct']:.1f}%\")"
```

**See**: [NSIGHT_AUTOMATED_PROFILING_GUIDE.md](docs/NSIGHT_AUTOMATED_PROFILING_GUIDE.md) for complete details.

---

## Summary: Production Profiling Setup

### Recommended Command

```bash
export VLLM_LOGGING_LEVEL=INFO  # Or DEBUG for more detail
export VLLM_DEBUG_MFU_METRICS=1  # Optional: detailed breakdowns

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-model-len 2048 \
    --cudagraph-metrics \
    --enable-mfu-metrics \
    --enable-logging-iteration-details \
    2>&1 | tee vllm_production.log
```

**Total overhead**: ~1-2%

### What You Get

✓ CUDA graph padding stats
✓ Compute performance (TF/s)
✓ Memory bandwidth (GB/s)
✓ Prefill/decode breakdown
✓ Scheduling efficiency
✓ KV cache utilization
✓ Iteration timing

### Quick Analysis

```bash
# Overall stats
grep "Engine 000:" vllm_production.log | tail -20

# CUDA graphs
grep "CUDAGraph Stats" -A 10 vllm_production.log

# MFU details (if VLLM_DEBUG_MFU_METRICS=1)
grep "MFU details:" vllm_production.log | tail -1 | python -m json.tool
```

---

## See Also

- [ADVANCED_PROFILING_GUIDE.md](docs/ADVANCED_PROFILING_GUIDE.md) - Comprehensive guide with code locations
- [CUDA_GRAPHS.md](docs/CUDA_GRAPHS.md) - CUDA graph metrics details
- [KV_CACHE_GUIDE.md](docs/KV_CACHE_GUIDE.md) - KV cache architecture and profiling
