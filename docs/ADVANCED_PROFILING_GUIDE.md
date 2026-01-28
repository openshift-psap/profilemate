# vLLM Advanced Profiling and Performance Analysis Guide

## Table of Contents

1. [CUDA Graph Metrics](#cuda-graph-metrics)
2. [CUDA Graph Modes Explained](#cuda-graph-modes-explained)
3. [Forward Pass Timing](#forward-pass-timing)
4. [Scheduling Efficiency Metrics](#scheduling-efficiency-metrics)
5. [Prefill/Decode Breakup Analysis](#prefilldecode-breakup-analysis)
6. [GPU Bandwidth Estimation](#gpu-bandwidth-estimation)
7. [Impact of max_model_len on Performance](#impact-of-max_model_len-on-performance)

---

## 1. CUDA Graph Metrics

### What is `--cudagraph-metrics`?

The `--cudagraph-metrics` flag enables built-in CUDA graph usage statistics in vLLM (v0.13.0+). It tracks padding overhead and execution patterns with minimal performance impact (<0.1%).

### What Gets Tracked

**Location**: `vllm/vllm/compilation/cuda_graph.py:34-119`

The metrics track:
1. **Unpadded Tokens**: Original number of tokens before padding
2. **Padded Tokens**: Number of tokens after padding for CUDA graph
3. **Num Paddings**: How many padding tokens were added
4. **Runtime Mode**: CUDA graph mode used (`FULL`, `PIECEWISE`, or `NONE`)
5. **Count**: Frequency of each configuration

### Enable CUDA Graph Metrics

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --cudagraph-metrics \
    --port 8000
```

### Output Format

```
**CUDAGraph Config Settings:**

- Mode: CUDAGraphMode.FULL_DECODE_ONLY
- Capture sizes: [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, ...]

**CUDAGraph Stats:**

| Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |
|-----------------|---------------|--------------|--------------|-------|
| 256             | 256           | 0            | FULL         | 450   |
| 127             | 128           | 1            | FULL         | 200   |
| 63              | 64            | 1            | FULL         | 150   |
```

### Processing the Output

#### Calculate Padding Efficiency

```python
import re
from collections import defaultdict

# Read the log file
with open('vllm.log', 'r') as f:
    content = f.read()

# Extract stats table
pattern = r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|'
matches = re.findall(pattern, content)

# Calculate metrics
total_executions = 0
total_padding = 0
total_tokens = 0
mode_counts = defaultdict(int)

for unpadded, padded, num_pad, mode, count in matches:
    unpadded, padded, num_pad, count = map(int, [unpadded, padded, num_pad, count])

    total_executions += count
    total_padding += num_pad * count
    total_tokens += padded * count
    mode_counts[mode] += count

    padding_pct = (num_pad / padded * 100) if padded > 0 else 0
    print(f"Batch size {padded}: {padding_pct:.2f}% padding, {count} executions")

padding_overhead = (total_padding / total_tokens * 100) if total_tokens > 0 else 0
print(f"\nTotal executions: {total_executions}")
print(f"Total padding tokens: {total_padding}")
print(f"Overall padding overhead: {padding_overhead:.2f}%")
print(f"Mode distribution: {dict(mode_counts)}")
```

### Limitations

**From `profilemate/docs/CUDA_GRAPHS.md:156-215`:**

The built-in metrics **do NOT track**:
- Individual CUDA graph instances (metrics aggregate by token count)
- Per-graph replay counts
- Full `BatchDescriptor` details (num_reqs, uniform, has_lora)
- Graph-specific latencies

Multiple distinct CUDA graphs with the same `num_tokens` but different `num_reqs` or `uniform` flags will be **aggregated together**.

---

## 2. CUDA Graph Modes Explained

**Location**: `vllm/vllm/config/compilation.py:55-99`

### Available Modes

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0                          # No cudagraph capture
    PIECEWISE = 1                     # Partial graphs (attention eager)
    FULL = 2                          # Full graphs for all batches
    FULL_DECODE_ONLY = (FULL, NONE)  # Full for decode, none for mixed
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)  # Full for decode, piecewise for mixed (default)
```

### Detailed Explanation

#### 1. **NONE** - No CUDA Graphs
- **Use Case**: Debugging, development
- **Performance**: Slowest
- **Memory**: Lowest
- **Compatibility**: Works with all operations
- **When to use**: When profiling with NVTX layerwise tracing

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enforce-eager
```

#### 2. **PIECEWISE** - Partial Graphs
- **Use Case**: Most compatible mode
- **Performance**: Good
- **Memory**: Moderate
- **Compatibility**: Keeps attention ops outside cudagraph for flexibility
- **Graph Structure**: Splits at attention operations
- **When to use**: When attention backend doesn't support cudagraphs

```bash
# Explicitly set (usually auto-detected)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    # Piecewise is used automatically when attention backend requires it
```

**Key Detail**: Piecewise mode builds cudagraph only for non-attention operations, keeping attention ops (like `vllm::unified_attention`) outside the cudagraph for general flexibility.

#### 3. **FULL** - Complete Graphs
- **Use Case**: Small models, simple workloads
- **Performance**: Best (when compatible)
- **Memory**: Highest (captures graphs for all batch sizes)
- **Compatibility**: Requires attention backend with cudagraph support
- **When to use**: Small models or workloads with small prompts; not supported by many backends

```bash
# Note: Not all attention backends support this
python -m vllm.entrypoints.openai.api_server \
    --model gpt2 \  # Small model
    # FULL mode set via config
```

#### 4. **FULL_DECODE_ONLY** - Decode-Only Graphs
- **Use Case**: Disaggregated prefill/decode setups
- **Performance**: Excellent for decode-heavy workloads
- **Memory**: Moderate (only decode graphs)
- **Graph Behavior**:
  - **Decode batches** (single token per request): Full cudagraph
  - **Mixed prefill-decode batches**: No cudagraphs (runs eager)
- **When to use**: Decode instances in P/D setup where prefill isn't critical, saves memory

#### 5. **FULL_AND_PIECEWISE** (Default for V1)
- **Use Case**: Production workloads
- **Performance**: Best overall for most models
- **Memory**: High
- **Graph Behavior**:
  - **Decode-only batches**: Full cudagraph (everything captured)
  - **Prefill batches**: Piecewise cudagraph (attention ops excluded)
  - **Mixed prefill-decode batches**: Piecewise cudagraph
- **When to use**: Default for V1 engine, most performant for production

### Mode Selection Flowchart

```
┌─────────────────────────────────────────┐
│  Is attention backend cudagraph-safe?   │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │ NO             │ YES
       v                v
┌──────────────┐  ┌─────────────────────┐
│  PIECEWISE   │  │  Want max perf?     │
│  (required)  │  └──────┬──────────────┘
└──────────────┘         │
                 ┌───────┴────────┐
                 │ YES            │ NO
                 v                v
       ┌──────────────────┐  ┌─────────────────┐
       │ Small model &    │  │ Decode instance │
       │ prompts?         │  │ in P/D setup?   │
       └────┬─────────────┘  └───────┬─────────┘
            │                        │
    ┌───────┴────────┐       ┌───────┴────────────┐
    │ YES            │ NO    │ YES                │ NO
    v                v       v                    v
┌──────────┐  ┌─────────────────────┐  ┌────────────────┐  ┌──────────────────────┐
│   FULL   │  │ FULL_AND_PIECEWISE  │  │ FULL_DECODE   │  │ FULL_AND_PIECEWISE   │
│          │  │    (RECOMMENDED)    │  │     _ONLY      │  │    (RECOMMENDED)     │
└──────────┘  └─────────────────────┘  └────────────────┘  └──────────────────────┘
```

### Configuration File Reference

**Location**: `vllm/vllm/config/compilation.py:473-509`

```python
cudagraph_mode: CUDAGraphMode = Field(default=None)
"""
The mode of the cudagraph:

- NONE: no cudagraph capture
- PIECEWISE: piecewise cudagraph only, keeping cudagraph-incompatible ops
  (i.e. some attention ops) outside cudagraph
- FULL: Capture full cudagraph for all batches. Good for small models or
  workloads with small prompts; not supported by many backends.
- FULL_DECODE_ONLY: Capture full cudagraph for decode batches only.
  Mixed prefill-decode batches run without cudagraphs.
- FULL_AND_PIECEWISE: Capture full cudagraph for decode batches and
  piecewise for prefill and mixed batches. Most performant mode. (V1 default)
"""
```

---

## 3. Forward Pass Timing

### Can We Add Timers for Each Forward Pass?

**Yes!** vLLM already has comprehensive timing infrastructure.

### Option 1: Built-in MFU Metrics (Recommended)

**Flag**: `--enable-mfu-metrics`
**Overhead**: ~0.5-1%
**Location**: `vllm/vllm/v1/metrics/perf.py`

This calculates:
- **FLOPs per GPU** (compute operations)
- **Read bytes per GPU** (memory bandwidth - reads)
- **Write bytes per GPU** (memory bandwidth - writes)
- **Per-component breakdowns** (attention, FFN, unembed)

**Enable:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-mfu-metrics
```

**Output:**

```
Engine 000: MFU: 245.3 TF/s/GPU 892.1 GB/s/GPU
```

**With Debug Breakdown** (set `VLLM_DEBUG_MFU_METRICS=1`):

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
  },
  "flops_breakdown": {
    "attn.qkv_proj": "45.2TF",
    "attn.attn_qk": "102.3TF",
    "attn.attn_av": "98.1TF",
    "ffn.dense_ffn": "156.7TF"
  },
  "num_read_bytes_breakdown": {
    "attn.qkv_weight": "12.3GB",
    "attn.attn_input": "45.6GB",
    "ffn.dense_up_gate_weights": "78.9GB"
  }
}
```

### Option 2: Iteration Details Logging

**Flag**: `--enable-logging-iteration-details`
**Overhead**: <0.1%
**Location**: `vllm/vllm/config/observability.py:78-83`

```python
enable_logging_iteration_details: bool = False
"""Enable detailed logging of iteration details.
If set, vllm EngineCore will log iteration details.
This includes number of context/generation requests and tokens
and the elapsed cpu time for the iteration."""
```

**Enable:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-logging-iteration-details
```

**Location in code**: `vllm/vllm/v1/engine/core.py:342`

### Option 3: Custom Per-Forward-Pass Timer

**Where to add timers:**

#### Low Overhead Location (Recommended)
**File**: `vllm/vllm/v1/worker/gpu_model_runner.py`

Add timing around the model execution:

```python
# Around line 3000+
import time

# Before forward pass
start_time = time.perf_counter()

# Forward pass happens here
output = self.model.forward(...)

# After forward pass
forward_time = time.perf_counter() - start_time

# Log with minimal overhead
if self.vllm_config.observability_config.enable_custom_timing:
    self.timing_stats.append(forward_time)
```

**Overhead**: ~0.1% (using `time.perf_counter()`)

#### Detailed Location (Higher Overhead)
**File**: `vllm/vllm/v1/worker/gpu_model_runner.py`

Wrap the entire `execute_model` method:

```python
def execute_model(self, scheduler_output, ...):
    import time

    t_start = time.perf_counter()

    # ... existing code ...

    t_forward = time.perf_counter()

    # ... rest of execution ...

    t_end = time.perf_counter()

    logger.info(
        "Timing: prepare=%.3fms forward=%.3fms total=%.3fms",
        (t_forward - t_start) * 1000,
        (t_end - t_forward) * 1000,
        (t_end - t_start) * 1000
    )
```

**Overhead**: ~0.5-1% (includes synchronization points)

### Overhead Analysis

| Method | Overhead | Granularity | GPU Sync Required |
|--------|----------|-------------|-------------------|
| MFU Metrics | 0.5-1% | Per batch | No (analytical) |
| Iteration Details | <0.1% | Per iteration | No |
| `time.perf_counter()` | ~0.1% | Per call | No |
| `torch.cuda.synchronize()` | 5-10% | Per call | Yes (blocks!) |
| CUDA events | ~0.2% | Per event pair | Partial |

**Recommendation**: Use MFU metrics for production, custom timers for development.

### Where to Place Timers (Summary)

```
Good Locations (Low Overhead):
✓ vllm/v1/worker/gpu_model_runner.py:execute_model()  - Overall execution
✓ vllm/v1/worker/gpu_model_runner.py:~line 3000      - Model forward only
✓ vllm/v1/engine/core.py:step()                       - Full iteration

Avoid:
✗ Inside attention kernels (too frequent)
✗ Per-layer timing (use NVTX + Nsight instead)
✗ Anywhere requiring torch.cuda.synchronize() (huge overhead)
```

---

## 4. Scheduling Efficiency Metrics

### Does Debug Log Print Scheduling Steps?

**Yes!** Both built-in metrics and debug logging capture scheduling details.

### Option 1: Built-in Scheduler Stats

**Available metrics** (from `vllm/vllm/v1/metrics/stats.py:165-191`):

```python
@dataclass
class SchedulerStats:
    num_running_reqs: int = 0      # Currently running requests
    num_waiting_reqs: int = 0      # Queued requests

    kv_cache_usage: float = 0.0    # KV cache utilization (0.0-1.0)

    prefix_cache_stats: PrefixCacheStats  # Cache hit rates

    cudagraph_stats: CUDAGraphStat | None  # CUDA graph usage

    perf_stats: PerfStats | None   # MFU metrics
```

**Access** (automatically logged at every interval):

```
Engine 000: Avg prompt throughput: 1250.3 tokens/s, Avg generation throughput: 892.7 tokens/s,
Running: 45 reqs, Waiting: 12 reqs, Preemptions: 3, GPU KV cache usage: 72.5%,
Prefix cache hit rate: 45.2%
```

### Option 2: Enable Detailed Iteration Logging

**Flag**: `--enable-logging-iteration-details`

**What it logs** (from `vllm/vllm/config/observability.py:78-83`):

```python
"""Enable detailed logging of iteration details.
If set, vllm EngineCore will log iteration details.
This includes:
- Number of context/generation requests
- Number of tokens processed
- Elapsed CPU time for the iteration
"""
```

**Enable:**

```bash
VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-logging-iteration-details
```

**Expected output format** (location: `vllm/vllm/v1/engine/core.py:342+`):

```
DEBUG: Iteration 1234: scheduled_new_reqs=5 scheduled_cached_reqs=45
       num_batched_tokens=2048 elapsed_cpu_ms=12.3
```

### Option 3: Prometheus Metrics

**Metrics available**:

```python
# From vllm/v1/metrics/loggers.py:387-1190
gauge_scheduler_running     # Number of running requests
gauge_scheduler_waiting     # Number of waiting requests
gauge_kv_cache_usage        # KV cache usage percentage
counter_num_preempted_reqs  # Total preemptions
histogram_iteration_tokens  # Tokens per iteration distribution
```

**Access via**:

```bash
curl http://localhost:8000/metrics | grep vllm
```

**Example output:**

```prometheus
vllm:num_requests_running{model_name="meta-llama/Llama-2-7b-hf",engine="0"} 45
vllm:num_requests_waiting{model_name="meta-llama/Llama-2-7b-hf",engine="0"} 12
vllm:kv_cache_usage_perc{model_name="meta-llama/Llama-2-7b-hf",engine="0"} 0.725
vllm:num_preemptions{model_name="meta-llama/Llama-2-7b-hf",engine="0"} 3
vllm:iteration_tokens_total_bucket{le="512",model_name="...",engine="0"} 1234
```

### What You Can Measure

#### Scheduling Efficiency Metrics

**Calculate batch utilization:**

```python
# From logs
avg_batch_size = num_batched_tokens / num_scheduled_reqs
max_theoretical = max_num_batched_tokens
utilization = (avg_batch_size / max_theoretical) * 100
```

**Track preemption rate:**

```python
preemption_rate = num_preemptions / total_iterations
```

**Monitor queue depth:**

```python
queue_pressure = num_waiting_reqs / num_running_reqs
```

### Capturing Scheduling Details

#### Method 1: Parse Standard Logs

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --cudagraph-metrics \
    --enable-logging-iteration-details \
    2>&1 | tee vllm_scheduling.log

# Extract stats
grep "Running:" vllm_scheduling.log | \
    awk '{print $5, $8, $11}'  # running, waiting, kv_usage
```

#### Method 2: Prometheus + Grafana

1. Scrape metrics endpoint
2. Visualize in Grafana:
   - Running vs Waiting requests over time
   - KV cache usage trends
   - Preemption frequency
   - Batch size distribution

#### Method 3: Custom Logging

Add to `vllm/vllm/v1/engine/core.py`:

```python
def step(self) -> list[EngineCoreOutput]:
    scheduler_output = self.scheduler.schedule()

    # Custom logging
    if self.vllm_config.observability_config.enable_logging_iteration_details:
        logger.info(
            "Scheduling: new_reqs=%d cached_reqs=%d total_tokens=%d "
            "kv_usage=%.1f%% waiting=%d",
            len(scheduler_output.scheduled_new_reqs),
            len(scheduler_output.scheduled_cached_reqs.req_ids),
            sum(scheduler_output.num_scheduled_tokens.values()),
            self.scheduler.kv_cache_usage * 100,
            len(self.scheduler.waiting_queue)
        )
```

---

## 5. Prefill/Decode Breakup Analysis

### Can We Track Prefill vs Decode?

**Yes!** vLLM tracks prefill/decode phases extensively.

### Built-in Metrics

**From `vllm/vllm/v1/metrics/perf.py:45-63`:**

```python
@dataclass
class DebugPerfStats:
    num_prefill_requests: int = 0
    num_decode_requests: int = 0
    context_breakdown: dict[str, int] | None = None
```

**From `vllm/vllm/v1/metrics/perf.py:80-104`:**

```python
class ExecutionContext:
    """Aggregates statistics across requests, separately tracking prefill and decode."""

    # Prefill phase statistics
    num_prefill_requests: int = 0
    prefill_num_tokens: int = 0        # Total tokens in prefill
    prefill_context_len: int = 0       # Total context for prefill
    prefill_token_context_product: int = 0

    # Decode phase statistics
    num_decode_requests: int = 0
    decode_num_tokens: int = 0         # Total tokens in decode
    decode_context_len: int = 0        # Total context for decode
    decode_token_context_product: int = 0
```

### Enable Prefill/Decode Tracking

#### Option 1: MFU Metrics with Debug Output

```bash
export VLLM_DEBUG_MFU_METRICS=1

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-mfu-metrics \
    2>&1 | tee vllm_mfu_debug.log
```

**Output:**

```json
{
  "prefill_reqs": 10,
  "decode_reqs": 50,
  "num_batches": 60,
  "context_breakdown": {
    "num_prefill_requests": 10,
    "prefill_num_tokens": 20480,
    "prefill_context_len": 20480,
    "prefill_token_context_product": 419430400,
    "num_decode_requests": 50,
    "decode_num_tokens": 50,
    "decode_context_len": 409600,
    "decode_token_context_product": 20480000
  },
  "flops_breakdown": {
    "attn.qkv_proj": "45.2TF",
    "attn.attn_qk": "102.3TF",
    "attn.attn_av": "98.1TF",
    "attn.out_proj": "38.7TF",
    "ffn.dense_ffn": "156.7TF"
  },
  "num_read_bytes_breakdown": {
    "attn.qkv_input": "2.5GB",
    "attn.qkv_weight": "12.3GB",
    "attn.attn_input": "45.6GB",
    "attn.out_input": "8.9GB",
    "attn.out_weight": "12.3GB",
    "ffn.dense_up_gate_input": "2.5GB",
    "ffn.dense_up_gate_weights": "78.9GB"
  },
  "duration": "5.2s",
  "mfu_calc_overhead": "0.08%"
}
```

#### Option 2: Prometheus Metrics

Prometheus exposes separate histograms:

```python
# From vllm/v1/metrics/loggers.py:868-886
histogram_prefill_time_request  # Time spent in prefill phase
histogram_decode_time_request   # Time spent in decode phase
histogram_prefill_kv_computed_request  # KV tokens computed during prefill
```

**Query:**

```bash
curl http://localhost:8000/metrics | grep -E "(prefill|decode)"
```

**Example:**

```prometheus
vllm:request_prefill_time_seconds_bucket{le="0.5",...} 1234
vllm:request_decode_time_seconds_bucket{le="2.5",...} 5678
vllm:request_prefill_kv_computed_tokens_bucket{le="512",...} 890
```

#### Option 3: Per-Request Finished Stats

**From `vllm/vllm/v1/metrics/stats.py:215-231`:**

```python
@dataclass
class FinishedRequestStats:
    e2e_latency: float = 0.0
    num_prompt_tokens: int = 0          # Prefill tokens
    num_generation_tokens: int = 0      # Decode tokens
    queued_time: float = 0.0
    prefill_time: float = 0.0           # Time in prefill phase
    inference_time: float = 0.0         # Total inference
    decode_time: float = 0.0            # Time in decode phase
    mean_time_per_output_token: float = 0.0
    num_cached_tokens: int = 0          # From prefix cache
```

**These stats are logged** for every completed request to Prometheus histograms.

### Analyzing Prefill/Decode Efficiency

#### Calculate Prefill vs Decode Ratio

```python
import json

# Parse MFU debug output
with open('vllm_mfu_debug.log') as f:
    for line in f:
        if 'context_breakdown' in line:
            data = json.loads(line.split('MFU details: ')[1])

            prefill_tokens = data['context_breakdown']['prefill_num_tokens']
            decode_tokens = data['context_breakdown']['decode_num_tokens']

            total = prefill_tokens + decode_tokens
            prefill_ratio = prefill_tokens / total * 100
            decode_ratio = decode_tokens / total * 100

            print(f"Prefill: {prefill_ratio:.1f}% ({prefill_tokens} tokens)")
            print(f"Decode: {decode_ratio:.1f}% ({decode_tokens} tokens)")
```

#### Identify Bottlenecks

```python
# From Prometheus metrics
prefill_p50 = get_histogram_percentile('vllm:request_prefill_time_seconds', 0.5)
decode_p50 = get_histogram_percentile('vllm:request_decode_time_seconds', 0.5)

if prefill_p50 > decode_p50 * 10:
    print("Prefill is bottleneck - consider chunked prefill or disaggregated setup")
elif decode_p50 > prefill_p50 * 10:
    print("Decode is bottleneck - check KV cache efficiency")
```

### Implementation Location

**Where prefill/decode is determined** (from `vllm/vllm/v1/attention/backends/utils.py:595-703`):

```python
def split_by_stage(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    """
    Returns:
        num_decodes: Number of decode requests
        num_prefills: Number of prefill requests
        num_extends: Number of extend requests
        num_decode_tokens: Tokens in decode requests
        num_extend_tokens: Tokens in extend requests
        num_prefill_tokens: Tokens in prefill requests
    """
```

**This function is called** in every attention backend to split the batch.

---

## 6. GPU Bandwidth Estimation

### Can We Calculate GPU Bandwidth?

**Yes!** vLLM's MFU metrics include bandwidth calculations.

### Built-in Bandwidth Metrics

**Enable:**

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
- **245.3 TF/s/GPU** = Compute throughput (FLOPs)
- **892.1 GB/s/GPU** = Memory bandwidth (read + write)

### How It's Calculated

**Location**: `vllm/vllm/v1/metrics/perf.py:1193-1220`

```python
def log(self, log_fn=logger.info, log_prefix: str = "") -> None:
    now = time.monotonic()
    delta_time = now - self.last_log_time

    # Compute bandwidth
    avg_tflops_per_gpu = self.total_num_flops_per_gpu / delta_time / 1e12
    avg_gbps_per_gpu = (
        (self.total_read_bytes_per_gpu + self.total_write_bytes_per_gpu)
        / delta_time
        / 1e9
    )

    log_fn(
        "%sMFU: %.1f TF/s/GPU %.1f GB/s/GPU",
        log_prefix,
        avg_tflops_per_gpu,
        avg_gbps_per_gpu,
    )
```

### Detailed Breakdown

**With `VLLM_DEBUG_MFU_METRICS=1`:**

```json
{
  "num_read_bytes_breakdown": {
    "attn.qkv_input": "2.5GB",      # Read activations
    "attn.qkv_weight": "12.3GB",    # Read weights
    "attn.attn_input": "45.6GB",    # Read KV cache
    "ffn.dense_up_gate_weights": "78.9GB"
  },
  "num_write_bytes_breakdown": {
    "attn.qkv_output": "3.1GB",     # Write activations
    "attn.kv_cache": "6.2GB",       # Write to KV cache
    "ffn.dense_up_gate_output": "9.8GB"
  }
}
```

### Per-Kernel Bandwidth (Advanced)

**For kernel-level bandwidth**, use NVIDIA Nsight Compute:

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

# Profile specific kernels
ncu --set full --target-processes all \
    --kernel-name regex:"attention|gemm|flash" \
    -o vllm_bandwidth_profile \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-hf

# View report
ncu-ui vllm_bandwidth_profile.ncu-rep
```

**Metrics to look for:**
- `dram__bytes_read.sum` - Total DRAM reads
- `dram__bytes_write.sum` - Total DRAM writes
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` - % of peak bandwidth
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` - Load efficiency
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` - Store efficiency

### Bandwidth Analysis Per Component

```python
# From vllm/v1/metrics/perf.py:359-486
class AttentionMetrics:
    def get_read_bytes_breakdown(self, ctx: ExecutionContext, per_gpu: bool = True):
        # Attention reads differ between prefill and decode

        # Prefill: read Q, K, V activations (all in activation_byte_size)
        if ctx.prefill_num_tokens > 0:
            read_bytes["attn_input"] = (
                (ctx.prefill_num_tokens * q + 2 * ctx.prefill_context_len * kv)
                * d * self.activation_byte_size * L
            )

        # Decode: read Q activations + read K, V from cache (in cache_byte_size)
        if ctx.decode_num_tokens > 0:
            read_bytes["attn_input"] = (
                ctx.decode_num_tokens * q * d * self.activation_byte_size * L
                + 2 * ctx.decode_context_len * kv * d * self.cache_byte_size * L
            )
```

**Key insight**: Decode phase reads from KV cache (potentially slower if cache_dtype is quantized).

### Compare Against Hardware Specs

```python
# Hardware specs (example: A100-80GB)
HW_PEAK_BANDWIDTH = 2000  # GB/s (HBM2e)
HW_PEAK_TFLOPS = 312      # TF/s (FP16 with Tensor Cores)

# From vLLM metrics
measured_bandwidth = 892.1  # GB/s
measured_tflops = 245.3     # TF/s

bandwidth_utilization = (measured_bandwidth / HW_PEAK_BANDWIDTH) * 100
compute_utilization = (measured_tflops / HW_PEAK_TFLOPS) * 100

print(f"Bandwidth utilization: {bandwidth_utilization:.1f}%")
print(f"Compute utilization: {compute_utilization:.1f}%")

if bandwidth_utilization > compute_utilization:
    print("Memory-bound workload")
else:
    print("Compute-bound workload")
```

### Rough Bandwidth Calculation

**Formula:**

```
Bandwidth (GB/s) = (Total Bytes Transferred) / (Time in seconds)

Total Bytes = Model Weights Read + KV Cache Access + Activation R/W
```

**Example calculation:**

```python
# Model: Llama-2-7B
model_size_gb = 14  # 7B params * 2 bytes (FP16)
kv_cache_gb = 8     # Typical for batch
activations_gb = 2  # Intermediate tensors

# Per forward pass
total_bytes_per_forward = (
    model_size_gb +      # Read model weights
    kv_cache_gb * 2 +    # Read + Write KV cache
    activations_gb * 2   # Read + Write activations
)

# If forward pass takes 10ms
forward_time_s = 0.010
bandwidth_gb_s = total_bytes_per_forward / forward_time_s

print(f"Estimated bandwidth: {bandwidth_gb_s:.1f} GB/s")
```

---

## 7. Impact of max_model_len on Performance

### How Does max_model_len Affect Performance?

**Critical parameter** that affects:
1. KV cache memory allocation
2. CUDA graph capture sizes
3. Attention computation complexity
4. Memory bandwidth requirements

### Memory Impact

**Formula** (from KV cache allocation):

```python
kv_cache_size = (
    max_model_len             # Maximum sequence length
    * num_layers              # Number of transformer layers
    * num_kv_heads            # Number of KV heads
    * head_dim                # Head dimension
    * 2                       # Separate K and V
    * dtype_bytes             # FP16=2, FP8=1
    * max_num_seqs            # Maximum batch size
)
```

**Example** (Llama-2-7B):

```python
# Configuration
max_model_len = 4096
num_layers = 32
num_kv_heads = 32
head_dim = 128
dtype_bytes = 2  # FP16
max_num_seqs = 256

kv_cache_gb = (
    max_model_len * num_layers * num_kv_heads * head_dim * 2 * dtype_bytes * max_num_seqs
) / (1024 ** 3)

print(f"KV cache: {kv_cache_gb:.2f} GB")

# Output: KV cache: 64.00 GB
```

**Doubling `max_model_len`** doubles KV cache memory!

### Performance Impact

#### 1. Attention Complexity

**Attention FLOPs** scales **quadratically** with sequence length (for prefill):

```
FLOPs_attention = 2 * num_heads * seq_len^2 * head_dim
```

**Example:**

```python
num_heads = 32
head_dim = 128

# max_model_len = 2048
flops_2k = 2 * num_heads * (2048 ** 2) * head_dim
# = 34.4 GFLOPs

# max_model_len = 4096
flops_4k = 2 * num_heads * (4096 ** 2) * head_dim
# = 137.4 GFLOPs

# 4x increase!
```

**For decode**, complexity is **linear** (only computing attention for 1 new token against full context):

```
FLOPs_decode = 2 * num_heads * context_len * head_dim
```

#### 2. CUDA Graph Capture

**From `vllm/vllm/config/compilation.py:559-574`:**

```python
max_cudagraph_capture_size: int | None = field(default=None)
"""The maximum cudagraph capture size.

If not specified, max_cudagraph_capture_size is set to min(max_num_seqs*2, 512)
by default. This avoids OOM in tight memory scenarios with small max_num_seqs.
"""
```

**Impact:** Larger `max_model_len` may require capturing more CUDA graph sizes, increasing startup time and memory.

#### 3. Memory Bandwidth

**KV cache bandwidth** during decode:

```python
# Read bandwidth for attention
kv_read_bytes = 2 * context_len * num_kv_heads * head_dim * dtype_bytes

# Example: context_len = 4096
kv_read_4k = 2 * 4096 * 32 * 128 * 2 / (1024 ** 2)
# = 64 MB per request

# If max_model_len = 8192
kv_read_8k = 2 * 8192 * 32 * 128 * 2 / (1024 ** 2)
# = 128 MB per request (2x)
```

### Measuring the Impact

#### Experiment Setup

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

#### Metrics to Compare

1. **Memory usage:**
```bash
grep "GPU KV cache usage" vllm_*.log
```

2. **Throughput:**
```bash
grep "Avg generation throughput" vllm_*.log
```

3. **Bandwidth:**
```bash
grep "MFU:" vllm_*.log | grep "GB/s"
```

4. **CUDA graph stats:**
```bash
grep "CUDAGraph Stats" -A 10 vllm_*.log
```

### Optimization Strategies

#### 1. Use Appropriate max_model_len

```bash
# Don't over-provision
# If your workload has max 2K tokens, don't set max_model_len=8K

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-model-len 2048  # Match your actual needs
```

#### 2. Enable Prefix Caching

```bash
# Reduces effective context length via caching
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-model-len 4096 \
    --enable-prefix-caching
```

#### 3. Use Quantization

```bash
# FP8 KV cache = half the memory
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-model-len 4096 \
    --kv-cache-dtype fp8
```

#### 4. Chunked Prefill

```bash
# Break long prefills into chunks
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096  # Chunk size
```

### Expected Performance Changes

**Rule of thumb:**

| max_model_len | KV Cache Mem | Decode Bandwidth | Prefill FLOPs | Recommended Use Case |
|---------------|--------------|------------------|---------------|----------------------|
| 512           | ~4 GB        | Low              | Very Low      | Chat (short context) |
| 2048          | ~16 GB       | Moderate         | Low           | Standard chat        |
| 4096          | ~32 GB       | High             | Moderate      | Document Q&A         |
| 8192          | ~64 GB       | Very High        | High          | Long documents       |
| 16384         | ~128 GB      | Extreme          | Very High     | Multi-doc reasoning  |

**Performance degradation:**
- **2K → 4K**: ~10-15% throughput decrease (decode)
- **4K → 8K**: ~15-20% throughput decrease (decode)
- Prefill is more sensitive: ~4x FLOPs when doubling length

### Code Reference

**Where max_model_len is used:**

1. **KV cache allocation**: `vllm/vllm/v1/worker/gpu_model_runner.py`
2. **Attention backend**: `vllm/vllm/v1/attention/backends/*.py`
3. **Scheduler constraints**: `vllm/vllm/v1/core/sched/scheduler.py`
4. **CUDA graph sizes**: `vllm/vllm/config/compilation.py`

---

## Summary Table

| Feature | Flag/Environment | Overhead | Use Case |
|---------|-----------------|----------|----------|
| CUDA Graph Metrics | `--cudagraph-metrics` | <0.1% | Padding analysis |
| MFU Metrics | `--enable-mfu-metrics` | 0.5-1% | Compute & bandwidth |
| MFU Debug | `VLLM_DEBUG_MFU_METRICS=1` | 0.5-1% | Detailed breakdowns |
| Iteration Details | `--enable-logging-iteration-details` | <0.1% | Scheduling stats |
| Debug Logging | `VLLM_LOGGING_LEVEL=DEBUG` | 1-2% | CUDA graph captures |
| NVTX Scopes | `VLLM_NVTX_SCOPES_FOR_PROFILING=1` | <1% | Nsight profiling |
| Layerwise NVTX | `--enable-layerwise-nvtx-tracing` | 5-10% | Per-layer profiling (no CG) |

## Best Practices

### Production Setup

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-model-len 2048 \
    --cudagraph-metrics \
    --enable-mfu-metrics \
    2>&1 | tee vllm_production.log
```

**Overhead**: ~1% total

### Development/Debugging Setup

```bash
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_DEBUG_MFU_METRICS=1

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --cudagraph-metrics \
    --enable-mfu-metrics \
    --enable-logging-iteration-details \
    2>&1 | tee vllm_debug.log
```

**Overhead**: ~2-3% total

### Profiling Setup (One-time Analysis)

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

nsys profile \
    -t cuda,nvtx \
    --cuda-graph-trace=node \
    -o vllm_profile.qdrep \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-hf \
        --cudagraph-metrics \
        --max-num-seqs 64

# View in GUI
nsys-ui vllm_profile.qdrep
```

**Overhead**: 10-20% (acceptable for one-time profiling)

---

## References

- **CUDA Graph Modes**: vllm/vllm/config/compilation.py:55-99
- **CUDA Graph Metrics**: vllm/vllm/compilation/cuda_graph.py:34-119
- **MFU Metrics**: vllm/vllm/v1/metrics/perf.py
- **Scheduler Stats**: vllm/vllm/v1/metrics/stats.py:165-191
- **Prometheus Metrics**: vllm/vllm/v1/metrics/loggers.py:387-1190
- **Prefill/Decode Split**: vllm/vllm/v1/attention/backends/utils.py:595-703
