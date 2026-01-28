# vLLM CUDA Graph Metrics Guide

## Overview

This document captures findings about CUDA graph usage tracking in vLLM, including built-in metrics, limitations, and workarounds for detailed tracking.

**See also**: [Advanced Profiling Guide](ADVANCED_PROFILING_GUIDE.md) for comprehensive profiling strategies including:
- Forward pass timing
- Scheduling efficiency metrics
- Prefill/decode breakup analysis
- GPU bandwidth estimation
- Impact of max_model_len on performance

## Table of Contents

- [CUDA Graph Metrics Support](#cuda-graph-metrics-support)
- [Enabling Metrics](#enabling-metrics)
- [What Gets Tracked](#what-gets-tracked)
- [Limitations](#limitations)
- [Tracking Individual CUDA Graphs](#tracking-individual-cuda-graphs)
- [Recommendations](#recommendations)
- [Code References](#code-references)

---

## CUDA Graph Metrics Support

vLLM has comprehensive support for printing CUDA graph usage statistics, added in **v0.13.0** (December 3, 2025).

### Feature Summary

- **Flag**: `--cudagraph-metrics`
- **Version Required**: vLLM v0.13.0+
- **Engine**: V1 engine architecture
- **Overhead**: Very low (<0.1%)
- **Configuration**: `vllm/vllm/config/observability.py:58-60`

### What It Tracks

The built-in metrics track:

1. **Unpadded Tokens**: Original number of tokens before padding
2. **Padded Tokens**: Number of tokens after padding for CUDA graph
3. **Num Paddings**: How many padding tokens were added
4. **Runtime Mode**: The CUDA graph mode used (FULL, PIECEWISE, or NONE)
5. **Count**: Frequency of each configuration

---

## Enabling Metrics

### Method 1: Python Module Syntax (Recommended)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --port 9999 \
    --cudagraph-metrics
```

### Method 2: vLLM CLI (Requires v0.13.0+)

```bash
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --port 9999 \
    --cudagraph-metrics
```

### Troubleshooting: "unrecognized arguments" Error

If you get `error: unrecognized arguments: --cudagraph-metrics`:

1. **Check vLLM version**:
   ```bash
   python -c "import vllm; print(vllm.__version__, vllm.__file__)"
   ```

2. **Upgrade if needed**:
   ```bash
   pip install 'vllm>=0.13.0' --upgrade
   ```

3. **Install from source** (if using local repository):
   ```bash
   cd /path/to/vllm
   pip install -e .
   ```

4. **Use Python module syntax** (works regardless):
   ```bash
   python -m vllm.entrypoints.openai.api_server --cudagraph-metrics ...
   ```

---

## What Gets Tracked

### Output Format

```
**CUDAGraph Config Settings:**

- Mode: CUDAGraphMode.FULL_AND_PIECEWISE
- Capture sizes: [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, ...]

**CUDAGraph Stats:**

| Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |
|-----------------|---------------|--------------|--------------|-------|
| 256             | 256           | 0            | FULL         | 450   |
| 127             | 128           | 1            | FULL         | 200   |
| 250             | 256           | 6            | FULL         | 150   |
| 500             | 512           | 12           | PIECEWISE    | 100   |
```

### Data Structure

The metrics are collected as `CUDAGraphStat` objects:

```python
@dataclasses.dataclass(frozen=True)
class CUDAGraphStat:
    num_unpadded_tokens: int
    num_padded_tokens: int
    num_paddings: int
    runtime_mode: str  # "FULL", "PIECEWISE", "NONE"
```

**Source**: `vllm/vllm/compilation/cuda_graph.py:27-31`

### How Stats Are Collected

1. **Collection Point**: `vllm/vllm/v1/worker/gpu_model_runner.py:3056-3062`
   ```python
   if self.vllm_config.observability_config.cudagraph_metrics:
       cudagraph_stats = CUDAGraphStat(
           num_unpadded_tokens=num_tokens,
           num_padded_tokens=batch_descriptor.num_tokens,
           num_paddings=batch_descriptor.num_tokens - num_tokens,
           runtime_mode=str(cudagraph_mode),
       )
   ```

2. **Aggregation**: `vllm/vllm/v1/metrics/loggers.py:179-182`
   ```python
   if (
       self.cudagraph_logging is not None
       and scheduler_stats.cudagraph_stats is not None
   ):
       self.cudagraph_logging.observe(scheduler_stats.cudagraph_stats)
   ```

3. **Logging**: `vllm/vllm/v1/metrics/loggers.py:264-265`
   ```python
   if self.cudagraph_logging is not None:
       self.cudagraph_logging.log(log_fn=log_fn)
   ```

---

## Limitations

### Critical Gap: Aggregation by Token Count

**The built-in metrics do NOT track individual CUDA graph instances.** They aggregate by token configuration.

#### How CUDA Graphs Are Actually Keyed

Each unique CUDA graph is keyed by a `BatchDescriptor`:

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int          # Padded token count
    num_reqs: int | None     # Number of requests
    uniform: bool            # All requests same length?
    has_lora: bool           # Has LoRA adapters?
```

**Source**: `vllm/vllm/forward_context.py:28-48`

CUDA graphs are stored per unique `BatchDescriptor`:

```python
# In CUDAGraphWrapper
self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}
```

**Source**: `vllm/vllm/compilation/cuda_graph.py:192`

#### What This Means

**Multiple distinct CUDA graphs can be counted together** if they have:
- Same padded token count
- Same runtime mode
- But different `num_reqs`, `uniform`, or `has_lora` values

#### Example

Two different CUDA graphs:
```python
Graph 1: BatchDescriptor(num_tokens=256, num_reqs=128, uniform=True, has_lora=False)
Graph 2: BatchDescriptor(num_tokens=256, num_reqs=64, uniform=False, has_lora=False)
```

Both appear as a **single aggregated row** in metrics:
```
| Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |
|-----------------|---------------|--------------|--------------|-------|
| 250             | 256           | 6            | FULL         | 150   |  <- Combined!
```

### What You Cannot Determine

From `--cudagraph-metrics` alone, you **cannot** answer:

1. How many unique CUDA graphs were created?
2. How many times each specific graph was replayed?
3. Which `BatchDescriptor` configurations are most common?
4. Whether graphs with different `num_reqs` or `uniform` settings exist

---

## Tracking Individual CUDA Graphs

### Method 1: Debug Logging (Most Accurate)

Enable debug logging to see each unique CUDA graph as it's captured:

```bash
VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --port 9999 \
    --cudagraph-metrics \
    2>&1 | tee vllm_cudagraph.log
```

#### Output Example

```
DEBUG: Capturing a cudagraph on (FULL, BatchDescriptor(num_tokens=256, num_reqs=128, uniform=True, has_lora=False))
DEBUG: Capturing a cudagraph on (FULL, BatchDescriptor(num_tokens=256, num_reqs=64, uniform=False, has_lora=False))
DEBUG: Capturing a cudagraph on (PIECEWISE, BatchDescriptor(num_tokens=512, num_reqs=None, uniform=False, has_lora=False))
INFO: Graph capturing finished in 5 secs, took 2.34 GiB
```

**Source**: `vllm/vllm/compilation/cuda_graph.py:234-243`

#### Analyze Debug Logs

```bash
# See all unique CUDA graphs captured
grep "Capturing a cudagraph" vllm_cudagraph.log

# Count unique graphs
grep "Capturing a cudagraph" vllm_cudagraph.log | sort -u | wc -l

# Extract all BatchDescriptor configurations
grep "Capturing a cudagraph" vllm_cudagraph.log | \
    sed 's/.*BatchDescriptor/BatchDescriptor/' | sort -u
```

### Method 2: Combine Debug Logging + Metrics

```bash
VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --port 9999 \
    --cudagraph-metrics \
    2>&1 | tee vllm_complete.log
```

**Benefits**:
- Debug logs show unique graph captures
- Metrics show aggregate usage patterns
- Combined view of graph creation and usage

**Analyze Combined Output**:

```bash
# See aggregate usage stats
grep "CUDAGraph Stats" -A 20 vllm_complete.log

# Count total unique graphs
grep "Capturing a cudagraph" vllm_complete.log | sort -u | wc -l

# Find which batch sizes have multiple graph variants
grep "Capturing a cudagraph" vllm_complete.log | \
    grep -oP 'num_tokens=\d+' | sort | uniq -c | sort -rn
```

### Method 3: Custom Instrumentation

The repository includes custom instrumentation in `vllm_instrumentation/`:

```bash
export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/vllm_instrumentation:$PYTHONPATH"

# Basic tracking
python -m vllm.entrypoints.openai.api_server \
    --model gpt2 \
    --cudagraph-metrics
```

**Available Tools**:
- `sitecustomize.py`: Basic graph usage tracking by batch size
- `advanced_sitecustomize.py`: Detailed profiling with CSV export
- `vllm_wrapper.py`: Wrapper-based approach

**Limitation**: These also aggregate by `num_tokens`, not full `BatchDescriptor`.

### Method 4: NVTX Profiling with Nsight Systems

For visual analysis of CUDA graph execution:

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

nsys profile \
    -t cuda,nvtx \
    --cuda-graph-trace=node \
    -o vllm_profile.qdrep \
    python -m vllm.entrypoints.openai.api_server \
        --model openai/gpt-oss-120b \
        --tensor-parallel-size 4 \
        --port 9999

# View in GUI
nsys-ui vllm_profile.qdrep
```

**Shows**:
- CUDA kernel launches
- NVTX markers for operations
- CUDA graph nodes and execution timeline
- Visual representation of graph replays

---

## Recommendations

### For Understanding Padding Overhead

Use the built-in `--cudagraph-metrics`:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    -tp 4 \
    --port 9999 \
    --cudagraph-metrics
```

**Good for**:
- Identifying inefficient batch sizes (high padding %)
- Understanding token distribution
- General usage patterns

### For Counting Unique CUDA Graphs

Use debug logging:

```bash
VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    -tp 4 \
    --port 9999 \
    2>&1 | grep "Capturing a cudagraph" | sort -u
```

**Good for**:
- Exact count of unique graphs
- Seeing full `BatchDescriptor` configurations
- Understanding graph diversity

### For Comprehensive Analysis

Combine all methods:

```bash
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --port 9999 \
    --cudagraph-metrics \
    2>&1 | tee vllm_analysis.log
```

**Provides**:
1. Unique graph captures (from DEBUG logs)
2. Aggregate usage patterns (from --cudagraph-metrics)
3. NVTX markers for profiling (optional nsys)
4. Complete execution trace

---

## Code References

### Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `vllm/config/observability.py` | Config flags | 58-60 |
| `vllm/compilation/cuda_graph.py` | CUDAGraphLogging class | 34-119 |
| `vllm/compilation/cuda_graph.py` | CUDAGraphWrapper class | 139-309 |
| `vllm/forward_context.py` | BatchDescriptor definition | 28-48 |
| `vllm/v1/worker/gpu_model_runner.py` | Stats collection | 3056-3062 |
| `vllm/v1/metrics/loggers.py` | Metrics integration | 111-116, 179-182, 264-265 |
| `vllm/engine/arg_utils.py` | CLI argument | 524, 1057-1058 |

### CUDA Graph Storage

CUDA graphs are stored per `BatchDescriptor`:

```python
# vllm/compilation/cuda_graph.py:192
self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}
```

Each entry contains:
```python
@dataclasses.dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None
    input_addresses: list[int] | None = None  # For debugging
```

### Graph Capture and Replay

**Capture** (first time for a BatchDescriptor):
- `vllm/compilation/cuda_graph.py:233-295`

**Replay** (subsequent uses):
- `vllm/compilation/cuda_graph.py:308`

```python
entry.cudagraph.replay()
```

---

## CUDA Graph Modes

vLLM supports multiple CUDA graph modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `NONE` | No CUDA graphs | Debugging |
| `PIECEWISE` | Partial graphs (attention eager) | Most compatible |
| `FULL` | Full graphs for all batches | Small models |
| `FULL_DECODE_ONLY` | Full graphs for decode only | Prefill/Decode split |
| `FULL_AND_PIECEWISE` | Full for decode, piecewise for mixed | Default, best performance |

**Configuration**: `vllm/config/compilation.py`

**Documentation**: `vllm/docs/design/cuda_graphs.md`

---

## Quick Decision Matrix

| I want to... | Solution | Overhead |
|--------------|----------|----------|
| See padding overhead | `--cudagraph-metrics` | <0.1% |
| Count unique graphs | `VLLM_LOGGING_LEVEL=DEBUG` | 1-2% |
| Profile GPU performance | `VLLM_NVTX_SCOPES_FOR_PROFILING=1` + nsys | <1% |
| Track replay latency | Custom instrumentation | <1% |
| Complete analysis | Combine all methods | ~2-3% |

---

## Version History

- **v0.13.0** (2025-12-03): Added `--cudagraph-metrics` flag
- Commit: `69520bc69` - "Add logging for cudagraph related info (#29825)"

---

## Additional Resources

- [vLLM CUDA Graphs Design Doc](../vllm/docs/design/cuda_graphs.md)
- [Observability Config](../vllm/vllm/config/observability.py)
- [CUDA Graph Profiling Guide](../VLLM_CUDA_GRAPH_AND_NVTX_GUIDE.md)
- [Quick Reference Card](../VLLM_PROFILING_QUICK_REFERENCE.md)
- [Custom Instrumentation](../vllm_instrumentation/README.md)

---

## Summary

### What `--cudagraph-metrics` Tracks

✅ Padding overhead
✅ Token distribution
✅ Runtime mode usage
✅ General usage patterns

### What It Does NOT Track

❌ Individual CUDA graph instances
❌ Per-graph replay counts
❌ Full `BatchDescriptor` details
❌ Graph-specific latencies

### Best Practice

**Use debug logging + metrics together** for complete visibility:

```bash
VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server \
    --model <your-model> \
    --cudagraph-metrics \
    2>&1 | tee vllm.log
```

Then analyze:
- Debug logs → Unique graph captures
- Metrics table → Aggregate usage patterns
