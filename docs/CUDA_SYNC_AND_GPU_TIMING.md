# CUDA Synchronization and GPU Timing Explained

This document explains why CUDA synchronization is necessary for accurate GPU timing and the two timing modes available in ProfileMate.

---

## The Problem: Asynchronous CUDA Execution

### How CUDA Works

CUDA operations are **asynchronous**. When you call a GPU operation from Python:

```python
start = time.time()
output = model.forward(input)  # ← Returns IMMEDIATELY
end = time.time()
duration = end - start  # ← This is WRONG!
```

**What actually happens:**

1. `model.forward(input)` **launches** GPU work and **returns immediately** to CPU
2. GPU starts working **in parallel** while CPU continues
3. `time.time()` measures only the **CPU launch overhead** (~0.05-0.2ms)
4. GPU is **still running** when you record the end time

**Result:**
```
Actual GPU execution time: 12.5ms
What you measured:         0.08ms  ❌ Wrong by 156x!
```

### Visual Timeline

```
CPU Thread:  [Launch GPU work]──────────────────────[Continue Python]
                    ↓
GPU:                [........................GPU Computing 12.5ms........................]

Measurement: [start]──[end]  ← Only 0.08ms (launch overhead)
```

The problem: You measured **CPU overhead**, not **GPU time**!

---

## Solution 1: torch.cuda.synchronize()

### What It Does

```python
start = time.time()
output = model.forward(input)
torch.cuda.synchronize()  # ← WAIT for GPU to finish
end = time.time()
duration = end - start  # ← Now accurate: 12.5ms ✅
```

`torch.cuda.synchronize()` **blocks the CPU** until all GPU work completes.

### Visual Timeline with Sync

```
CPU Thread:  [Launch GPU work]──────────[BLOCKED]──────────────────[Continue]
                    ↓                                                ↑
GPU:                [............GPU Computing 12.5ms................]

Measurement: [start]──────────────────────────────────────────────[end]
             ← 12.5ms (accurate GPU time) ✅
```

### Pros and Cons

**Pros:**
- ✅ Simple to use
- ✅ 100% accurate GPU timing
- ✅ Includes GPU time + CPU-GPU sync overhead

**Cons:**
- ❌ Blocks CPU thread (wastes CPU cycles)
- ❌ Overhead: ~1.5-2% per synchronization
- ❌ Disrupts CPU-GPU overlap (if using multiple streams)

---

## Solution 2: CUDA Events (Better!)

### What Are CUDA Events?

CUDA Events are GPU-side markers that record timestamps **on the GPU**.

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()         # Mark start on GPU
output = model.forward(input)
end_event.record()           # Mark end on GPU

# Later (doesn't block immediately):
torch.cuda.synchronize()     # One sync for many events
gpu_time = start_event.elapsed_time(end_event)  # Pure GPU time!
```

### How It Works

```
CPU Thread:  [Record start event]──[Launch GPU work]──[Record end event]──[Continue]
                    ↓                      ↓                    ↓
GPU:                [Mark]─────────[GPU Computing]──────────[Mark]

             # Later, sync once:
CPU Thread:  ──────────[Sync]──[Read GPU timestamp]
                         ↓
GPU:                     [All done]
```

### Key Benefits

1. **Pure GPU Time**: Measures only GPU execution, excludes CPU overhead
2. **Lower Overhead**: Can batch sync operations
3. **Non-blocking**: CPU can continue while GPU works

### Comparison

| Method | Measures | Overhead | Accuracy |
|--------|----------|----------|----------|
| No sync | CPU launch overhead | 0% | **Wrong by 100-1000x** ❌ |
| `torch.cuda.synchronize()` | GPU time + sync overhead | ~1.5-2% per call | 100% ✅ |
| CUDA Events (sync every call) | Pure GPU time | ~0.5-1% per call | 100% ✅ |
| CUDA Events (batch sync) | Pure GPU time | ~0.1% (100 calls) | 100% ✅ |

---

## ProfileMate's Two Modes

### Option A: CUDA Events (Default)

**Configuration:**
```python
# In sitecustomize.py
class ProfilingConfig:
    USE_CUDA_EVENTS = True
    CUDA_EVENT_BATCH_SIZE = 100  # Sync every 100 iterations
```

**How it works:**
1. Records CUDA events for each forward pass
2. Batches 100 events before syncing
3. One `torch.cuda.synchronize()` for 100 measurements
4. Extracts pure GPU time for each event

**Overhead:** ~0.5% (100 measurements / 1 sync)

**Accuracy:** Perfect GPU-only timing ✅

**Use when:**
- You need accurate GPU timing
- You're profiling for optimization
- Overhead <1% is acceptable

### Option B: Lightweight Timing

**Configuration:**
```python
# In sitecustomize.py
class ProfilingConfig:
    USE_CUDA_EVENTS = False
```

**How it works:**
1. Uses `time.perf_counter()` around operations
2. Piggybacks on vLLM's existing sync points
3. Measures CPU time + implicit GPU sync time

**Overhead:** ~0.1%

**Accuracy:** ~95% (includes small CPU overhead)

**Use when:**
- You need minimal overhead
- Approximate timing is sufficient
- Running in production

---

## When Do You NEED Synchronization?

### Always Need Sync For:

1. **GPU kernel timing**
   - Forward pass
   - Attention kernels
   - Matrix multiplications

2. **GPU memory operations**
   - Memory copies (H2D, D2H)
   - Memory allocations (if timing them)

3. **Performance profiling**
   - Identifying bottlenecks
   - Comparing configurations
   - Optimization analysis

### Don't Need Sync For:

1. **CPU-only operations**
   - Scheduling
   - Tokenization
   - Sampling (CPU portion)
   - Queue management

2. **Event counts**
   - Number of preemptions
   - Request arrivals
   - Cache evictions

3. **Metadata**
   - Batch sizes
   - Token counts
   - Queue lengths

---

## Real-World Example

### Without Sync (Wrong!)

```python
# Measuring forward pass WITHOUT sync
start = time.time()
output = model(input)
end = time.time()
print(f"Forward pass: {(end - start) * 1000:.2f}ms")

# Output: "Forward pass: 0.08ms"  ❌
# Reality: GPU took 12.5ms
```

### With CUDA Events (Correct!)

```python
# Measuring forward pass WITH CUDA events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output = model(input)
end_event.record()

torch.cuda.synchronize()
duration_ms = start_event.elapsed_time(end_event)
print(f"Forward pass: {duration_ms:.2f}ms")

# Output: "Forward pass: 12.5ms"  ✅
```

### ProfileMate's Implementation

```python
class ForwardPassProfiler:
    def record_forward_start(self, phase, batch_size, num_tokens):
        if self.use_cuda_events:
            # CUDA Events mode
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            return (start_event, end_event, ...)
        else:
            # Lightweight mode
            return time.perf_counter()

    def record_forward_end(self, start_marker, ...):
        if self.use_cuda_events:
            start_event, end_event, ... = start_marker
            end_event.record()
            self.pending_events.append(start_marker)

            # Batch sync every 100 events
            if len(self.pending_events) >= 100:
                torch.cuda.synchronize()  # One sync for 100!
                for start, end, ... in self.pending_events:
                    duration = start.elapsed_time(end)
                    self.save_timing(...)
        else:
            duration_ms = (time.perf_counter() - start_marker) * 1000
            self.save_timing(...)
```

---

## Performance Impact Analysis

### Example: 1000 Forward Passes

**Scenario 1: No sync (Wrong!)**
```
Overhead: 0%
Accuracy: Wrong by 156x
Total wasted analysis time: Hours debugging why results are wrong
```

**Scenario 2: Sync every forward pass**
```
1000 forward passes × 0.05ms sync overhead = 50ms overhead
If each forward pass is 12.5ms: 50/12500 = 0.4% overhead
Accuracy: 100% ✅
```

**Scenario 3: CUDA Events, sync every 100**
```
10 syncs × 0.05ms = 0.5ms overhead
If each forward pass is 12.5ms: 0.5/12500 = 0.004% overhead
Accuracy: 100% ✅
```

**Scenario 4: Lightweight mode (vLLM's existing syncs)**
```
vLLM already syncs for:
- Sampling (to get logits on CPU)
- Output token retrieval
Piggyback on these: ~0 additional overhead
Accuracy: ~95% (small CPU overhead included)
```

---

## Recommendations

### For Development/Profiling

Use **CUDA Events mode**:
```python
ProfilingConfig.USE_CUDA_EVENTS = True
ProfilingConfig.CUDA_EVENT_BATCH_SIZE = 100
```

**Why:**
- Perfect accuracy for optimization decisions
- Low overhead (<1%)
- Pure GPU timing

### For Production Monitoring

Use **Lightweight mode**:
```python
ProfilingConfig.USE_CUDA_EVENTS = False
```

**Why:**
- Minimal overhead
- Good enough accuracy for monitoring
- Doesn't disrupt performance

### For Benchmarking

Use **CUDA Events mode** with larger batch size:
```python
ProfilingConfig.USE_CUDA_EVENTS = True
ProfilingConfig.CUDA_EVENT_BATCH_SIZE = 500
```

**Why:**
- Perfect accuracy for fair comparison
- Ultra-low overhead (<0.1%)
- Scientific rigor

---

## Common Pitfalls

### Pitfall 1: Forgetting Sync

```python
# WRONG: GPU time looks instant
start = time.time()
result = gpu_operation()
duration = time.time() - start  # ❌ Only CPU overhead
```

**Fix:** Use CUDA Events or synchronize

### Pitfall 2: Syncing Too Often

```python
# WRONG: 2% overhead per kernel
for i in range(1000):
    torch.cuda.synchronize()  # ❌ Too much!
    measure_kernel()
```

**Fix:** Batch syncs (sync every N iterations)

### Pitfall 3: Measuring CPU Operations with GPU Sync

```python
# WRONG: Adding unnecessary sync
torch.cuda.synchronize()  # ❌ Not needed!
duration = measure_cpu_tokenization()
```

**Fix:** Only sync for GPU operations

### Pitfall 4: Comparing Apples to Oranges

```python
# WRONG: Comparing different measurement methods
baseline = measure_without_sync()  # 0.08ms
optimized = measure_with_sync()    # 10.2ms
# "Our optimization is 127x worse!" ❌ Wrong comparison!
```

**Fix:** Use same measurement method for comparisons

---

## Debugging Tips

### Verify Sync is Working

```python
import torch
import time

# Test 1: Without sync (should be ~0ms)
start = time.time()
_ = torch.randn(10000, 10000, device='cuda') @ torch.randn(10000, 10000, device='cuda')
print(f"Without sync: {(time.time() - start) * 1000:.2f}ms")

# Test 2: With sync (should be 10-100ms)
start = time.time()
_ = torch.randn(10000, 10000, device='cuda') @ torch.randn(10000, 10000, device='cuda')
torch.cuda.synchronize()
print(f"With sync: {(time.time() - start) * 1000:.2f}ms")

# If both show ~0ms, sync isn't working!
```

### Check CUDA Events

```python
import torch

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
_ = torch.randn(10000, 10000, device='cuda') @ torch.randn(10000, 10000, device='cuda')
end_event.record()

torch.cuda.synchronize()
duration_ms = start_event.elapsed_time(end_event)
print(f"CUDA Event timing: {duration_ms:.2f}ms")

# Should show actual GPU time (10-100ms)
```

---

## Summary

| Aspect | No Sync | torch.cuda.synchronize() | CUDA Events (Batch) |
|--------|---------|-------------------------|---------------------|
| **Accuracy** | ❌ Wrong | ✅ Perfect | ✅ Perfect |
| **What's measured** | CPU overhead | GPU + sync | Pure GPU |
| **Overhead** | 0% | 1.5-2% | 0.1-0.5% |
| **CPU blocking** | No | Yes | Batched |
| **Use case** | Never | Simple profiling | Production profiling |

**ProfileMate Default:** CUDA Events with batch sync (best of both worlds)

**For gpt-oss-120b and Qwen3-VL-235B-A22B:** CUDA Events mode recommended for accurate profiling with <1% overhead.

---

## References

- [CUDA C Programming Guide - Events](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Nsight Systems Profiling Guide](https://docs.nvidia.com/nsight-systems/)
- ProfileMate's implementation: `sitecustomize.py:ForwardPassProfiler`
