# ProfileMate Patching System - Complete Documentation

This directory contains comprehensive documentation about how ProfileMate's patching system works and how to troubleshoot issues.

## Quick Links

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[PATCH_FLOW_VISUAL.md](PATCH_FLOW_VISUAL.md)** | Visual one-page summary | Want a quick overview with diagrams |
| **[PATCH_ACTIVATION_FLOW.md](docs/PATCH_ACTIVATION_FLOW.md)** | Complete technical guide | Need detailed code flow explanation |
| **[TROUBLESHOOTING_PATCHES.md](TROUBLESHOOTING_PATCHES.md)** | Common issues and fixes | Patches aren't working |
| **[diagnose_patches.py](diagnose_patches.py)** | Diagnostic script | Want automated problem detection |

## Your Issues and Solutions

### Issue 1: "Patches not getting activated during my run"

**Run the diagnostic first:**
```bash
cd /home/nmiriyal/Documents/MLPERF-6.0/profilemate
python diagnose_patches.py
```

**Most likely causes:**

1. **PYTHONPATH not set**
   ```bash
   export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"
   ```

2. **Activation guard blocking your command**
   ```bash
   export VLLM_ENABLE_PROFILING=1  # Force enable
   ```

3. **vLLM didn't import those specific modules**
   - Some patches only activate for specific models/configurations
   - Example: MoE patches only for Mixtral models
   - Example: CUDA graph patches disabled with `--enforce-eager`
   - **This is normal!** Not all patches activate for every run.

**See:** [TROUBLESHOOTING_PATCHES.md](TROUBLESHOOTING_PATCHES.md) for detailed solutions

### Issue 2: "Patch getting activated for any python process"

**Good news:** This issue is already fixed!

The activation guard (lines 1488-1528 in sitecustomize.py) prevents ProfileMate from activating for non-vLLM processes.

**How it works:**
- ProfileMate only activates if:
  1. `VLLM_ENABLE_PROFILING=1` is explicitly set, OR
  2. Command line contains vLLM indicators (`vllm.entrypoints`, `--model`, etc.)

**Test it:**
```bash
# Should NOT activate
unset VLLM_ENABLE_PROFILING
python -c "print('Hello')"
# Expected: No ProfileMate messages

# Should activate
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf
# Expected: [sitecustomize] vLLM Comprehensive Instrumentation Loaded
```

**If it's still activating unexpectedly:**
Check if you have the environment variable set:
```bash
env | grep VLLM_ENABLE_PROFILING
# If you see VLLM_ENABLE_PROFILING=1, unset it:
unset VLLM_ENABLE_PROFILING
```

**See:** [TROUBLESHOOTING_PATCHES.md - Issue #5](TROUBLESHOOTING_PATCHES.md#common-issue-5-patch-activating-for-any-python-process)

### Issue 3: "Can't we activate all patches upfront instead of on first import?"

**Short answer:** No, because the modules don't exist yet.

**Explanation:**

When sitecustomize.py loads (at Python startup), vLLM hasn't been imported yet:

```python
# sitecustomize.py loads at time 0ms
# If we try to patch immediately:
from vllm.compilation.cuda_graph import CUDAGraphWrapper
# ❌ ImportError: No module named 'vllm'
# The module literally doesn't exist in memory yet!
```

**Timeline:**
```
0ms:   Python starts
5ms:   sitecustomize.py loads
       - vLLM modules don't exist yet! ❌
       - Can't patch what doesn't exist!

100ms: vLLM starts importing modules
       - NOW modules exist ✅
       - Import hook intercepts and patches immediately

1000ms: All modules loaded and patched ✅
```

**Why import hooks are necessary:**

Import hooks solve the "chicken and egg" problem:
- We need vLLM modules to exist before we can patch them
- But we need to patch them as soon as they're imported
- Import hooks intercept the import and patch immediately after loading

This is the **standard Python pattern** for transparent instrumentation used by tools like:
- pytest (test discovery)
- coverage.py (code coverage)
- debuggers (pdb, pydevd)
- APM tools (New Relic, DataDog)

**See:** [PATCH_ACTIVATION_FLOW.md - Why We Can't Activate All Patches Upfront](docs/PATCH_ACTIVATION_FLOW.md#why-we-cant-activate-all-patches-upfront)

## Complete Code Flow Documentation

For a complete understanding of how patches work:

1. **Start with visual guide:** [PATCH_FLOW_VISUAL.md](PATCH_FLOW_VISUAL.md)
   - One-page visual summary
   - Diagrams of all 6 phases
   - Quick reference checklist

2. **Deep dive:** [PATCH_ACTIVATION_FLOW.md](docs/PATCH_ACTIVATION_FLOW.md)
   - Complete startup sequence
   - Patch activation mechanism
   - Architecture diagrams
   - Debugging guide

3. **Troubleshoot:** [TROUBLESHOOTING_PATCHES.md](TROUBLESHOOTING_PATCHES.md)
   - Common issues and solutions
   - Step-by-step debugging
   - Advanced diagnostics

## Quick Start

### Step 1: Set PYTHONPATH
```bash
export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"

# Make it permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Run Diagnostic
```bash
python /home/nmiriyal/Documents/MLPERF-6.0/profilemate/diagnose_patches.py
```

This will check:
- PYTHONPATH configuration
- vLLM installation
- Activation environment variables
- Output directory permissions
- Target module availability

### Step 3: Run vLLM
```bash
# Force enable profiling (recommended for testing)
export VLLM_ENABLE_PROFILING=1

# Run vLLM
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000 \
    2>&1 | tee vllm_output.log
```

### Step 4: Verify Activation
```bash
# Check startup message
grep "sitecustomize" vllm_output.log
# Expected: [sitecustomize] vLLM Comprehensive Instrumentation Loaded

# Check patch messages
grep "Successfully patched" vllm_output.log
# Expected:
# [Instrumentation] ✅ Successfully patched CUDAGraphWrapper
# [Instrumentation] ✅ Successfully patched Scheduler
# ... etc
```

### Step 5: Send Test Request
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### Step 6: Check Output
```bash
# Find latest session
ls -lt /tmp/vllm_profiling/ | head -5

# Check CSV files
ls -lh /tmp/vllm_profiling/session_*/

# Verify data (should have more than just header row)
wc -l /tmp/vllm_profiling/session_*/cuda_graph_usage.csv
head -20 /tmp/vllm_profiling/session_*/cuda_graph_usage.csv
```

## Understanding Patch Behavior

### Not All Patches Activate for Every Run

This is **normal and expected!** Patches only activate when vLLM imports the corresponding modules.

| Patch | Activates When | Doesn't Activate When |
|-------|---------------|---------------------|
| **CUDA Graph** | Default vLLM run | `--enforce-eager` flag used |
| **KV Cache** | v1 engine used | Older vLLM versions |
| **Scheduler** | Always (for v1 engine) | v0 engine used |
| **GPU Model Runner** | v1 engine used | v0 engine used |
| **MoE** | Mixtral, DeepSeekMoE models | Non-MoE models (Llama, GPT) |
| **Block Pool** | v1 engine used | v0 engine used |

### Example: Why MoE Patch Doesn't Activate

```bash
# Running Llama model (not MoE)
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf

# vLLM will NOT import: vllm.model_executor.layers.fused_moe.layer
# Because Llama doesn't use MoE architecture
# Result: No MoE patch message ← This is CORRECT behavior!
```

```bash
# Running Mixtral model (uses MoE)
python -m vllm.entrypoints.openai.api_server --model mistralai/Mixtral-8x7B-v0.1

# vLLM WILL import: vllm.model_executor.layers.fused_moe.layer
# Because Mixtral uses MoE architecture
# Result: [Instrumentation] ✅ Successfully patched FusedMoE ← Success!
```

## Architecture Summary

### 6 Patch Functions

| Function | Target Module | Instruments | CSV Output |
|----------|--------------|-------------|------------|
| `patch_cuda_graph_wrapper()` | `vllm.compilation.cuda_graph` | CUDA graph captures/replays | `cuda_graph_*.csv` |
| `patch_kv_cache_manager()` | `vllm.v1.core.kv_cache_manager` | KV cache usage/evictions | `kv_cache_*.csv` |
| `patch_block_pool()` | `vllm.v1.core.block_pool` | Memory block allocations | (tracked in KV cache CSVs) |
| `patch_fused_moe()` | `vllm.model_executor.layers.fused_moe.layer` | Expert selections/timing | `moe_expert_*.csv` |
| `patch_gpu_model_runner()` | `vllm.v1.worker.gpu_model_runner` | Forward pass timing | `forward_pass_timing.csv` |
| `patch_scheduler()` | `vllm.v1.core.sched.scheduler` | Batch util, preemptions | `batch_utilization.csv`, `preemption_events.csv` |

### 8 Profiler Classes

All profilers are instantiated once at startup and shared across patches:

1. `CUDAGraphProfiler` - CUDA graph metrics
2. `KVCacheProfiler` - KV cache metrics
3. `MoEExpertProfiler` - MoE expert metrics
4. `ForwardPassProfiler` - Forward pass timing
5. `CPUTimingProfiler` - CPU operation timing
6. `BatchUtilizationProfiler` - Batch efficiency
7. `PreemptionProfiler` - Request preemptions
8. `EncoderDecoderProfiler` - Encoder-decoder timing

### Import Hook

`VllmInstrumentationHook` class (lines 1434-1478 in sitecustomize.py):
- Inserted into `sys.meta_path[0]` (first in import chain)
- Intercepts imports of target vLLM modules
- Applies appropriate patch function after module loads
- Returns patched module to vLLM

## File Structure

```
profilemate/
├── sitecustomize.py                    # Main implementation (1577 lines)
│   ├── Lines 45-67:    ProfilingConfig
│   ├── Lines 74-935:   Profiler classes (8 classes)
│   ├── Lines 1031-1422: Patch functions (6 functions)
│   ├── Lines 1434-1478: VllmInstrumentationHook
│   ├── Lines 1488-1528: should_activate_profiling()
│   └── Lines 1536-1577: Main activation block
│
├── diagnose_patches.py                 # Diagnostic script
├── README_PATCHES.md                   # This file
├── PATCH_FLOW_VISUAL.md               # Visual one-page summary
├── TROUBLESHOOTING_PATCHES.md         # Issue troubleshooting
│
└── docs/
    ├── PATCH_ACTIVATION_FLOW.md       # Complete technical guide
    └── ACTIVATION_CONTROL.md          # Activation control details
```

## Common Commands

### Check Configuration
```bash
# Verify PYTHONPATH
echo $PYTHONPATH | grep profilemate

# Check activation variable
echo $VLLM_ENABLE_PROFILING

# Run diagnostic
python diagnose_patches.py
```

### Force Enable/Disable
```bash
# Force enable (always activate)
export VLLM_ENABLE_PROFILING=1

# Force disable (never activate)
export VLLM_ENABLE_PROFILING=0

# Auto-detect (default)
unset VLLM_ENABLE_PROFILING
```

### Check Output
```bash
# List sessions
ls -lt /tmp/vllm_profiling/

# View latest session files
ls -lh /tmp/vllm_profiling/session_*/

# Check if CSVs have data
wc -l /tmp/vllm_profiling/session_*/cuda_graph_usage.csv

# View CUDA graph data
cat /tmp/vllm_profiling/session_*/cuda_graph_usage.csv
```

### Debug Activation
```bash
# Capture all output
python -m vllm.entrypoints.openai.api_server --model <model> 2>&1 | tee vllm_output.log

# Check startup
grep sitecustomize vllm_output.log

# Check patches
grep "Successfully patched" vllm_output.log

# Check for errors
grep -i error vllm_output.log
```

## Getting Help

If patches still aren't working after reading this documentation:

1. Run `python diagnose_patches.py` and save output
2. Capture vLLM startup: `python -m vllm... 2>&1 | tee vllm_output.log`
3. Check both files for clues
4. Share the following information:
   - Diagnostic output
   - vLLM command used
   - vllm_output.log (first 100 lines)
   - vLLM version: `pip show vllm`
   - Python version: `python --version`
   - ProfileMate session directory contents: `ls /tmp/vllm_profiling/session_*/`

## Additional Resources

- **Main README:** [README.md](README.md) - Feature overview and usage
- **Debug Guide:** [DEBUG_GUIDE.md](DEBUG_GUIDE.md) - Comprehensive debugging
- **Quick Debug:** [QUICK_DEBUG.txt](QUICK_DEBUG.txt) - Quick reference
- **vLLM 0.14.1 Fix:** [VLLM_0.14.1_FIX.md](VLLM_0.14.1_FIX.md) - Version compatibility
- **Activation Control:** [docs/ACTIVATION_CONTROL.md](docs/ACTIVATION_CONTROL.md) - Detailed activation logic
