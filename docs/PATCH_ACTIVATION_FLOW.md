# ProfileMate Patch Activation Flow - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Complete Startup Flow](#complete-startup-flow)
3. [Patch Activation Mechanism](#patch-activation-mechanism)
4. [Why Patches Might Not Activate](#why-patches-might-not-activate)
5. [Why We Can't Activate All Patches Upfront](#why-we-cant-activate-all-patches-upfront)
6. [Debugging Non-Activating Patches](#debugging-non-activating-patches)
7. [Architecture Diagrams](#architecture-diagrams)

---

## Overview

ProfileMate uses **Python import hooks** to transparently instrument vLLM at runtime. This approach:
- ✅ Requires zero code changes to vLLM
- ✅ Works across vLLM versions
- ✅ Activates patches lazily as modules load
- ✅ Avoids profiling non-vLLM Python processes

**Key Insight:** Patches CANNOT be activated upfront because the target modules don't exist yet. They must be applied when vLLM imports its modules.

---

## Complete Startup Flow

### Phase 1: Python Interpreter Starts
```
1. Python process launches (e.g., python -m vllm.entrypoints.openai.api_server)
   │
   ├─→ Python reads PYTHONPATH environment variable
   │   PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"
   │
   └─→ Python automatically imports sitecustomize.py from PYTHONPATH
       (This happens BEFORE any user code runs)
```

**File:** `profilemate/sitecustomize.py`
**Lines executed:** 1-1577 (entire module loads)

### Phase 2: Activation Decision (sitecustomize.py loads)
```
2. sitecustomize.py top-level code executes
   │
   ├─→ Define all classes (lines 70-1425)
   │   ├─ ProfilingConfig
   │   ├─ CUDAGraphProfiler
   │   ├─ KVCacheProfiler
   │   ├─ MoEExpertProfiler
   │   ├─ ForwardPassProfiler
   │   ├─ CPUTimingProfiler
   │   ├─ BatchUtilizationProfiler
   │   ├─ PreemptionProfiler
   │   ├─ EncoderDecoderProfiler
   │   └─ VllmInstrumentationHook
   │
   ├─→ Define all patch functions (lines 1031-1422)
   │   ├─ patch_cuda_graph_wrapper()
   │   ├─ patch_kv_cache_manager()
   │   ├─ patch_block_pool()
   │   ├─ patch_fused_moe()
   │   ├─ patch_gpu_model_runner()
   │   └─ patch_scheduler()
   │
   └─→ Execute main block (lines 1536-1577)
       │
       └─→ if should_activate_profiling():  # KEY DECISION POINT
           │
           ├─→ YES: Continue to Phase 3
           └─→ NO: Exit silently (non-vLLM process)
```

**Critical Function:** `should_activate_profiling()` (lines 1488-1528)

```python
def should_activate_profiling():
    # Priority 1: Explicit control via environment variable
    explicit_enable = os.getenv("VLLM_ENABLE_PROFILING")
    if explicit_enable is not None:
        return explicit_enable == "1"

    # Priority 2: Auto-detect vLLM from command line
    cmdline = ' '.join(sys.argv)
    vllm_indicators = [
        'vllm.entrypoints',  # python -m vllm.entrypoints.openai.api_server
        'vllm.engine',       # python -m vllm.engine.llm_engine
        'vllm',              # python run_vllm.py (if script imports vllm)
        '--model'            # Common vLLM CLI argument
    ]
    for indicator in vllm_indicators:
        if indicator in cmdline:
            return True

    # Priority 3: Check if vLLM is even installed
    try:
        import importlib.util
        spec = importlib.util.find_spec('vllm')
        if spec is None:
            return False
    except ImportError:
        return False

    # Default: Don't activate (safety)
    return False
```

**Decision Examples:**

| Command | `sys.argv` | Decision | Reason |
|---------|-----------|----------|--------|
| `python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf` | `['-m', 'vllm.entrypoints.openai.api_server', '--model', 'meta-llama/Llama-2-7b-hf']` | ✅ ACTIVATE | Contains 'vllm.entrypoints' and '--model' |
| `python my_script.py` | `['my_script.py']` | ❌ DON'T ACTIVATE | No vLLM indicators |
| `VLLM_ENABLE_PROFILING=1 python my_script.py` | `['my_script.py']` | ✅ ACTIVATE | Explicit enable |
| `VLLM_ENABLE_PROFILING=0 python -m vllm.entrypoints.openai.api_server` | `['-m', 'vllm.entrypoints.openai.api_server', ...]` | ❌ DON'T ACTIVATE | Explicit disable |

### Phase 3: Import Hook Installation (if activated)
```
3. Install import hook into sys.meta_path
   │
   ├─→ hook = VllmInstrumentationHook()
   ├─→ sys.meta_path.insert(0, hook)
   │   (Now hook intercepts ALL future imports)
   │
   ├─→ Create session directory
   │   /tmp/vllm_profiling/session_20260128_123456/
   │
   ├─→ Initialize global profiler instances (lines 938-949)
   │   ├─ _cuda_profiler = CUDAGraphProfiler(session_dir)
   │   ├─ _kv_profiler = KVCacheProfiler(session_dir)
   │   ├─ _moe_profiler = MoEExpertProfiler(session_dir)
   │   ├─ _forward_pass_profiler = ForwardPassProfiler(session_dir)
   │   ├─ _cpu_profiler = CPUTimingProfiler(session_dir)
   │   ├─ _batch_util_profiler = BatchUtilizationProfiler(session_dir)
   │   ├─ _preemption_profiler = PreemptionProfiler(session_dir)
   │   └─ _encoder_decoder_profiler = EncoderDecoderProfiler(session_dir)
   │
   ├─→ Register exit handler (line 1000)
   │   atexit.register(save_all_stats)
   │
   └─→ Print startup message to stderr
       [sitecustomize] vLLM Comprehensive Instrumentation Loaded
       Session ID: 20260128_123456
       Output directory: /tmp/vllm_profiling/session_20260128_123456
```

### Phase 4: vLLM Imports Begin
```
4. Python continues normal execution
   │
   └─→ vLLM code starts importing its modules
       │
       ├─→ import vllm.compilation.cuda_graph
       │   ├─ Python checks sys.meta_path finders
       │   ├─ VllmInstrumentationHook.find_module() called
       │   │   └─→ Returns self (we handle this import)
       │   ├─ VllmInstrumentationHook.load_module() called
       │   │   ├─ Loads module normally via importlib
       │   │   ├─ Calls patch_cuda_graph_wrapper()
       │   │   │   └─→ Replaces CUDAGraphWrapper.__call__
       │   │   ├─ Updates _patch_status['CUDAGraphWrapper'] = True
       │   │   ├─ Prints: "[Instrumentation] ✅ Successfully patched CUDAGraphWrapper"
       │   │   └─→ Returns patched module
       │   └─→ Module is now available with instrumentation
       │
       ├─→ import vllm.v1.core.kv_cache_manager
       │   └─→ Similar flow, calls patch_kv_cache_manager()
       │
       ├─→ import vllm.v1.worker.gpu_model_runner
       │   └─→ Similar flow, calls patch_gpu_model_runner()
       │
       ├─→ import vllm.v1.core.sched.scheduler (or vllm.v1.core.scheduler)
       │   └─→ Similar flow, calls patch_scheduler()
       │
       ├─→ import vllm.model_executor.layers.fused_moe.layer
       │   └─→ Similar flow, calls patch_fused_moe()
       │
       └─→ import vllm.v1.core.block_pool
           └─→ Similar flow, calls patch_block_pool()
```

**Import Hook Code:** (lines 1450-1478)

```python
class VllmInstrumentationHook(importlib.abc.MetaPathFinder):
    def find_module(self, fullname, path=None):
        # Target modules to intercept
        target_modules = {
            'vllm.compilation.cuda_graph',
            'vllm.v1.core.kv_cache_manager',
            'vllm.v1.core.sched.scheduler',
            'vllm.v1.core.scheduler',
            'vllm.v1.worker.gpu_model_runner',
            'vllm.model_executor.layers.fused_moe.layer',
            'vllm.v1.core.block_pool',
        }

        if fullname in target_modules:
            return self  # We handle this import
        return None  # Let Python handle normally

    def load_module(self, fullname):
        # Load module normally
        module = importlib.import_module(fullname)

        # Apply appropriate patch
        if fullname == 'vllm.compilation.cuda_graph':
            patch_cuda_graph_wrapper()
        elif fullname == 'vllm.v1.core.kv_cache_manager':
            patch_kv_cache_manager()
        elif fullname == 'vllm.v1.core.sched.scheduler':
            patch_scheduler()
        elif fullname == 'vllm.v1.core.scheduler':
            patch_scheduler()
        elif fullname == 'vllm.v1.worker.gpu_model_runner':
            patch_gpu_model_runner()
        elif fullname == 'vllm.model_executor.layers.fused_moe.layer':
            patch_fused_moe()
        elif fullname == 'vllm.v1.core.block_pool':
            patch_block_pool()

        return module
```

### Phase 5: Runtime Execution
```
5. vLLM server runs
   │
   ├─→ Receives inference requests
   │
   ├─→ Executes model forward passes
   │   ├─ Calls CUDAGraphWrapper.__call__()
   │   │   └─→ Instrumented version executes
   │   │       ├─ Records capture/replay to _cuda_profiler
   │   │       └─ Calls original method
   │   │
   │   ├─ Calls Scheduler.schedule()
   │   │   └─→ Instrumented version executes
   │   │       ├─ Records batch utilization to _batch_util_profiler
   │   │       ├─ Records preemptions to _preemption_profiler
   │   │       └─ Calls original method
   │   │
   │   └─ Similar for all other patched methods
   │
   └─→ Metrics accumulate in global profiler instances
```

### Phase 6: Shutdown
```
6. Process exits (Ctrl+C or natural termination)
   │
   ├─→ atexit handler triggered
   │
   ├─→ save_all_stats() called
   │   ├─ _cuda_profiler.save()
   │   ├─ _kv_profiler.save()
   │   ├─ _moe_profiler.save()
   │   ├─ _forward_pass_profiler.save()
   │   ├─ _cpu_profiler.save()
   │   ├─ _batch_util_profiler.save()
   │   ├─ _preemption_profiler.save()
   │   └─ _encoder_decoder_profiler.save()
   │
   └─→ CSV files written to /tmp/vllm_profiling/session_XXX/
       ├─ cuda_graph_captures.csv
       ├─ cuda_graph_usage.csv
       ├─ kv_cache_usage.csv
       ├─ kv_cache_evictions.csv
       ├─ moe_expert_selection.csv
       ├─ moe_expert_timing.csv
       ├─ forward_pass_timing.csv
       ├─ cpu_operations_timing.csv
       ├─ batch_utilization.csv
       ├─ preemption_events.csv
       ├─ encoder_decoder_timing.csv
       └─ metadata.json
```

---

## Patch Activation Mechanism

### How Each Patch Works

All patches use the **method wrapping** pattern:

```python
# Pattern used by all 6 patch functions
def patch_<component>():
    # 1. Import the target class
    from vllm.xxx.yyy import TargetClass

    # 2. Store original method
    original_method = TargetClass.method_name

    # 3. Define instrumented wrapper
    def instrumented_method(self, *args, **kwargs):
        # Collect metrics BEFORE
        start_time = time.perf_counter()

        # Call original method
        result = original_method(self, *args, **kwargs)

        # Collect metrics AFTER
        end_time = time.perf_counter()
        _profiler.record_something(end_time - start_time, ...)

        return result

    # 4. Replace method on class
    TargetClass.method_name = instrumented_method

    # 5. Update status
    _patch_status['TargetClass'] = True

    # 6. Print confirmation
    print("[Instrumentation] ✅ Successfully patched TargetClass", file=sys.stderr)
```

### Example: CUDA Graph Patch (lines 1043-1077)

```python
def patch_cuda_graph_wrapper():
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    original_call = CUDAGraphWrapper.__call__

    def instrumented_call(self, *args, **kwargs):
        batch_descriptor = get_forward_context().batch_descriptor

        # Determine if this is a CAPTURE or REPLAY
        if batch_descriptor not in self.concrete_cudagraph_entries:
            # This is a CAPTURE (first time seeing this batch shape)
            mode = "prefill" if batch_descriptor.is_prefill else "decode"
            start = time.perf_counter()
            result = original_call(self, *args, **kwargs)
            duration = time.perf_counter() - start
            _cuda_profiler.record_capture(batch_descriptor, mode, duration)
        else:
            # This is a REPLAY (reusing existing graph)
            mode = "prefill" if batch_descriptor.is_prefill else "decode"
            start = time.perf_counter()
            result = original_call(self, *args, **kwargs)
            duration = time.perf_counter() - start
            _cuda_profiler.record_replay(batch_descriptor, mode, duration)

        return result

    CUDAGraphWrapper.__call__ = instrumented_call
    _patch_status['CUDAGraphWrapper'] = True
    print("[Instrumentation] ✅ Successfully patched CUDAGraphWrapper", file=sys.stderr)
```

**Key Points:**
1. Patch wraps `CUDAGraphWrapper.__call__()` (invoked on every forward pass)
2. Determines capture vs replay by checking `self.concrete_cudagraph_entries`
3. Records timing and batch info to global `_cuda_profiler`
4. Returns original result (transparent to vLLM)

---

## Why Patches Might Not Activate

### Common Causes

#### 1. **Activation Guard Blocked ProfileMate**

**Symptom:** No startup message appears

**Diagnosis:**
```bash
# Run your vLLM command and check stderr
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf 2>&1 | grep sitecustomize

# If you see NOTHING, activation was blocked
```

**Possible Causes:**
- Command line doesn't contain vLLM indicators
- `VLLM_ENABLE_PROFILING=0` is set
- Running a Python script that imports vLLM indirectly

**Solution:**
```bash
# Force enable
export VLLM_ENABLE_PROFILING=1
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf
```

#### 2. **vLLM Didn't Import Target Modules**

**Symptom:** Startup message appears, but specific patch messages don't

**Example:**
```
[sitecustomize] vLLM Comprehensive Instrumentation Loaded  ← Shows activation worked
[Instrumentation] ✅ Successfully patched Scheduler          ← Scheduler loaded
                                                              ← Missing CUDA graph message!
```

**Diagnosis:**
```python
# Add debug logging to sitecustomize.py line 1451
def find_module(self, fullname, path=None):
    print(f"[DEBUG] Import intercepted: {fullname}", file=sys.stderr)
    if fullname in target_modules:
        return self
    return None
```

**Possible Causes:**
- vLLM version doesn't use that component
  - Example: No MoE patch if not using Mixtral/DeepSeekMoE
  - Example: No CUDA graph patch if `--enforce-eager` is used
- Module path changed in vLLM version
  - Old vLLM: `vllm.core.scheduler`
  - New vLLM: `vllm.v1.core.sched.scheduler`
- Component is optional and not loaded

**Solution:**
```bash
# Check which modules vLLM actually imports
strace -e trace=openat python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf 2>&1 | grep vllm

# Or use Python's import logging
python -X importtime -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf 2>&1 | grep vllm
```

#### 3. **PYTHONPATH Not Set**

**Symptom:** No output at all from ProfileMate

**Diagnosis:**
```bash
echo $PYTHONPATH
# Should include: /home/nmiriyal/Documents/MLPERF-6.0/profilemate
```

**Solution:**
```bash
export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"
```

#### 4. **Import Hook Registered Too Late**

**Symptom:** Some patches work, others don't

**Cause:** vLLM imported modules BEFORE sitecustomize.py ran

**Example:**
```python
# BAD: This imports vLLM before ProfileMate can hook it
import vllm
import sitecustomize  # Too late!

# GOOD: PYTHONPATH ensures sitecustomize loads first
# (Python automatically imports sitecustomize before user code)
```

**Solution:** Always use PYTHONPATH, never manually import sitecustomize

#### 5. **Patch Function Has Errors**

**Symptom:** Startup message appears, but patch fails silently

**Diagnosis:**
```python
# Wrap patch call in try/except (line 1467)
if fullname == 'vllm.compilation.cuda_graph':
    try:
        patch_cuda_graph_wrapper()
    except Exception as e:
        print(f"[ERROR] Failed to patch CUDA graph: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
```

**Common Errors:**
- Import error (module structure changed)
- Attribute error (method renamed)
- Type error (function signature changed)

---

## Why We Can't Activate All Patches Upfront

### The Fundamental Problem

**Question:** Why not just call all 6 patch functions immediately in sitecustomize.py?

```python
# Why doesn't this work?
if should_activate_profiling():
    patch_cuda_graph_wrapper()  # ❌ FAILS
    patch_kv_cache_manager()    # ❌ FAILS
    patch_scheduler()           # ❌ FAILS
    # ... etc
```

**Answer:** **The modules don't exist yet!**

### Timeline Visualization

```
Time: 0ms - Python starts
  ↓
Time: 5ms - sitecustomize.py loads
  ├─ ProfilingConfig defined ✅
  ├─ CUDAGraphProfiler defined ✅
  ├─ patch functions defined ✅
  │
  └─ Attempt to patch:
      from vllm.compilation.cuda_graph import CUDAGraphWrapper
      ❌ ImportError: No module named 'vllm'
      (vLLM hasn't been imported yet!)
  ↓
Time: 100ms - Install import hook ✅
  ↓
Time: 500ms - vLLM starts importing
  ├─ import vllm.compilation.cuda_graph
  │   └─→ Now module exists, patch can work! ✅
  ↓
Time: 1000ms - vLLM fully loaded
  └─ All patches applied ✅
```

### Why Import Hooks Are Necessary

Import hooks solve the "chicken and egg" problem:

| Approach | Problem | Solution |
|----------|---------|----------|
| **Patch immediately** | vLLM modules don't exist yet | ❌ ImportError |
| **Modify vLLM source** | Requires code changes, breaks updates | ❌ Invasive |
| **Use import hooks** | Wait for module to load, then patch | ✅ Works! |

### Technical Explanation

Python's import system has an extension point: `sys.meta_path`

```python
# Python's import algorithm (simplified)
def import_module(name):
    # 1. Check if already imported
    if name in sys.modules:
        return sys.modules[name]

    # 2. Ask each meta_path finder if they handle this module
    for finder in sys.meta_path:
        loader = finder.find_module(name)
        if loader is not None:
            # Finder wants to handle this import
            module = loader.load_module(name)
            sys.modules[name] = module
            return module

    # 3. Fall back to standard import mechanism
    return standard_import(name)
```

ProfileMate inserts `VllmInstrumentationHook` into `sys.meta_path`:

```python
sys.meta_path.insert(0, VllmInstrumentationHook())
# Now our hook gets first chance at every import
```

When vLLM runs `import vllm.compilation.cuda_graph`:

```
1. Python calls VllmInstrumentationHook.find_module('vllm.compilation.cuda_graph')
   └─→ Returns self (we handle this)

2. Python calls VllmInstrumentationHook.load_module('vllm.compilation.cuda_graph')
   ├─ Calls standard import to load module
   ├─ Module now exists in memory ✅
   ├─ Calls patch_cuda_graph_wrapper()
   │   ├─ from vllm.compilation.cuda_graph import CUDAGraphWrapper ✅
   │   └─ Wraps methods ✅
   └─ Returns patched module

3. vLLM gets the patched module ✅
```

### Alternative Approaches (and why we don't use them)

#### Option A: Patch After vLLM Imports

```python
# user_script.py
import vllm  # Import vLLM first
import profilemate  # Then activate profiling

# Problem: vLLM already imported its submodules!
# By the time profilemate loads, vLLM's internal imports are done.
# Patching now won't affect already-imported code.
```

#### Option B: Modify vLLM Source Code

```python
# vllm/compilation/cuda_graph.py
from profilemate import instrument_cuda_graph

class CUDAGraphWrapper:
    @instrument_cuda_graph  # Decorator from ProfileMate
    def __call__(self, *args, **kwargs):
        ...

# Problems:
# ❌ Requires changing vLLM code
# ❌ Breaks on vLLM updates
# ❌ Can't be distributed separately
# ❌ Tight coupling
```

#### Option C: Import Hooks (Current Approach) ✅

```python
# sitecustomize.py (auto-loaded by Python)
sys.meta_path.insert(0, VllmInstrumentationHook())

# Advantages:
# ✅ Zero vLLM code changes
# ✅ Works across vLLM versions
# ✅ Distributed separately
# ✅ Can be enabled/disabled via env var
# ✅ Transparent to vLLM
```

---

## Debugging Non-Activating Patches

### Step-by-Step Debugging Guide

#### Step 1: Verify PYTHONPATH

```bash
echo $PYTHONPATH
# Expected: /home/nmiriyal/Documents/MLPERF-6.0/profilemate:<other paths>

# If missing:
export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"
```

#### Step 2: Force Enable Profiling

```bash
export VLLM_ENABLE_PROFILING=1
```

#### Step 3: Run vLLM and Capture Output

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000 \
    2>&1 | tee vllm_output.log
```

#### Step 4: Check Startup Messages

```bash
grep sitecustomize vllm_output.log
# Expected output:
# [sitecustomize] vLLM Comprehensive Instrumentation Loaded
# Session ID: 20260128_123456
# Output directory: /tmp/vllm_profiling/session_20260128_123456
```

**If missing:** Activation was blocked, see Step 2

#### Step 5: Check Patch Messages

```bash
grep "Successfully patched" vllm_output.log
# Expected output:
# [Instrumentation] ✅ Successfully patched CUDAGraphWrapper
# [Instrumentation] ✅ Successfully patched KVCacheManager
# [Instrumentation] ✅ Successfully patched Scheduler
# [Instrumentation] ✅ Successfully patched GPUModelRunner
# (MoE and BlockPool may be absent if not used)
```

**Analysis:**

| Patches Activated | Diagnosis |
|------------------|-----------|
| None | Import hook failed to install |
| Some but not all | vLLM didn't import missing components |
| All | ✅ Working correctly |

#### Step 6: Add Debug Logging

Edit `sitecustomize.py` and add logging:

```python
# Line 1451 - Log all imports
def find_module(self, fullname, path=None):
    print(f"[DEBUG] Import attempt: {fullname}", file=sys.stderr)
    target_modules = {
        'vllm.compilation.cuda_graph',
        'vllm.v1.core.kv_cache_manager',
        'vllm.v1.core.sched.scheduler',
        'vllm.v1.core.scheduler',
        'vllm.v1.worker.gpu_model_runner',
        'vllm.model_executor.layers.fused_moe.layer',
        'vllm.v1.core.block_pool',
    }

    if fullname in target_modules:
        print(f"[DEBUG] ✅ Will patch: {fullname}", file=sys.stderr)
        return self
    return None

# Line 1467 - Log patch attempts
def load_module(self, fullname):
    print(f"[DEBUG] Loading and patching: {fullname}", file=sys.stderr)
    module = importlib.import_module(fullname)

    if fullname == 'vllm.compilation.cuda_graph':
        print(f"[DEBUG] Calling patch_cuda_graph_wrapper()", file=sys.stderr)
        try:
            patch_cuda_graph_wrapper()
        except Exception as e:
            print(f"[ERROR] Patch failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
```

Rerun and check output:

```bash
grep DEBUG vllm_output.log
```

#### Step 7: Verify CSV Output

```bash
ls -la /tmp/vllm_profiling/session_*/
# Expected files:
# cuda_graph_captures.csv
# cuda_graph_usage.csv
# kv_cache_usage.csv
# forward_pass_timing.csv
# etc.

# Check if files have data
wc -l /tmp/vllm_profiling/session_*/cuda_graph_usage.csv
# Expected: >1 line (header + data rows)
```

**If files are empty:**
- Patch activated but vLLM didn't use that component
- Run some inference requests to generate data

#### Step 8: Send Test Request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

Then check CSV files again - they should have new rows.

---

## Architecture Diagrams

### Component Relationship

```
┌─────────────────────────────────────────────────────────────┐
│                     Python Interpreter                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    sys.meta_path                       │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │      VllmInstrumentationHook (Position 0)       │  │  │
│  │  │  - find_module()                                │  │  │
│  │  │  - load_module()                                │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │  [ Standard import machinery ]                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               Global Profiler Instances                │  │
│  │  ┌──────────────────┐  ┌──────────────────┐          │  │
│  │  │ _cuda_profiler   │  │ _kv_profiler     │          │  │
│  │  │ (CSV writer)     │  │ (CSV writer)     │          │  │
│  │  └──────────────────┘  └──────────────────┘          │  │
│  │  ┌──────────────────┐  ┌──────────────────┐          │  │
│  │  │ _moe_profiler    │  │_forward_profiler │          │  │
│  │  └──────────────────┘  └──────────────────┘          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ Records metrics
                            │
┌─────────────────────────────────────────────────────────────┐
│                         vLLM Process                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │          CUDAGraphWrapper (PATCHED)                    │  │
│  │  def __call__(self, *args, **kwargs):                 │  │
│  │      _cuda_profiler.record_capture(...)  ←──────────┐ │  │
│  │      return original_call(self, *args, **kwargs)    │ │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │          Scheduler (PATCHED)                           │  │
│  │  def schedule(self, ...):                             │  │
│  │      _batch_util_profiler.record(...)  ←────────────┼─┘  │
│  │      return original_schedule(...)                   │    │
│  └───────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Patch Activation Timeline

```
Time →
─────────────────────────────────────────────────────────────────
  0ms  Python process starts
   ↓
  5ms  sitecustomize.py auto-imported
   │    ├─ Define classes
   │    ├─ Define patch functions
   │    └─ Check should_activate_profiling()
   ↓
 10ms  Install import hook → sys.meta_path.insert(0, hook)
   │    ├─ Initialize profilers
   │    └─ Register atexit handler
   ↓
100ms  vLLM starts importing modules
   │
   ├─→ import vllm.compilation.cuda_graph
   │    ├─ Hook intercepts
   │    ├─ Load module
   │    ├─ Apply patch_cuda_graph_wrapper()
   │    └─ Print "✅ Successfully patched CUDAGraphWrapper"
   │
   ├─→ import vllm.v1.core.sched.scheduler
   │    ├─ Hook intercepts
   │    ├─ Load module
   │    ├─ Apply patch_scheduler()
   │    └─ Print "✅ Successfully patched Scheduler"
   │
   └─→ (Similar for other modules)
   ↓
1000ms vLLM fully loaded, server starts
   ↓
   │   [Runtime - requests processed, metrics recorded]
   ↓
   │   Ctrl+C
   ↓
   │   atexit handler runs
   │    ├─ save_all_stats()
   │    └─ Write CSV files
   ↓
  END
```

---

## Summary

### Key Takeaways

1. **Patches activate lazily** via import hooks, not upfront
2. **This is necessary** because target modules don't exist until vLLM imports them
3. **Activation can be controlled** via `VLLM_ENABLE_PROFILING` environment variable
4. **Activation guard prevents profiling** non-vLLM Python processes
5. **Some patches may not activate** if vLLM doesn't use that component
6. **Debug by checking stderr** for startup and patch messages
7. **CSV files prove patches work** - if they contain data, patches are working

### Activation Checklist

- [ ] PYTHONPATH includes profilemate directory
- [ ] See startup message: `[sitecustomize] vLLM Comprehensive Instrumentation Loaded`
- [ ] See patch messages: `[Instrumentation] ✅ Successfully patched <Component>`
- [ ] CSV files created in `/tmp/vllm_profiling/session_*/`
- [ ] CSV files contain data after running inference

### Next Steps

If patches still aren't activating after following this guide:

1. Check `profilemate/DEBUG_GUIDE.md` for advanced troubleshooting
2. Run with debug logging enabled (edit sitecustomize.py)
3. Share vLLM command, startup messages, and patch status for further help
