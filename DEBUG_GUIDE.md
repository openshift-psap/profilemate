# ProfileMate Debug Guide

Complete guide to debugging ProfileMate when you see zero stats or no data.

---

## Quick Debug Checklist

```bash
# 1. Enable debug mode
export VLLM_PROFILING_DEBUG=1
export VLLM_NO_USAGE_STATS=1  # Fix cpuinfo error

# 2. Set PYTHONPATH
export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"

# 3. Run vLLM with small model
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --port 8000 \
    2>&1 | tee vllm_debug.log

# 4. Look for these messages in the log:
grep -E "sitecustomize|Instrumentation|CUDA Graph" vllm_debug.log
```

---

## Part 1: Understanding VLLM_NO_USAGE_STATS

### What is it?

vLLM has a telemetry system that collects anonymous usage stats:
- Model name
- GPU type
- vLLM version
- etc.

It sends this data to help vLLM developers understand usage patterns.

### The cpuinfo Bug

On some systems, the `py-cpuinfo` library crashes when parsing `lscpu` output:

```python
File "vllm/usage/usage_lib.py", line 213, in _report_usage_once
    info = cpuinfo.get_cpu_info()
           ^^^^^^^^^^^^^^^^^^^^^^
JSONDecodeError: Extra data: line 3 column 1 (char 2029)
```

This happens BEFORE ProfileMate even loads. It's a vLLM bug, not ProfileMate.

### Why Set It?

```bash
export VLLM_NO_USAGE_STATS=1
```

This tells vLLM: "Don't collect telemetry data"

**Result:**
- ✅ No cpuinfo crash
- ✅ Slightly faster startup (no telemetry thread)
- ✅ No data sent to vLLM servers
- ✅ Completely safe - just disables optional feature

### Alternative Fixes

If you don't want to disable telemetry:

```bash
# Option 1: Upgrade py-cpuinfo
pip install --upgrade py-cpuinfo

# Option 2: Downgrade to working version
pip install py-cpuinfo==8.0.0

# Option 3: Uninstall (vLLM will skip telemetry)
pip uninstall py-cpuinfo -y
```

---

## Part 2: Debug Mode

### Enable Debug Mode

```bash
export VLLM_PROFILING_DEBUG=1
```

### What You'll See

**At startup:**
```
[sitecustomize] vLLM Comprehensive Instrumentation Loaded
  Session ID: 20260127_123456
  Output directory: /tmp/vllm_profiling/session_20260127_123456
  ...
  Debug mode: True

[sitecustomize] 🔍 DEBUG MODE ENABLED
  You will see detailed logs when events are captured.
  Look for messages like:
    - ✅ CAPTURED: (new CUDA graph)
    - 🔄 REPLAY: (CUDA graph replay)
    - ✅ Successfully patched: (module instrumentation)
```

**When vLLM loads modules:**
```
[Instrumentation] ✅ Successfully patched CUDAGraphWrapper
   - Intercepting: CUDAGraphWrapper.__call__
   - Profiler enabled: True

[Instrumentation] ✅ Successfully patched GPUModelRunner
[Instrumentation] ✅ Successfully patched Scheduler
[Instrumentation] ✅ Successfully patched KVCacheManager
```

**When CUDA graphs are captured:**
```
[CUDA Graph Patch] Capturing new graph: BatchDescriptor(num_tokens=256, ...)
[CUDA Graph] ✅ CAPTURED: FULL:BatchDescriptor(num_tokens=256, ...)
  Total unique graphs: 1
```

**When CUDA graphs are replayed:**
```
[CUDA Graph] 🔄 REPLAY #1: FULL:BatchDescriptor(num_tokens=256, ...) (12.45ms)
[CUDA Graph] 🔄 REPLAY #2: FULL:BatchDescriptor(num_tokens=256, ...) (12.38ms)
[CUDA Graph] 🔄 REPLAY #3: FULL:BatchDescriptor(num_tokens=256, ...) (12.42ms)
```

**After 2 seconds:**
```
============================================================
ProfileMate Patch Diagnostics
============================================================
  ✅ CUDAGraphWrapper: Applied
  ✅ GPUModelRunner: Applied
  ✅ Scheduler: Applied
  ✅ KVCacheManager: Applied
  ⏳ FusedMoE: Waiting (will patch when module loads)
  ⏳ BlockPool: Waiting (will patch when module loads)

Note: Patches are applied lazily when vLLM modules are imported.
Check logs for '✅ Successfully patched' messages as vLLM initializes.
============================================================
```

### Debug Logging Frequency

```bash
# Log every CUDA graph replay (can be verbose)
export VLLM_PROFILING_DEBUG_INTERVAL=1

# Log every 10th replay (less verbose)
export VLLM_PROFILING_DEBUG_INTERVAL=10

# Log every 100th replay (minimal)
export VLLM_PROFILING_DEBUG_INTERVAL=100
```

---

## Part 3: Checking if Patches Were Applied

### Method 1: Look for "Successfully patched" Messages

```bash
# Run vLLM and capture logs
python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m 2>&1 | tee vllm.log

# Check for successful patches
grep "Successfully patched" vllm.log
```

**Expected output:**
```
[Instrumentation] Successfully patched CUDAGraphWrapper
[Instrumentation] Successfully patched GPUModelRunner
[Instrumentation] Successfully patched Scheduler
[Instrumentation] Successfully patched KVCacheManager
```

**If you see nothing:**
- ProfileMate didn't load or patches failed
- Check PYTHONPATH
- Check vLLM version compatibility

### Method 2: Use Patch Diagnostics (Debug Mode)

```bash
export VLLM_PROFILING_DEBUG=1
python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m 2>&1 | grep -A 10 "Patch Diagnostics"
```

**Good output:**
```
============================================================
ProfileMate Patch Diagnostics
============================================================
  ✅ CUDAGraphWrapper: Applied
  ✅ GPUModelRunner: Applied
  ✅ Scheduler: Applied
```

**Bad output:**
```
============================================================
ProfileMate Patch Diagnostics
============================================================
  ⏳ CUDAGraphWrapper: Waiting
  ⏳ GPUModelRunner: Waiting
  ⏳ Scheduler: Waiting
```
This means modules haven't loaded yet or import hook isn't working.

### Method 3: Check for Event Capture Logs

```bash
# Enable debug mode and send requests
export VLLM_PROFILING_DEBUG=1
python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m --port 8000 2>&1 | grep "CUDA Graph"
```

After sending requests, you should see:
```
[CUDA Graph] ✅ CAPTURED: FULL:BatchDescriptor(...)
[CUDA Graph] 🔄 REPLAY #1: FULL:BatchDescriptor(...) (12.45ms)
[CUDA Graph] 🔄 REPLAY #2: FULL:BatchDescriptor(...) (12.38ms)
```

**If you see nothing:**
- CUDA graphs aren't being used (try different workload)
- Patch wasn't applied
- CUDAGraphWrapper isn't being called

---

## Part 4: Step-by-Step Debugging

### Step 1: Verify ProfileMate Loads

```bash
export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"
export VLLM_PROFILING_DEBUG=1
export VLLM_NO_USAGE_STATS=1

python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m --port 8000 2>&1 | head -50
```

**Look for:**
```
[sitecustomize] vLLM Comprehensive Instrumentation Loaded
```

**If NOT found:**
- Check: `echo $PYTHONPATH`
- Verify: `ls $PYTHONPATH/sitecustomize.py`
- Test: `python3 -c "import sitecustomize; print('OK')"`

### Step 2: Verify Patches Apply

**Continue watching vLLM startup logs:**

**Look for:**
```
[Instrumentation] ✅ Successfully patched CUDAGraphWrapper
[Instrumentation] ✅ Successfully patched GPUModelRunner
```

**If NOT found:**
- vLLM modules may have different paths (check vLLM version)
- Import hook not working
- Run compatibility check: `python3 check_vllm_compatibility.py`

### Step 3: Send Test Requests

```bash
# In another terminal
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "Hello, how are you?",
        "max_tokens": 50
    }'
```

**Watch vLLM logs for:**
```
[CUDA Graph] ✅ CAPTURED: ...
[CUDA Graph] 🔄 REPLAY #1: ...
```

### Step 4: Check Output Files

```bash
# List session directory
ls -lh /tmp/vllm_profiling/session_*/

# Check file sizes (should NOT be empty)
wc -l /tmp/vllm_profiling/session_*/cuda_graph_captures.csv
wc -l /tmp/vllm_profiling/session_*/cuda_graph_usage.csv

# Check content
head -20 /tmp/vllm_profiling/session_*/cuda_graph_captures.csv
```

**Good output:**
```
2  # Header + 1 data line minimum
```

**Bad output:**
```
1  # Only header, no data
```

### Step 5: Analyze Why No Data

**Possible reasons:**

1. **No CUDA graphs were captured**
   ```bash
   # Check if vLLM is using CUDA graphs
   grep -i "cudagraph" vllm.log
   ```
   Some workloads don't trigger CUDA graph capture.

2. **Patches weren't applied**
   ```bash
   grep "Successfully patched" vllm.log
   ```
   If empty, patches failed.

3. **ProfileMate didn't load**
   ```bash
   grep "sitecustomize" vllm.log
   ```
   If empty, PYTHONPATH issue.

4. **Data not saved yet (still running)**
   - ProfileMate saves on shutdown via `atexit`
   - Stop vLLM server to trigger save
   - Or wait for periodic saves

---

## Part 5: Focused CUDA Graph Debugging

### Why Start with CUDA Graphs?

1. **Easy to verify** - Clear capture/replay events
2. **Happens early** - During warmup
3. **Visible in logs** - With debug mode enabled
4. **Independent** - Doesn't depend on other profilers

### Enable CUDA Graph-Only Debugging

Edit `sitecustomize.py`:

```python
class ProfilingConfig:
    ENABLE_CUDA_GRAPH_TRACKING = True   # ✅ Enable this
    ENABLE_KV_CACHE_TRACKING = False    # ❌ Disable others
    ENABLE_MOE_EXPERT_TRACKING = False
    ENABLE_FORWARD_PASS_TIMING = False
    ENABLE_CPU_TIMING = False
    ENABLE_BATCH_UTILIZATION_TRACKING = False
    ENABLE_PREEMPTION_TRACKING = False
    ENABLE_ENCODER_DECODER_TIMING = False

    DEBUG = True  # Enable debug mode
```

### Expected Debug Output

```
[sitecustomize] vLLM Comprehensive Instrumentation Loaded
  Session ID: 20260127_123456
  CUDA graph tracking: True
  KV cache tracking: False
  MoE expert tracking: False
  ...
  Debug mode: True

[Instrumentation] ✅ Successfully patched CUDAGraphWrapper
   - Intercepting: CUDAGraphWrapper.__call__
   - Profiler enabled: True

# During warmup / first requests:
[CUDA Graph Patch] Capturing new graph: BatchDescriptor(num_tokens=1, ...)
[CUDA Graph] ✅ CAPTURED: FULL:BatchDescriptor(num_tokens=1, ...)
  Total unique graphs: 1

[CUDA Graph Patch] Capturing new graph: BatchDescriptor(num_tokens=2, ...)
[CUDA Graph] ✅ CAPTURED: FULL:BatchDescriptor(num_tokens=2, ...)
  Total unique graphs: 2

# During subsequent requests:
[CUDA Graph] 🔄 REPLAY #1: FULL:BatchDescriptor(num_tokens=1, ...) (8.45ms)
[CUDA Graph] 🔄 REPLAY #2: FULL:BatchDescriptor(num_tokens=2, ...) (9.23ms)
[CUDA Graph] 🔄 REPLAY #3: FULL:BatchDescriptor(num_tokens=1, ...) (8.38ms)
```

### If You Don't See CAPTURED Messages

**Check 1: Is CUDAGraphWrapper being called?**

Add this to `patch_cuda_graph_wrapper()` (for debugging):

```python
def instrumented_call(self, *args, **kwargs):
    # ADD THIS LINE:
    print(f"[DEBUG] CUDAGraphWrapper.__call__ invoked!", file=sys.stderr)

    # Rest of the function...
```

If you see "CUDAGraphWrapper.__call__ invoked!" but no CAPTURED messages:
- `batch_descriptor` might be None
- `_cuda_profiler` might be None
- Logic isn't reaching capture code

**Check 2: Is vLLM using CUDA graphs?**

```bash
# Look for vLLM CUDA graph messages
grep -i "cuda.*graph" vllm.log

# Check vLLM config
python -c "
import vllm
from vllm.config import VllmConfig
# Check if CUDA graphs are enabled in your run
"
```

**Check 3: Module path correct?**

```bash
# Check if vLLM has CUDAGraphWrapper
python3 -c "
from vllm.compilation.cuda_graph import CUDAGraphWrapper
print('✅ CUDAGraphWrapper found')
print(f'Location: {CUDAGraphWrapper.__module__}')
"
```

---

## Part 6: Common Issues and Solutions

### Issue 1: Zero Stats in All Files

**Symptoms:**
- CSV files have only headers
- No capture/replay messages in logs

**Diagnosis:**
```bash
# Check if patches applied
grep "Successfully patched" vllm.log

# Check if ProfileMate loaded
grep "sitecustomize" vllm.log

# Check if requests were sent
curl http://localhost:8000/v1/completions ...
```

**Solutions:**
1. Enable debug mode: `export VLLM_PROFILING_DEBUG=1`
2. Check patches applied
3. Send more requests
4. Wait longer (some stats need warmup)

### Issue 2: Patches Not Applied

**Symptoms:**
- No "Successfully patched" messages
- Patch diagnostics shows "Waiting" or "NotFound"

**Diagnosis:**
```bash
# Check vLLM version
python3 -c "import vllm; print(vllm.__version__)"

# Check module paths
python3 check_vllm_compatibility.py
```

**Solutions:**
1. Update ProfileMate for your vLLM version
2. Check module paths in `sitecustomize.py`
3. Verify vLLM installation

### Issue 3: ProfileMate Doesn't Load

**Symptoms:**
- No "sitecustomize" messages at all

**Diagnosis:**
```bash
echo $PYTHONPATH
python3 -c "import sitecustomize; print('OK')"
```

**Solutions:**
```bash
# Fix PYTHONPATH
export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"

# Or force enable
export VLLM_ENABLE_PROFILING=1
```

### Issue 4: CUDA Graphs Not Captured

**Symptoms:**
- Patches applied successfully
- No CAPTURED messages
- No replay messages

**Possible causes:**
1. vLLM not using CUDA graphs (small model, short sequences)
2. CUDA graphs disabled in config
3. Eager mode enabled

**Solutions:**
```bash
# Check if CUDA graphs are used
# Look for vLLM's own CUDA graph messages

# Try with different workload
curl http://localhost:8000/v1/completions \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "'"$(python3 -c 'print("test " * 100)')"'",
        "max_tokens": 100
    }'
```

---

## Part 7: Complete Debug Session Example

```bash
# 1. Clean start
rm -rf /tmp/vllm_profiling/session_*
unset PYTHONPATH

# 2. Set environment
export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"
export VLLM_PROFILING_DEBUG=1
export VLLM_NO_USAGE_STATS=1

# 3. Start vLLM with logs
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --port 8000 \
    2>&1 | tee vllm_debug.log &

# 4. Wait for server to start
sleep 30

# 5. Check ProfileMate loaded
grep "sitecustomize" vllm_debug.log
# Expected: "[sitecustomize] vLLM Comprehensive Instrumentation Loaded"

# 6. Check patches applied
grep "Successfully patched" vllm_debug.log
# Expected: Multiple "✅ Successfully patched" messages

# 7. Send requests
for i in {1..10}; do
    curl -s http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "facebook/opt-125m",
            "prompt": "Hello world",
            "max_tokens": 10
        }' > /dev/null
    echo "Request $i sent"
    sleep 1
done

# 8. Check for capture/replay messages
grep "CUDA Graph" vllm_debug.log
# Expected: CAPTURED and REPLAY messages

# 9. Stop server
kill %1

# 10. Check output files
ls -lh /tmp/vllm_profiling/session_*/
cat /tmp/vllm_profiling/session_*/cuda_graph_captures.csv
cat /tmp/vllm_profiling/session_*/cuda_graph_usage.csv

# 11. Analyze
python3 << 'EOF'
import pandas as pd
import glob

sessions = glob.glob('/tmp/vllm_profiling/session_*/')
if sessions:
    latest = max(sessions)
    print(f"Session: {latest}")

    captures = pd.read_csv(f'{latest}/cuda_graph_captures.csv')
    print(f"\nUnique graphs captured: {len(captures)}")
    print(captures)

    usage = pd.read_csv(f'{latest}/cuda_graph_usage.csv')
    print(f"\nTotal replays: {usage['replay_count'].sum()}")
    print(usage)
else:
    print("No sessions found!")
EOF
```

---

## Summary

### Debug Mode Benefits

✅ See exactly when events are captured
✅ Verify patches are applied
✅ Diagnose why stats are zero
✅ Real-time feedback

### Key Debug Flags

```bash
export VLLM_PROFILING_DEBUG=1          # Enable debug logging
export VLLM_PROFILING_VERBOSE=1        # Enable verbose logging
export VLLM_PROFILING_DEBUG_INTERVAL=1 # Log every event
export VLLM_NO_USAGE_STATS=1           # Fix cpuinfo crash
```

### What to Look For

1. **Startup:** "vLLM Comprehensive Instrumentation Loaded"
2. **Patches:** "✅ Successfully patched"
3. **Captures:** "✅ CAPTURED"
4. **Replays:** "🔄 REPLAY"
5. **Diagnostics:** Patch status table

### Next Steps

1. Start with CUDA graph debugging (easiest to verify)
2. Enable debug mode to see events
3. Check patches applied successfully
4. Send test requests and watch logs
5. Verify output files have data
