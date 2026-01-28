# Troubleshooting: Patches Not Activating

## Quick Diagnostic

Run the diagnostic script first:

```bash
cd /home/nmiriyal/Documents/MLPERF-6.0/profilemate
python diagnose_patches.py
```

This will check your configuration and identify issues.

---

## Common Issue #1: Patches Not Activating At All

### Symptoms
- No startup message appears
- No CSV files created
- No patch confirmation messages

### Diagnosis
```bash
# Check if sitecustomize is being loaded
python -c "import sitecustomize; print('Loaded!')"

# If you see an error, sitecustomize is not in PYTHONPATH
```

### Solution
```bash
# Add ProfileMate to PYTHONPATH
export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"

# Verify
echo $PYTHONPATH | grep profilemate

# Make it permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## Common Issue #2: Activation Blocked by Guard

### Symptoms
- You know PYTHONPATH is set correctly
- Still no startup message
- Running a script that imports vLLM

### Diagnosis
The activation guard is preventing ProfileMate from activating because your command doesn't look like a vLLM command.

**Example of commands that get blocked:**
```bash
python my_custom_script.py  # Even if it imports vLLM internally
python -m pytest            # Testing vLLM code
jupyter notebook            # Using vLLM in notebooks
```

### Solution
Force enable profiling:

```bash
export VLLM_ENABLE_PROFILING=1
python my_custom_script.py
```

Or modify your script to include vLLM indicators:
```bash
python -m vllm.entrypoints.openai.api_server --model <model>  # Auto-detects ✅
```

---

## Common Issue #3: Some Patches Activate, Others Don't

### Symptoms
- Startup message appears ✅
- Some patches show success messages ✅
- Some patches never print confirmation ❌
- Some CSV files are empty

### Diagnosis
vLLM didn't import those modules because:
1. **Your vLLM version doesn't use that component**
2. **Feature is disabled via command-line flags**
3. **Component is optional and not triggered**

### Examples

#### CUDA Graph Patch Not Activating
**Reason:** You used `--enforce-eager` flag

```bash
# This disables CUDA graphs
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --enforce-eager

# Expected: No CUDA graph patch message (this is normal!)
```

**Solution:** Remove `--enforce-eager` if you want CUDA graph profiling

#### MoE Patch Not Activating
**Reason:** Your model is not a Mixture-of-Experts model

```bash
# This model doesn't use MoE
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf

# Expected: No MoE patch message (this is normal!)
```

**Solution:** Use a MoE model like `mistralai/Mixtral-8x7B-v0.1` if you want MoE profiling

#### Scheduler Patch Not Activating
**Reason:** vLLM version mismatch

```bash
# Check which scheduler path exists
python -c "import importlib.util; print('New path:', importlib.util.find_spec('vllm.v1.core.sched.scheduler') is not None)"
python -c "import importlib.util; print('Old path:', importlib.util.find_spec('vllm.v1.core.scheduler') is not None)"
```

**Solution:** Check `sitecustomize.py` lines 1467-1478 to ensure both paths are handled

---

## Common Issue #4: Patches Activate But No Data in CSV Files

### Symptoms
- Startup message appears ✅
- Patch messages appear ✅
- CSV files created ✅
- CSV files only have headers (no data rows) ❌

### Diagnosis
Patches are working, but vLLM hasn't executed the instrumented code yet.

### Solution
Send inference requests to trigger the instrumented code:

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --port 8000

# In another terminal, send a request
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'

# Now check CSV files again
cat /tmp/vllm_profiling/session_*/cuda_graph_usage.csv
```

---

## Common Issue #5: "Patch Activating for Any Python Process"

### Symptoms
- Running `python my_script.py` activates ProfileMate
- Don't want ProfileMate active for non-vLLM scripts
- Want to restrict to vLLM only

### Diagnosis
This issue was already fixed in the code! The activation guard (lines 1488-1528 in `sitecustomize.py`) prevents activation for non-vLLM processes.

### How It Works
ProfileMate only activates if:
1. `VLLM_ENABLE_PROFILING=1` is explicitly set, OR
2. Command line contains vLLM indicators (`vllm.entrypoints`, `--model`, etc.), OR
3. Command line contains `vllm` keyword

### Verify It's Working
```bash
# Test 1: Normal Python script (should NOT activate)
unset VLLM_ENABLE_PROFILING
python -c "print('Hello')"
# Expected: No ProfileMate messages ✅

# Test 2: vLLM command (should activate)
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf
# Expected: [sitecustomize] vLLM Comprehensive Instrumentation Loaded ✅

# Test 3: Force enable for any script
export VLLM_ENABLE_PROFILING=1
python -c "print('Hello')"
# Expected: [sitecustomize] vLLM Comprehensive Instrumentation Loaded ✅
```

### If It's Still Activating Unexpectedly
Check if you have `VLLM_ENABLE_PROFILING=1` set in your shell:

```bash
env | grep VLLM_ENABLE_PROFILING

# If you see VLLM_ENABLE_PROFILING=1, unset it
unset VLLM_ENABLE_PROFILING
```

---

## Why We Can't Activate All Patches Upfront

### The Question
"Can't we just activate all patches when sitecustomize loads, instead of waiting for imports?"

### The Answer
**No, because the target modules don't exist yet!**

### Example

```python
# sitecustomize.py loads at time 0ms
# At this point, vLLM hasn't been imported yet

# If we try to patch immediately:
from vllm.compilation.cuda_graph import CUDAGraphWrapper
# ❌ ImportError: No module named 'vllm'

# The module literally doesn't exist in memory yet!
```

### The Timeline

```
0ms:   Python starts
  ↓
5ms:   sitecustomize.py loads
       - ProfileMate code is loaded ✅
       - But vLLM code doesn't exist yet ❌
  ↓
10ms:  Install import hook
       - Hook waits for vLLM imports
  ↓
100ms: vLLM code runs: import vllm.compilation.cuda_graph
       - NOW the module gets loaded
       - Hook intercepts it
       - NOW we can patch it ✅
  ↓
1000ms: All vLLM modules loaded and patched
```

### Why Import Hooks Are The Solution

Import hooks solve this "chicken and egg" problem:
- We need vLLM modules to exist before we can patch them
- But we need to patch them as soon as they're imported
- Import hooks intercept the import and patch immediately after loading

This is the standard Python pattern for transparent instrumentation!

### Alternative That Doesn't Work

```python
# BAD: Try to import and patch immediately
import vllm.compilation.cuda_graph  # ❌ Too early, module doesn't exist

# GOOD: Wait for vLLM to import it, then patch via hook
sys.meta_path.insert(0, ImportHook())  # ✅ Patches when module loads
```

---

## Advanced Debugging

### Enable Debug Logging

Edit `sitecustomize.py` and add debug prints:

```python
# Line 1451 - Log all import attempts
def find_module(self, fullname, path=None):
    if 'vllm' in fullname:
        print(f"[DEBUG] Import intercepted: {fullname}", file=sys.stderr)

    target_modules = {...}
    if fullname in target_modules:
        print(f"[DEBUG] Will patch: {fullname}", file=sys.stderr)
        return self
    return None
```

### Check Import Order

```bash
# See which vLLM modules are imported and when
python -X importtime -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf 2>&1 | grep vllm
```

### Trace System Calls

```bash
# See which files Python opens
strace -e trace=openat python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf 2>&1 | grep vllm
```

---

## Quick Reference: Activation Flow

```
1. Set PYTHONPATH
   export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"

2. (Optional) Force enable
   export VLLM_ENABLE_PROFILING=1

3. Run vLLM
   python -m vllm.entrypoints.openai.api_server --model <model>

4. Check for startup message
   [sitecustomize] vLLM Comprehensive Instrumentation Loaded

5. Check for patch messages
   [Instrumentation] ✅ Successfully patched <Component>

6. Send requests to generate data

7. Check CSV files
   ls /tmp/vllm_profiling/session_*/
```

---

## Getting Help

If patches still aren't activating after following this guide:

1. Run `python diagnose_patches.py` and save output
2. Run your vLLM command with stderr captured:
   ```bash
   python -m vllm.entrypoints.openai.api_server --model <model> 2>&1 | tee vllm_output.log
   ```
3. Share:
   - Diagnostic output
   - vLLM command used
   - vllm_output.log
   - vLLM version (`pip show vllm`)
   - Python version (`python --version`)
