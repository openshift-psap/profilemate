# ProfileMate Activation Control

ProfileMate now includes smart activation guards to **only activate for vLLM processes**, preventing it from affecting other Python scripts.

---

## How It Works

When `sitecustomize.py` is in your PYTHONPATH, it's imported by **every Python process**. ProfileMate now checks if it's running in a vLLM process before activating.

### Activation Logic

```python
def should_activate_profiling():
    # 1. Check explicit enable/disable
    if VLLM_ENABLE_PROFILING is set:
        return True/False based on value

    # 2. Auto-detect vLLM usage
    if command line contains 'vllm' or '--model':
        return True

    # 3. Default: don't activate
    return False
```

---

## Usage Methods

### Method 1: Auto-Detection (Recommended)

**No configuration needed!** ProfileMate automatically detects vLLM processes:

```bash
# Set PYTHONPATH once
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"

# Run vLLM - ProfileMate activates automatically ✅
python -m vllm.entrypoints.openai.api_server --model gpt2 --port 8000

# Run other Python scripts - ProfileMate stays silent ✅
python my_script.py
python -c "print('hello')"
```

**Auto-detection triggers on:**
- `vllm.entrypoints` in command
- `vllm.engine` in command
- `vllm` module name
- `--model` argument (common vLLM flag)

### Method 2: Explicit Control

Force enable or disable regardless of auto-detection:

```bash
# Force enable (even for non-vLLM scripts)
export VLLM_ENABLE_PROFILING=1
python my_script.py  # ProfileMate will activate

# Force disable (even for vLLM)
export VLLM_ENABLE_PROFILING=0
python -m vllm.entrypoints.openai.api_server --model gpt2  # ProfileMate won't activate

# Unset to use auto-detection
unset VLLM_ENABLE_PROFILING
```

### Method 3: Session-Specific Enable

Enable only for specific commands:

```bash
# Enable for one command only
VLLM_ENABLE_PROFILING=1 python -m vllm.entrypoints.openai.api_server --model gpt2

# Disable for one command only
VLLM_ENABLE_PROFILING=0 python -m vllm.entrypoints.openai.api_server --model gpt2
```

---

## Examples

### Example 1: Keep PYTHONPATH Set Permanently

```bash
# Add to ~/.bashrc or ~/.zshrc
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"

# Now you can run vLLM anytime with profiling
python -m vllm.entrypoints.openai.api_server --model gpt-oss-120b  # ✅ Profiling active

# Other Python scripts work normally
python my_analysis.py  # ✅ No profiling overhead
jupyter notebook       # ✅ No profiling overhead
pytest                 # ✅ No profiling overhead
```

### Example 2: Profile Some Runs, Not Others

```bash
# Profile this run
python -m vllm.entrypoints.openai.api_server --model gpt2 --port 8000

# Don't profile this run (disable explicitly)
VLLM_ENABLE_PROFILING=0 python -m vllm.entrypoints.openai.api_server --model gpt2 --port 8001
```

### Example 3: Debug Activation

```bash
# Check if ProfileMate would activate
python -c "
import os, sys
sys.argv = ['python', '-m', 'vllm.entrypoints.openai.api_server', '--model', 'gpt2']
exec(open('/path/to/profilemate/sitecustomize.py').read())
" 2>&1 | grep "vLLM Comprehensive Instrumentation Loaded"

# If you see the message, ProfileMate would activate
# If not, it wouldn't activate
```

---

## Verification

### Test 1: Normal Python Script (Should NOT Activate)

```bash
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"

python -c "print('Hello world')" 2>&1 | grep sitecustomize
# No output = ProfileMate didn't activate ✅
```

### Test 2: vLLM Process (Should Activate)

```bash
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"

python -m vllm.entrypoints.openai.api_server --model gpt2 --port 8000 2>&1 | grep "vLLM Comprehensive Instrumentation Loaded"
# Should see: "[sitecustomize] vLLM Comprehensive Instrumentation Loaded" ✅
```

### Test 3: Force Enable for Any Script

```bash
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
export VLLM_ENABLE_PROFILING=1

python -c "print('test')" 2>&1 | grep sitecustomize
# Should see ProfileMate messages ✅
```

---

## Troubleshooting

### Issue: ProfileMate Activates for Non-vLLM Scripts

**Cause:** Command line contains vLLM-like keywords

**Solution:** Use explicit disable:
```bash
export VLLM_ENABLE_PROFILING=0
python my_script.py
```

### Issue: ProfileMate Doesn't Activate for vLLM

**Possible causes:**

1. **VLLM_ENABLE_PROFILING=0 is set**
   ```bash
   echo $VLLM_ENABLE_PROFILING
   # If it shows "0", unset it:
   unset VLLM_ENABLE_PROFILING
   ```

2. **PYTHONPATH not set**
   ```bash
   echo $PYTHONPATH
   # Should contain path to profilemate
   ```

3. **Command doesn't match auto-detection patterns**
   ```bash
   # Force enable:
   export VLLM_ENABLE_PROFILING=1
   ```

4. **Running vLLM in unusual way**
   ```bash
   # Check what auto-detection sees:
   python -c "import sys; print(sys.argv)"

   # If it doesn't contain 'vllm' or '--model', force enable:
   export VLLM_ENABLE_PROFILING=1
   ```

### Issue: Want to Profile Only Specific vLLM Runs

**Solution:** Use session-specific enable/disable:

```bash
# Profile this one
VLLM_ENABLE_PROFILING=1 python -m vllm.entrypoints.openai.api_server --model gpt2 --port 8000

# Don't profile this one
VLLM_ENABLE_PROFILING=0 python -m vllm.entrypoints.openai.api_server --model gpt2 --port 8001

# Auto-detect for this one (will activate because it's vLLM)
python -m vllm.entrypoints.openai.api_server --model gpt2 --port 8002
```

---

## Advanced: Custom Activation Logic

If you need custom activation logic, edit `sitecustomize.py`:

```python
def should_activate_profiling():
    """Determine if profiling should be activated for this process"""

    # Your custom logic here
    import os

    # Example: Only activate on specific hostname
    import socket
    if socket.gethostname() != 'production-server':
        return False

    # Example: Only activate during specific hours
    import datetime
    hour = datetime.datetime.now().hour
    if hour < 9 or hour > 17:  # Only 9am-5pm
        return False

    # Example: Only activate for specific models
    cmdline = ' '.join(sys.argv)
    if 'gpt-oss-120b' not in cmdline:
        return False

    # Default auto-detection
    # ... (existing logic)
```

---

## Environment Variables Summary

| Variable | Value | Effect |
|----------|-------|--------|
| `VLLM_ENABLE_PROFILING` | (unset) | Auto-detect vLLM processes (default) |
| `VLLM_ENABLE_PROFILING` | `1` | Force enable profiling for all Python processes |
| `VLLM_ENABLE_PROFILING` | `0` | Force disable profiling even for vLLM |
| `VLLM_PROFILING_VERBOSE` | `1` | Enable verbose profiling logs (if active) |
| `VLLM_PROFILING_DIR` | `/path` | Change output directory (if active) |
| `VLLM_PROFILING_LOG_INTERVAL` | `N` | Sample every N operations (if active) |

---

## Best Practices

### For Development

```bash
# Keep PYTHONPATH set permanently
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"

# Auto-detection handles everything
python -m vllm.entrypoints.openai.api_server --model <model>  # ✅ Profiled
python my_dev_script.py                                        # ✅ Not profiled
```

### For Production

```bash
# Option 1: Only enable when needed
VLLM_ENABLE_PROFILING=1 python -m vllm.entrypoints.openai.api_server --model <model>

# Option 2: Set PYTHONPATH but disable by default
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
export VLLM_ENABLE_PROFILING=0

# Enable for specific debugging sessions:
VLLM_ENABLE_PROFILING=1 python -m vllm.entrypoints.openai.api_server --model <model>
```

### For CI/CD

```bash
# Profile specific test runs
if [ "$PROFILE_RUN" = "true" ]; then
  export VLLM_ENABLE_PROFILING=1
fi

python -m vllm.entrypoints.openai.api_server --model <model>
```

---

## Migration from Old Behavior

**Before (activated for all Python processes):**
```bash
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
python my_script.py  # ❌ ProfileMate activated (unwanted)
```

**After (only activates for vLLM):**
```bash
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
python my_script.py  # ✅ ProfileMate doesn't activate (desired)
```

**If you want the old behavior:**
```bash
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
export VLLM_ENABLE_PROFILING=1  # Force enable for all
python my_script.py  # ProfileMate activates
```

---

## Summary

✅ **Auto-detection**: ProfileMate only activates for vLLM processes by default
✅ **Explicit control**: Use `VLLM_ENABLE_PROFILING` to force enable/disable
✅ **Zero impact**: Non-vLLM scripts have zero overhead
✅ **Flexible**: Works with any vLLM invocation method

**Recommended Setup:**
```bash
# Add to ~/.bashrc
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"

# That's it! ProfileMate auto-detects vLLM processes
```
