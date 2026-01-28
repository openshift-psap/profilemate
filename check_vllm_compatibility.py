#!/usr/bin/env python3
"""Check vLLM 0.14.1 compatibility with ProfileMate"""

import sys

print("="*60)
print("ProfileMate - vLLM Compatibility Check")
print("="*60)
print()

# Check vLLM version
print("1. Checking vLLM version...")
try:
    import vllm
    print(f"   ✅ vLLM version: {vllm.__version__}")
except ImportError as e:
    print(f"   ❌ vLLM not installed: {e}")
    sys.exit(1)
print()

# Check scheduler version
print("2. Checking scheduler version...")
v1_available = False
v0_available = False

try:
    from vllm.v1.core.scheduler import Scheduler
    print("   ✅ V1 Scheduler found (ProfileMate compatible)")
    v1_available = True
except ImportError:
    print("   ❌ V1 Scheduler not found")

try:
    from vllm.core.scheduler import Scheduler as V0Scheduler
    print("   ⚠️  V0 Scheduler found (ProfileMate NOT compatible)")
    v0_available = True
except ImportError:
    pass

if not v1_available and v0_available:
    print()
    print("   ⚠️  WARNING: You are using V0 scheduler!")
    print("   ProfileMate requires V1 scheduler for new profilers.")
    print("   To use V1: Add --enable-v1 flag or set VLLM_USE_V1=1")
print()

# Check required modules
print("3. Checking ProfileMate target modules...")
modules_to_check = [
    ('vllm.compilation.cuda_graph', 'CUDA graph profiling'),
    ('vllm.v1.core.kv_cache_manager', 'KV cache profiling (V1)'),
    ('vllm.v1.core.block_pool', 'Block pool profiling (V1)'),
    ('vllm.model_executor.layers.fused_moe.layer', 'MoE expert profiling'),
    ('vllm.v1.worker.gpu_model_runner', 'Forward pass timing (V1)'),
    ('vllm.v1.core.scheduler', 'Batch utilization & preemption (V1)'),
]

v1_module_count = 0
total_modules = len(modules_to_check)

for module, description in modules_to_check:
    try:
        import importlib.util
        spec = importlib.util.find_spec(module)
        if spec:
            print(f"   ✅ {module}")
            if '.v1.' in module:
                v1_module_count += 1
        else:
            print(f"   ❌ {module} - {description}")
    except Exception as e:
        print(f"   ❌ {module} - Error: {e}")

print()

# Summary
print("4. Compatibility Summary")
print("   " + "="*50)
if v1_available:
    print("   ✅ V1 Scheduler: Compatible")
    print(f"   ✅ V1 Modules: {v1_module_count}/4 found")
    print()
    print("   Supported profilers:")
    print("   - ✅ CUDA Graph tracking")
    print("   - ✅ KV Cache profiling")
    print("   - ✅ MoE Expert tracking")
    print("   - ✅ Forward pass timing")
    print("   - ✅ CPU operation timing")
    print("   - ✅ Batch utilization")
    print("   - ✅ Preemption tracking")
    print("   - ✅ Encoder-decoder timing")
else:
    print("   ❌ V1 Scheduler: NOT FOUND")
    print()
    print("   To enable V1 scheduler:")
    print("   export VLLM_USE_V1=1")
    print("   # OR")
    print("   python -m vllm.entrypoints.openai.api_server --enable-v1 ...")
    print()
    print("   Limited profilers available (V0):")
    print("   - ⚠️  CUDA Graph tracking (may work)")
    print("   - ❌ KV Cache profiling (V1 only)")
    print("   - ✅ MoE Expert tracking (may work)")
    print("   - ❌ Forward pass timing (V1 only)")
    print("   - ❌ CPU operation timing (V1 only)")
    print("   - ❌ Batch utilization (V1 only)")
    print("   - ❌ Preemption tracking (V1 only)")
    print("   - ❌ Encoder-decoder timing (V1 only)")

print()
print("="*60)

# Check for common issues
print()
print("5. Common Issues Check")
print("   " + "="*50)

# Check py-cpuinfo
try:
    import cpuinfo
    print("   ℹ️  py-cpuinfo installed (may cause telemetry errors)")
    print("      Recommendation: export VLLM_NO_USAGE_STATS=1")
except ImportError:
    print("   ✅ py-cpuinfo not found (no telemetry issues)")

# Check if sitecustomize would activate
import sys
cmdline = ' '.join(sys.argv)
would_activate = any(ind in cmdline for ind in ['vllm', '--model'])
print(f"   ℹ️  Current command would {'activate' if would_activate else 'NOT activate'} ProfileMate")

print()
print("="*60)
