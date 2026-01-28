# ProfileMate Patch Flow - Visual Guide

## One-Page Visual Summary

```
╔══════════════════════════════════════════════════════════════════════╗
║                    PROFILEMATE PATCH ACTIVATION FLOW                  ║
╚══════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: PYTHON STARTUP (0-5ms)                                     │
│                                                                      │
│  User runs: python -m vllm.entrypoints.openai.api_server --model X  │
│       │                                                              │
│       ├─→ Python checks PYTHONPATH for sitecustomize.py             │
│       │                                                              │
│       └─→ Auto-imports sitecustomize.py (BEFORE any user code!)     │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: ACTIVATION DECISION (5-10ms)                               │
│                                                                      │
│  sitecustomize.py executes:                                         │
│       │                                                              │
│       ├─→ Define all classes (Profilers, Hook, etc.)                │
│       │                                                              │
│       ├─→ Define all patch functions                                │
│       │                                                              │
│       └─→ Call: should_activate_profiling()                         │
│                                                                      │
│           ┌──────────────────────────────────────────┐              │
│           │ ACTIVATION DECISION LOGIC                │              │
│           ├──────────────────────────────────────────┤              │
│           │ 1. Is VLLM_ENABLE_PROFILING set?         │              │
│           │    └─→ YES: Use that value (1=on, 0=off) │              │
│           │                                           │              │
│           │ 2. Does command contain vLLM indicators? │              │
│           │    (vllm.entrypoints, --model, etc.)     │              │
│           │    └─→ YES: Activate                     │              │
│           │                                           │              │
│           │ 3. Is vLLM even installed?               │              │
│           │    └─→ NO: Don't activate                │              │
│           │                                           │              │
│           │ 4. Default: Don't activate               │              │
│           └──────────────────────────────────────────┘              │
│                                                                      │
│       Decision: YES → Continue to Phase 3                           │
│       Decision: NO  → Exit silently ✅ (Zero overhead!)              │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: HOOK INSTALLATION (10ms)                                   │
│                                                                      │
│  Install import hook:                                               │
│       hook = VllmInstrumentationHook()                              │
│       sys.meta_path.insert(0, hook)  ← First in import chain!       │
│                                                                      │
│  Initialize profilers:                                              │
│       _cuda_profiler = CUDAGraphProfiler()                          │
│       _kv_profiler = KVCacheProfiler()                              │
│       _moe_profiler = MoEExpertProfiler()                           │
│       [... and 5 more ...]                                          │
│                                                                      │
│  Register exit handler:                                             │
│       atexit.register(save_all_stats)  ← Saves CSVs on exit         │
│                                                                      │
│  Print startup message:                                             │
│       [sitecustomize] vLLM Comprehensive Instrumentation Loaded     │
│       Session ID: 20260128_123456                                   │
│       Output directory: /tmp/vllm_profiling/session_20260128_123456 │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4: vLLM IMPORTS & PATCHING (100-1000ms)                       │
│                                                                      │
│  vLLM code executes: import vllm.compilation.cuda_graph             │
│       │                                                              │
│       ├─→ Python asks: Who handles this import?                     │
│       │                                                              │
│       ├─→ VllmInstrumentationHook.find_module() called              │
│       │   └─→ Returns: "I'll handle it!"                            │
│       │                                                              │
│       ├─→ VllmInstrumentationHook.load_module() called              │
│       │   ├─ Load module normally (module now exists!)              │
│       │   ├─ Call patch_cuda_graph_wrapper()                        │
│       │   │  └─→ Replace CUDAGraphWrapper.__call__                  │
│       │   │      with instrumented version                          │
│       │   ├─ Print: [Instrumentation] ✅ Successfully patched...    │
│       │   └─→ Return patched module                                 │
│       │                                                              │
│       └─→ vLLM gets the patched module!                             │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ SIMILAR PATCHING FOR OTHER MODULES:                           │  │
│  │                                                                │  │
│  │ import vllm.v1.core.sched.scheduler                            │  │
│  │   └─→ patch_scheduler()                                        │  │
│  │       └─→ Wraps Scheduler.schedule()                           │  │
│  │                                                                │  │
│  │ import vllm.v1.core.kv_cache_manager                           │  │
│  │   └─→ patch_kv_cache_manager()                                 │  │
│  │       └─→ Wraps KVCacheManager methods                         │  │
│  │                                                                │  │
│  │ import vllm.v1.worker.gpu_model_runner                         │  │
│  │   └─→ patch_gpu_model_runner()                                 │  │
│  │       └─→ Wraps GPUModelRunner.execute_model()                 │  │
│  │                                                                │  │
│  │ import vllm.model_executor.layers.fused_moe.layer              │  │
│  │   └─→ patch_fused_moe()                                        │  │
│  │       └─→ Wraps FusedMoE.forward()                             │  │
│  │                                                                │  │
│  │ import vllm.v1.core.block_pool                                 │  │
│  │   └─→ patch_block_pool()                                       │  │
│  │       └─→ Wraps BlockPool methods                              │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 5: RUNTIME EXECUTION (vLLM serving requests)                  │
│                                                                      │
│  Request arrives → vLLM processes it                                │
│       │                                                              │
│       ├─→ CUDAGraphWrapper.__call__() invoked                       │
│       │   │                                                          │
│       │   ├─ [Instrumented wrapper executes]                        │
│       │   ├─ start_time = time.perf_counter()                       │
│       │   ├─ result = original___call__(...)  ← Call real method    │
│       │   ├─ duration = time.perf_counter() - start_time            │
│       │   ├─ _cuda_profiler.record_replay(...)  ← Record metric     │
│       │   └─ return result                                          │
│       │                                                              │
│       ├─→ Scheduler.schedule() invoked                              │
│       │   ├─ [Instrumented wrapper executes]                        │
│       │   ├─ _batch_util_profiler.record(...)                       │
│       │   └─ return original_schedule(...)                          │
│       │                                                              │
│       └─→ ... (similar for all patched methods)                     │
│                                                                      │
│  Metrics accumulate in global profiler instances:                   │
│       _cuda_profiler.captures = [...]                               │
│       _cuda_profiler.replays = [...]                                │
│       _kv_profiler.usage_data = [...]                               │
│       _batch_util_profiler.utilization_data = [...]                 │
│       ... etc                                                        │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 6: SHUTDOWN (User presses Ctrl+C or process exits)            │
│                                                                      │
│  atexit handler triggered                                           │
│       │                                                              │
│       └─→ save_all_stats() called                                   │
│           │                                                          │
│           ├─→ _cuda_profiler.save()                                 │
│           │   └─→ Write cuda_graph_captures.csv                     │
│           │   └─→ Write cuda_graph_usage.csv                        │
│           │                                                          │
│           ├─→ _kv_profiler.save()                                   │
│           │   └─→ Write kv_cache_usage.csv                          │
│           │   └─→ Write kv_cache_evictions.csv                      │
│           │                                                          │
│           ├─→ _moe_profiler.save()                                  │
│           │   └─→ Write moe_expert_selection.csv                    │
│           │   └─→ Write moe_expert_timing.csv                       │
│           │                                                          │
│           ├─→ _forward_pass_profiler.save()                         │
│           │   └─→ Write forward_pass_timing.csv                     │
│           │                                                          │
│           ├─→ _cpu_profiler.save()                                  │
│           │   └─→ Write cpu_operations_timing.csv                   │
│           │                                                          │
│           ├─→ _batch_util_profiler.save()                           │
│           │   └─→ Write batch_utilization.csv                       │
│           │                                                          │
│           ├─→ _preemption_profiler.save()                           │
│           │   └─→ Write preemption_events.csv                       │
│           │                                                          │
│           └─→ _encoder_decoder_profiler.save()                      │
│               └─→ Write encoder_decoder_timing.csv                  │
│                                                                      │
│  Final output location:                                             │
│       /tmp/vllm_profiling/session_20260128_123456/                  │
│       ├─ cuda_graph_captures.csv                                    │
│       ├─ cuda_graph_usage.csv                                       │
│       ├─ kv_cache_usage.csv                                         │
│       ├─ kv_cache_evictions.csv                                     │
│       ├─ moe_expert_selection.csv                                   │
│       ├─ moe_expert_timing.csv                                      │
│       ├─ forward_pass_timing.csv                                    │
│       ├─ cpu_operations_timing.csv                                  │
│       ├─ batch_utilization.csv                                      │
│       ├─ preemption_events.csv                                      │
│       ├─ encoder_decoder_timing.csv                                 │
│       └─ metadata.json                                              │
└─────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════╗
║                         KEY INSIGHTS                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║ 1. Patches CANNOT be activated upfront because modules don't exist   ║
║    yet. They must be applied when vLLM imports its modules.          ║
║                                                                       ║
║ 2. Import hooks are the standard Python pattern for transparent      ║
║    instrumentation. They intercept imports and patch immediately.    ║
║                                                                       ║
║ 3. The activation guard prevents profiling non-vLLM processes.       ║
║    Set VLLM_ENABLE_PROFILING=1 to force enable for any script.       ║
║                                                                       ║
║ 4. Not all patches will activate for every vLLM run:                 ║
║    - CUDA graph patch won't activate if --enforce-eager is used      ║
║    - MoE patch won't activate for non-MoE models                     ║
║    - This is normal and expected!                                    ║
║                                                                       ║
║ 5. Global profiler instances are initialized once at startup and     ║
║    shared across all patches. This ensures consistent session IDs    ║
║    and unified output directory.                                     ║
║                                                                       ║
║ 6. atexit handler ensures CSV files are written even if process      ║
║    crashes or is interrupted (Ctrl+C).                               ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║                    TROUBLESHOOTING CHECKLIST                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║ □ PYTHONPATH includes profilemate directory                          ║
║   export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"               ║
║                                                                       ║
║ □ Activation not blocked                                             ║
║   export VLLM_ENABLE_PROFILING=1  # Force enable                     ║
║                                                                       ║
║ □ See startup message in stderr                                      ║
║   [sitecustomize] vLLM Comprehensive Instrumentation Loaded          ║
║                                                                       ║
║ □ See patch confirmation messages                                    ║
║   [Instrumentation] ✅ Successfully patched <Component>              ║
║                                                                       ║
║ □ CSV files created                                                  ║
║   ls /tmp/vllm_profiling/session_*/                                  ║
║                                                                       ║
║ □ CSV files contain data (not just headers)                          ║
║   Send inference requests to trigger instrumented code               ║
║                                                                       ║
║ □ Run diagnostic script if issues persist                            ║
║   python diagnose_patches.py                                         ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║                      METHOD WRAPPING PATTERN                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  All patches use the same pattern:                                   ║
║                                                                       ║
║  def patch_component():                                              ║
║      # 1. Import target class                                        ║
║      from vllm.xxx import TargetClass                                ║
║                                                                       ║
║      # 2. Store original method                                      ║
║      original_method = TargetClass.method                            ║
║                                                                       ║
║      # 3. Define instrumented wrapper                                ║
║      def instrumented_method(self, *args, **kwargs):                 ║
║          # Collect metrics BEFORE                                    ║
║          start = time.perf_counter()                                 ║
║                                                                       ║
║          # Call original method                                      ║
║          result = original_method(self, *args, **kwargs)             ║
║                                                                       ║
║          # Collect metrics AFTER                                     ║
║          duration = time.perf_counter() - start                      ║
║          _profiler.record(duration, ...)                             ║
║                                                                       ║
║          return result                                               ║
║                                                                       ║
║      # 4. Replace method on class                                    ║
║      TargetClass.method = instrumented_method                        ║
║                                                                       ║
║      # 5. Confirm success                                            ║
║      print("[Instrumentation] ✅ Successfully patched TargetClass")  ║
║                                                                       ║
║  This pattern:                                                       ║
║  ✅ Preserves original behavior (transparent to vLLM)                ║
║  ✅ Adds timing/metric collection                                    ║
║  ✅ Works across vLLM versions (no ABI dependencies)                 ║
║  ✅ Can be enabled/disabled via env var                              ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║                      QUICK START COMMANDS                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  # 1. Set PYTHONPATH                                                 ║
║  export PYTHONPATH="/home/nmiriyal/Documents/MLPERF-6.0/profilemate:$PYTHONPATH"
║                                                                       ║
║  # 2. (Optional) Force enable                                        ║
║  export VLLM_ENABLE_PROFILING=1                                      ║
║                                                                       ║
║  # 3. Run vLLM                                                       ║
║  python -m vllm.entrypoints.openai.api_server \                      ║
║      --model meta-llama/Llama-2-7b-hf \                              ║
║      --port 8000                                                     ║
║                                                                       ║
║  # 4. Send test request                                              ║
║  curl http://localhost:8000/v1/completions \                         ║
║      -H "Content-Type: application/json" \                           ║
║      -d '{"model": "meta-llama/Llama-2-7b-hf", \                     ║
║           "prompt": "Hello!", "max_tokens": 50}'                     ║
║                                                                       ║
║  # 5. Check output                                                   ║
║  ls /tmp/vllm_profiling/session_*/                                   ║
║  cat /tmp/vllm_profiling/session_*/cuda_graph_usage.csv              ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
```
