"""
Simple vLLM Forward Pass Timing Profiler
========================================

A simplified profiler that immediately patches vLLM to track forward pass timing.
No import hooks - patches are applied eagerly when this module loads.

Usage:
    export PYTHONPATH="/path/to/profilemate/simple:$PYTHONPATH"
    export VLLM_ENABLE_PROFILING=1  # Optional: force enable
    python -m vllm.entrypoints.openai.api_server --model <model>

Output:
    /tmp/vllm_profiling/session_<timestamp>/
        - forward_pass_timing.csv
        - summary.json
"""

import sys
import os
import time
import csv
import json
import atexit
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Simple configuration"""
    OUTPUT_DIR = os.getenv("VLLM_PROFILING_DIR", "/tmp/vllm_profiling")
    ENABLE_PROFILING = os.getenv("VLLM_ENABLE_PROFILING")
    VERBOSE = os.getenv("VLLM_PROFILING_VERBOSE", "0") == "1"
    
    # Auto-detect vLLM if VLLM_ENABLE_PROFILING not set
    if ENABLE_PROFILING is None:
        cmdline = ' '.join(sys.argv)
        ENABLE_PROFILING = any(indicator in cmdline for indicator in [
            'vllm.entrypoints', 'vllm.engine', 'vllm', '--model'
        ])
    else:
        ENABLE_PROFILING = (ENABLE_PROFILING == "1")


# ============================================================================
# Data Collection
# ============================================================================

class ForwardPassProfiler:
    """Simple profiler for forward pass timing"""
    
    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.timings = []  # (timestamp, phase, batch_size, num_tokens, duration_ms)
        self.start_time = time.time()
        
    def record(self, phase: str, batch_size: int, num_tokens: int, duration_ms: float):
        """Record a forward pass timing"""
        timestamp = time.time() - self.start_time
        self.timings.append((timestamp, phase, batch_size, num_tokens, duration_ms))
        
        if Config.VERBOSE and len(self.timings) % 100 == 0:
            print(f"[Forward Pass] {phase}: {duration_ms:.2f}ms (batch={batch_size}, tokens={num_tokens})",
                  file=sys.stderr)
    
    def save(self):
        """Save timing data to CSV and summary"""
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Save CSV timeline
        csv_file = os.path.join(self.session_dir, "forward_pass_timing.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_sec', 'phase', 'batch_size', 'num_tokens', 
                           'duration_ms', 'throughput_tokens_per_sec'])
            
            for ts, phase, batch_size, num_tokens, duration_ms in self.timings:
                throughput = (num_tokens / (duration_ms / 1000)) if duration_ms > 0 else 0
                writer.writerow([
                    f"{ts:.3f}", phase, batch_size, num_tokens,
                    f"{duration_ms:.3f}", f"{throughput:.1f}"
                ])
        
        # Save summary statistics
        summary = self._compute_summary()
        summary_file = os.path.join(self.session_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[Simple Profiler] Saved statistics to {self.session_dir}/", file=sys.stderr)
        print(f"  - Total forward passes: {len(self.timings)}", file=sys.stderr)
        if summary:
            for phase, stats in summary.items():
                print(f"  - {phase}: mean={stats['mean_ms']:.2f}ms, count={stats['count']}", file=sys.stderr)
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics"""
        if not self.timings:
            return {}
        
        # Group by phase
        by_phase = defaultdict(list)
        for _, phase, _, _, duration_ms in self.timings:
            by_phase[phase].append(duration_ms)
        
        summary = {}
        for phase, durations in by_phase.items():
            durations_sorted = sorted(durations)
            n = len(durations_sorted)
            summary[phase] = {
                'count': n,
                'mean_ms': sum(durations) / n,
                'min_ms': durations_sorted[0],
                'max_ms': durations_sorted[-1],
                'p50_ms': durations_sorted[n // 2] if n > 0 else 0,
                'p95_ms': durations_sorted[int(n * 0.95)] if n > 1 else durations_sorted[0],
                'p99_ms': durations_sorted[int(n * 0.99)] if n > 1 else durations_sorted[0],
            }
        
        return summary


# ============================================================================
# Global Profiler Instance
# ============================================================================

_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
_session_dir = os.path.join(Config.OUTPUT_DIR, f"session_{_session_id}")
_profiler = None

if Config.ENABLE_PROFILING:
    _profiler = ForwardPassProfiler(_session_dir)
    print(f"\n[Simple Profiler] Forward pass timing enabled", file=sys.stderr)
    print(f"  Session ID: {_session_id}", file=sys.stderr)
    print(f"  Output directory: {_session_dir}", file=sys.stderr)


# ============================================================================
# Patching Functions
# ============================================================================

def patch_gpu_model_runner():
    """Patch GPUModelRunner.execute_model to track forward pass timing"""
    try:
        # Try to import the module - this will work if vLLM is installed
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        import torch
        
        # Check if already patched
        if hasattr(GPUModelRunner.execute_model, '_simple_profiler_patched'):
            return
        
        # Store original method
        original_execute_model = GPUModelRunner.execute_model
        
        def instrumented_execute_model(self, scheduler_output, intermediate_tensors=None):
            """Instrumented execute_model with timing"""
            if _profiler is None:
                return original_execute_model(self, scheduler_output, intermediate_tensors)
            
            # Determine phase (prefill or decode)
            # Try different ways to detect prefill based on vLLM version
            phase = 'decode'  # Default
            if hasattr(scheduler_output, 'num_prefill_reqs'):
                phase = 'prefill' if scheduler_output.num_prefill_reqs > 0 else 'decode'
            elif hasattr(scheduler_output, 'prefill_reqs') and scheduler_output.prefill_reqs:
                phase = 'prefill'
            
            # Get batch size
            batch_size = 0
            if hasattr(scheduler_output, 'scheduled_new_reqs'):
                batch_size += len(scheduler_output.scheduled_new_reqs)
            if hasattr(scheduler_output, 'scheduled_running_reqs'):
                batch_size += len(scheduler_output.scheduled_running_reqs)
            if batch_size == 0:
                batch_size = 1  # Fallback
            
            # Get token count
            num_tokens = 0
            if hasattr(scheduler_output, 'total_num_scheduled_tokens'):
                num_tokens = scheduler_output.total_num_scheduled_tokens
            elif hasattr(scheduler_output, 'num_scheduled_tokens'):
                # Might be a dict or single value
                tokens = scheduler_output.num_scheduled_tokens
                if isinstance(tokens, dict):
                    num_tokens = sum(tokens.values())
                else:
                    num_tokens = tokens
            
            # Time the forward pass
            start_time = time.perf_counter()
            
            # Execute original method
            result = original_execute_model(self, scheduler_output, intermediate_tensors)
            
            # Synchronize for accurate GPU timing
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Record timing
            _profiler.record(phase, batch_size, num_tokens, duration_ms)
            
            return result
        
        # Mark as patched
        instrumented_execute_model._simple_profiler_patched = True
        
        # Apply patch
        GPUModelRunner.execute_model = instrumented_execute_model
        
        print(f"[Simple Profiler] ✅ Successfully patched GPUModelRunner.execute_model", file=sys.stderr)
        
    except ImportError:
        # vLLM not installed or module not found - that's okay
        if Config.VERBOSE:
            print(f"[Simple Profiler] ⚠️  GPUModelRunner not found (vLLM may not be installed)", file=sys.stderr)
    except Exception as e:
        print(f"[Simple Profiler] ❌ Failed to patch GPUModelRunner: {e}", file=sys.stderr)
        if Config.VERBOSE:
            import traceback
            traceback.print_exc()


# ============================================================================
# Immediate Patching (Eager Loading)
# ============================================================================

def apply_patches():
    """Apply all patches immediately - try to import and patch right away"""
    if not Config.ENABLE_PROFILING:
        return
    
    # Try to patch immediately (if vLLM is already imported)
    patch_gpu_model_runner()
    
    # Set up import hook for when vLLM modules load later
    import importlib.util
    import importlib.abc
    
    class SimpleImportHook(importlib.abc.MetaPathFinder):
        """Simple import hook to patch vLLM modules when they're imported"""
        
        def find_spec(self, fullname, path, target=None):
            # Only intercept vLLM worker module
            if fullname == 'vllm.v1.worker.gpu_model_runner':
                # Let Python find the spec normally
                spec = importlib.util.find_spec(fullname)
                if spec is not None and spec.loader is not None:
                    # Wrap the loader to patch after loading
                    original_loader = spec.loader
                    
                    class PatchedLoader:
                        def __init__(self, loader):
                            self.loader = loader
                        
                        def create_module(self, spec):
                            return None  # Use default module creation
                        
                        def exec_module(self, module):
                            # Execute module normally
                            self.loader.exec_module(module)
                            # Then patch it
                            patch_gpu_model_runner()
                    
                    spec.loader = PatchedLoader(original_loader)
                return spec
            return None
    
    # Install the hook
    if not any(isinstance(finder, SimpleImportHook) for finder in sys.meta_path):
        sys.meta_path.insert(0, SimpleImportHook())


# ============================================================================
# Save on Exit
# ============================================================================

def save_stats():
    """Save profiling statistics on exit"""
    if _profiler:
        _profiler.save()

if Config.ENABLE_PROFILING:
    atexit.register(save_stats)
    # Try to apply patches immediately
    apply_patches()
