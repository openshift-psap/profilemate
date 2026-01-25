"""
vLLM Comprehensive Runtime Instrumentation
==========================================

Captures CUDA graph and KV cache metrics during vLLM server runtime.

Features:
- CUDA graph usage tracking with full BatchDescriptor details
- KV cache allocation, usage, and eviction metrics
- Block pool statistics
- Automatic CSV export with detailed analysis

Installation:
    export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
    python -m vllm.entrypoints.openai.api_server --model <model>

Output Location:
    /tmp/vllm_profiling/session_<timestamp>/
        - cuda_graph_usage.csv
        - kv_cache_stats.csv
        - block_allocations.csv
        - summary.txt
"""

import sys
import os
import time
import csv
import atexit
import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================================================
# Configuration
# ============================================================================

class ProfilingConfig:
    """Configuration for profiling output"""
    OUTPUT_DIR = os.getenv("VLLM_PROFILING_DIR", "/tmp/vllm_profiling")
    LOG_INTERVAL = int(os.getenv("VLLM_PROFILING_LOG_INTERVAL", "100"))
    ENABLE_CUDA_GRAPH_TRACKING = True
    ENABLE_KV_CACHE_TRACKING = True
    VERBOSE = os.getenv("VLLM_PROFILING_VERBOSE", "0") == "1"


# ============================================================================
# Data Collection Classes
# ============================================================================

class CUDAGraphProfiler:
    """Profiles CUDA graph execution with detailed BatchDescriptor tracking"""

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.stats = {
            'graphs_captured': {},  # BatchDescriptor -> capture_time
            'graph_replays': defaultdict(int),  # BatchDescriptor -> count
            'replay_times': [],  # (timestamp, descriptor, duration)
        }
        self.start_time = time.time()

    def record_capture(self, batch_descriptor: str, runtime_mode: str):
        """Record CUDA graph capture event"""
        key = f"{runtime_mode}:{batch_descriptor}"
        self.stats['graphs_captured'][key] = time.time() - self.start_time

        if ProfilingConfig.VERBOSE:
            print(f"[CUDA Graph] Captured: {key}", file=sys.stderr)

    def record_replay(self, batch_descriptor: str, runtime_mode: str, duration: float = 0):
        """Record CUDA graph replay event"""
        key = f"{runtime_mode}:{batch_descriptor}"
        self.stats['graph_replays'][key] += 1
        self.stats['replay_times'].append(
            (time.time() - self.start_time, key, duration)
        )

    def save_stats(self):
        """Save CUDA graph statistics"""
        # Captured graphs
        captures_file = os.path.join(self.session_dir, "cuda_graph_captures.csv")
        with open(captures_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['runtime_mode', 'num_tokens', 'num_reqs',
                           'uniform', 'has_lora', 'capture_time_sec'])

            for descriptor, capture_time in self.stats['graphs_captured'].items():
                mode, desc = descriptor.split(':', 1)
                # Parse BatchDescriptor
                parts = self._parse_descriptor(desc)
                writer.writerow([mode, *parts, f"{capture_time:.3f}"])

        # Replay statistics
        usage_file = os.path.join(self.session_dir, "cuda_graph_usage.csv")
        with open(usage_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['runtime_mode', 'num_tokens', 'num_reqs',
                           'uniform', 'has_lora', 'replay_count'])

            for descriptor, count in sorted(
                self.stats['graph_replays'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                mode, desc = descriptor.split(':', 1)
                parts = self._parse_descriptor(desc)
                writer.writerow([mode, *parts, count])

        # Detailed timeline
        timeline_file = os.path.join(self.session_dir, "cuda_graph_timeline.csv")
        with open(timeline_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_sec', 'descriptor', 'duration_ms'])
            for ts, desc, duration in self.stats['replay_times']:
                writer.writerow([f"{ts:.3f}", desc, f"{duration*1000:.3f}"])

        print(f"\n[CUDA Graph Profiler] Saved statistics to {self.session_dir}/")
        print(f"  - Unique graphs captured: {len(self.stats['graphs_captured'])}")
        print(f"  - Total replays: {sum(self.stats['graph_replays'].values())}")

    def _parse_descriptor(self, desc: str) -> List[str]:
        """Parse BatchDescriptor string into components"""
        # Expected format: "BatchDescriptor(num_tokens=X, num_reqs=Y, uniform=Z, has_lora=W)"
        try:
            # Extract values using simple parsing
            import re
            num_tokens = re.search(r'num_tokens=(\d+)', desc)
            num_reqs = re.search(r'num_reqs=(\w+)', desc)
            uniform = re.search(r'uniform=(\w+)', desc)
            has_lora = re.search(r'has_lora=(\w+)', desc)

            return [
                num_tokens.group(1) if num_tokens else 'unknown',
                num_reqs.group(1) if num_reqs else 'None',
                uniform.group(1) if uniform else 'False',
                has_lora.group(1) if has_lora else 'False',
            ]
        except:
            return [desc, 'unknown', 'unknown', 'unknown']


class KVCacheProfiler:
    """Profiles KV cache allocation and usage"""

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'current_blocks': 0,
            'peak_blocks': 0,
            'usage_samples': [],  # (timestamp, usage_pct, num_blocks)
            'evictions': [],  # (timestamp, lifetime_sec, idle_sec)
        }
        self.start_time = time.time()
        self.total_blocks = 0

    def set_total_blocks(self, num_blocks: int):
        """Set total number of available KV cache blocks"""
        self.total_blocks = num_blocks

    def record_allocation(self):
        """Record a block allocation"""
        self.stats['allocations'] += 1
        self.stats['current_blocks'] += 1
        self.stats['peak_blocks'] = max(
            self.stats['peak_blocks'],
            self.stats['current_blocks']
        )

    def record_deallocation(self):
        """Record a block deallocation"""
        self.stats['deallocations'] += 1
        self.stats['current_blocks'] = max(0, self.stats['current_blocks'] - 1)

    def record_usage(self, usage_pct: float, num_blocks: int):
        """Record KV cache usage snapshot"""
        self.stats['usage_samples'].append(
            (time.time() - self.start_time, usage_pct, num_blocks)
        )

    def record_eviction(self, lifetime_sec: float, idle_sec: float):
        """Record a block eviction event"""
        self.stats['evictions'].append(
            (time.time() - self.start_time, lifetime_sec, idle_sec)
        )

    def save_stats(self):
        """Save KV cache statistics"""
        # Usage timeline
        usage_file = os.path.join(self.session_dir, "kv_cache_usage.csv")
        with open(usage_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_sec', 'usage_pct', 'num_blocks', 'total_blocks'])
            for ts, usage, blocks in self.stats['usage_samples']:
                writer.writerow([f"{ts:.3f}", f"{usage:.2f}", blocks, self.total_blocks])

        # Eviction events
        if self.stats['evictions']:
            evictions_file = os.path.join(self.session_dir, "kv_cache_evictions.csv")
            with open(evictions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp_sec', 'lifetime_sec', 'idle_sec'])
                for ts, lifetime, idle in self.stats['evictions']:
                    writer.writerow([f"{ts:.3f}", f"{lifetime:.3f}", f"{idle:.3f}"])

        # Summary
        summary_file = os.path.join(self.session_dir, "kv_cache_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("KV Cache Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total blocks available: {self.total_blocks}\n")
            f.write(f"Total allocations: {self.stats['allocations']}\n")
            f.write(f"Total deallocations: {self.stats['deallocations']}\n")
            f.write(f"Peak blocks used: {self.stats['peak_blocks']}\n")
            if self.total_blocks > 0:
                peak_pct = (self.stats['peak_blocks'] / self.total_blocks) * 100
                f.write(f"Peak usage: {peak_pct:.2f}%\n")
            f.write(f"Total evictions: {len(self.stats['evictions'])}\n")

            if self.stats['evictions']:
                lifetimes = [e[1] for e in self.stats['evictions']]
                idles = [e[2] for e in self.stats['evictions']]
                f.write(f"Avg block lifetime: {sum(lifetimes)/len(lifetimes):.2f}s\n")
                f.write(f"Avg idle before eviction: {sum(idles)/len(idles):.2f}s\n")

        print(f"\n[KV Cache Profiler] Saved statistics to {self.session_dir}/")
        print(f"  - Total blocks: {self.total_blocks}")
        print(f"  - Peak usage: {self.stats['peak_blocks']} blocks")


# ============================================================================
# Global Profiler Instances
# ============================================================================

_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
_session_dir = os.path.join(ProfilingConfig.OUTPUT_DIR, f"session_{_session_id}")
os.makedirs(_session_dir, exist_ok=True)

_cuda_profiler = CUDAGraphProfiler(_session_dir) if ProfilingConfig.ENABLE_CUDA_GRAPH_TRACKING else None
_kv_profiler = KVCacheProfiler(_session_dir) if ProfilingConfig.ENABLE_KV_CACHE_TRACKING else None


def save_all_stats():
    """Save all profiling statistics on exit"""
    if _cuda_profiler:
        _cuda_profiler.save_stats()
    if _kv_profiler:
        _kv_profiler.save_stats()

    # Write metadata
    metadata_file = os.path.join(_session_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'session_id': _session_id,
            'start_time': datetime.now().isoformat(),
            'output_dir': _session_dir,
            'cuda_graph_tracking': ProfilingConfig.ENABLE_CUDA_GRAPH_TRACKING,
            'kv_cache_tracking': ProfilingConfig.ENABLE_KV_CACHE_TRACKING,
        }, f, indent=2)


atexit.register(save_all_stats)


# ============================================================================
# Instrumentation Patches
# ============================================================================

def patch_cuda_graph_wrapper():
    """Patch CUDAGraphWrapper to track graph usage"""
    try:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.logger import init_logger
        import torch

        logger = init_logger(__name__)

        # Store original methods
        original_call = CUDAGraphWrapper.__call__

        def instrumented_call(self, *args, **kwargs):
            """Instrumented __call__ with tracking"""
            from vllm.forward_context import get_forward_context

            forward_context = get_forward_context()
            batch_descriptor = forward_context.batch_descriptor
            cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

            if batch_descriptor is not None and _cuda_profiler:
                # Check if this is a new capture or replay
                if batch_descriptor not in self.concrete_cudagraph_entries:
                    # About to capture
                    _cuda_profiler.record_capture(
                        str(batch_descriptor),
                        str(cudagraph_runtime_mode)
                    )
                else:
                    # Replay
                    start_time = time.perf_counter()
                    result = original_call(self, *args, **kwargs)

                    if hasattr(torch.cuda, 'synchronize'):
                        torch.cuda.synchronize()
                    duration = time.perf_counter() - start_time

                    _cuda_profiler.record_replay(
                        str(batch_descriptor),
                        str(cudagraph_runtime_mode),
                        duration
                    )
                    return result

            return original_call(self, *args, **kwargs)

        CUDAGraphWrapper.__call__ = instrumented_call
        logger.info("[Instrumentation] Successfully patched CUDAGraphWrapper")

    except ImportError:
        pass
    except Exception as e:
        print(f"[Instrumentation] Failed to patch CUDAGraphWrapper: {e}", file=sys.stderr)


def patch_kv_cache_manager():
    """Patch KVCacheManager to track cache usage"""
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.logger import init_logger

        logger = init_logger(__name__)

        # Store original methods
        original_init = KVCacheManager.__init__
        original_usage = KVCacheManager.usage

        def instrumented_init(self, *args, **kwargs):
            """Instrumented __init__ to capture total blocks"""
            result = original_init(self, *args, **kwargs)

            if _kv_profiler and hasattr(self, 'block_pool'):
                total_blocks = getattr(self.block_pool, 'num_blocks', 0)
                _kv_profiler.set_total_blocks(total_blocks)
                logger.info(f"[KV Cache] Initialized with {total_blocks} blocks")

            return result

        @property
        def instrumented_usage(self):
            """Instrumented usage property to track cache usage"""
            usage = original_usage.fget(self)

            if _kv_profiler and hasattr(self, 'block_pool'):
                num_blocks = getattr(self.block_pool, 'num_free_blocks', 0)
                total = getattr(self.block_pool, 'num_blocks', 1)
                used_blocks = total - num_blocks
                _kv_profiler.record_usage(usage * 100, used_blocks)

            return usage

        KVCacheManager.__init__ = instrumented_init
        KVCacheManager.usage = instrumented_usage

        logger.info("[Instrumentation] Successfully patched KVCacheManager")

    except ImportError:
        pass
    except Exception as e:
        print(f"[Instrumentation] Failed to patch KVCacheManager: {e}", file=sys.stderr)


def patch_block_pool():
    """Patch BlockPool to track allocations/deallocations"""
    try:
        from vllm.v1.core.block_pool import BlockPool
        from vllm.logger import init_logger

        logger = init_logger(__name__)

        # Store original methods
        original_allocate = BlockPool.allocate_immutable_block

        def instrumented_allocate(self, *args, **kwargs):
            """Instrumented allocate to track allocations"""
            result = original_allocate(self, *args, **kwargs)

            if _kv_profiler and result is not None:
                _kv_profiler.record_allocation()

            return result

        BlockPool.allocate_immutable_block = instrumented_allocate

        logger.info("[Instrumentation] Successfully patched BlockPool")

    except ImportError:
        pass
    except Exception as e:
        print(f"[Instrumentation] Failed to patch BlockPool: {e}", file=sys.stderr)


# ============================================================================
# Import Hook Installation
# ============================================================================

def install_import_hook():
    """Install import hook to patch vLLM modules after they're loaded"""
    import importlib.abc
    import importlib.machinery

    class VllmInstrumentationHook(importlib.abc.MetaPathFinder):
        """Import hook that patches vLLM modules after they're loaded"""

        def find_module(self, fullname, path=None):
            # Trigger on specific vLLM modules
            target_modules = [
                'vllm.compilation.cuda_graph',
                'vllm.v1.core.kv_cache_manager',
                'vllm.v1.core.block_pool',
            ]

            if fullname in target_modules:
                return self
            return None

        def load_module(self, fullname):
            """Load module and apply patches"""
            if fullname in sys.modules:
                return sys.modules[fullname]

            # Standard import
            import importlib
            module = importlib.import_module(fullname)

            # Apply appropriate patches
            if fullname == 'vllm.compilation.cuda_graph':
                patch_cuda_graph_wrapper()
            elif fullname == 'vllm.v1.core.kv_cache_manager':
                patch_kv_cache_manager()
            elif fullname == 'vllm.v1.core.block_pool':
                patch_block_pool()

            return module

    # Install the hook
    sys.meta_path.insert(0, VllmInstrumentationHook())


# ============================================================================
# Initialization
# ============================================================================

# Install the import hook immediately
install_import_hook()

print(f"\n[sitecustomize] vLLM Comprehensive Instrumentation Loaded", file=sys.stderr)
print(f"  Session ID: {_session_id}", file=sys.stderr)
print(f"  Output directory: {_session_dir}", file=sys.stderr)
print(f"  CUDA graph tracking: {ProfilingConfig.ENABLE_CUDA_GRAPH_TRACKING}", file=sys.stderr)
print(f"  KV cache tracking: {ProfilingConfig.ENABLE_KV_CACHE_TRACKING}", file=sys.stderr)
print("", file=sys.stderr)
