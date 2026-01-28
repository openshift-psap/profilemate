"""
vLLM Comprehensive Runtime Instrumentation
==========================================

Captures CUDA graph, KV cache, and MoE expert metrics during vLLM server runtime.

Features:
- CUDA graph usage tracking with full BatchDescriptor details
- KV cache allocation, usage, and eviction metrics
- MoE expert activation patterns and load balancing analysis
- Block pool statistics
- Automatic CSV export with detailed analysis

Installation:
    export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
    python -m vllm.entrypoints.openai.api_server --model <model>

Output Location:
    /tmp/vllm_profiling/session_<timestamp>/
        - cuda_graph_usage.csv
        - kv_cache_stats.csv
        - moe_expert_tracking/
            - moe_expert_activations.csv
            - moe_expert_coselection.csv
            - moe_routing_weights_hist.csv
            - moe_load_imbalance.csv
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
    ENABLE_MOE_EXPERT_TRACKING = True
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


class MoEExpertProfiler:
    """Profiles MoE expert activations and routing patterns"""

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.stats = {
            'expert_activations': defaultdict(lambda: defaultdict(int)),  # {layer_idx: {expert_id: count}}
            'routing_weights': [],  # (layer_idx, token_idx, expert_id, weight)
            'co_selection_patterns': defaultdict(lambda: defaultdict(int)),  # {layer_idx: {(expert1, expert2): count}}
            'expert_load_imbalance': [],  # (layer_idx, timestamp, std_dev, max_min_ratio)
        }
        self.start_time = time.time()
        self.num_experts_per_layer = {}  # {layer_idx: num_experts}
        self.top_k = {}  # {layer_idx: top_k value}
        self.sample_count = 0
        self.log_interval = ProfilingConfig.LOG_INTERVAL

    def record_expert_selection(
        self,
        layer_idx: int,
        topk_ids: Any,  # Tensor of shape [num_tokens, top_k]
        topk_weights: Any,  # Tensor of shape [num_tokens, top_k]
        num_experts: int,
        top_k: int
    ):
        """Record expert selection for a layer"""
        import torch

        self.sample_count += 1

        # Store layer configuration
        if layer_idx not in self.num_experts_per_layer:
            self.num_experts_per_layer[layer_idx] = num_experts
            self.top_k[layer_idx] = top_k

        # Convert to CPU numpy for analysis
        if isinstance(topk_ids, torch.Tensor):
            topk_ids_np = topk_ids.detach().cpu().numpy()
            topk_weights_np = topk_weights.detach().cpu().numpy()
        else:
            topk_ids_np = topk_ids
            topk_weights_np = topk_weights

        num_tokens = topk_ids_np.shape[0]

        # Track expert activations
        expert_counts = defaultdict(int)
        for token_idx in range(num_tokens):
            for k_idx in range(top_k):
                expert_id = int(topk_ids_np[token_idx, k_idx])
                weight = float(topk_weights_np[token_idx, k_idx])

                # Count activations
                self.stats['expert_activations'][layer_idx][expert_id] += 1
                expert_counts[expert_id] += 1

                # Sample routing weights (not all tokens to save memory)
                if self.sample_count % self.log_interval == 0:
                    self.stats['routing_weights'].append(
                        (layer_idx, token_idx, expert_id, weight)
                    )

                # Track co-selection patterns (which experts are selected together)
                if k_idx > 0:
                    prev_expert = int(topk_ids_np[token_idx, k_idx - 1])
                    co_key = tuple(sorted([prev_expert, expert_id]))
                    self.stats['co_selection_patterns'][layer_idx][co_key] += 1

        # Calculate load imbalance for this batch
        if expert_counts:
            counts = list(expert_counts.values())
            if len(counts) > 1:
                import numpy as np
                std_dev = float(np.std(counts))
                max_min_ratio = max(counts) / max(min(counts), 1)
                self.stats['expert_load_imbalance'].append(
                    (layer_idx, time.time() - self.start_time, std_dev, max_min_ratio)
                )

        if ProfilingConfig.VERBOSE and self.sample_count % self.log_interval == 0:
            unique_experts = len(expert_counts)
            print(f"[MoE Expert] Layer {layer_idx}: {unique_experts}/{num_experts} experts activated",
                  file=sys.stderr)

    def save_stats(self):
        """Save MoE expert tracking statistics"""
        import numpy as np

        moe_dir = os.path.join(self.session_dir, "moe_expert_tracking")
        os.makedirs(moe_dir, exist_ok=True)

        # Expert activations per layer
        activations_file = os.path.join(moe_dir, "moe_expert_activations.csv")
        with open(activations_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['layer_idx', 'expert_id', 'activation_count', 'percentage'])

            for layer_idx in sorted(self.stats['expert_activations'].keys()):
                expert_counts = self.stats['expert_activations'][layer_idx]
                total_activations = sum(expert_counts.values())

                for expert_id in sorted(expert_counts.keys()):
                    count = expert_counts[expert_id]
                    pct = (count / total_activations * 100) if total_activations > 0 else 0
                    writer.writerow([layer_idx, expert_id, count, f"{pct:.2f}"])

        # Co-selection patterns
        coselection_file = os.path.join(moe_dir, "moe_expert_coselection.csv")
        with open(coselection_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['layer_idx', 'expert_id_1', 'expert_id_2', 'coselection_count'])

            for layer_idx in sorted(self.stats['co_selection_patterns'].keys()):
                patterns = self.stats['co_selection_patterns'][layer_idx]
                for (exp1, exp2), count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                    writer.writerow([layer_idx, exp1, exp2, count])

        # Routing weights histogram (sampled)
        if self.stats['routing_weights']:
            weights_file = os.path.join(moe_dir, "moe_routing_weights_hist.csv")
            with open(weights_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['layer_idx', 'token_idx', 'expert_id', 'weight'])
                for layer_idx, token_idx, expert_id, weight in self.stats['routing_weights']:
                    writer.writerow([layer_idx, token_idx, expert_id, f"{weight:.6f}"])

        # Load imbalance timeline
        if self.stats['expert_load_imbalance']:
            imbalance_file = os.path.join(moe_dir, "moe_load_imbalance.csv")
            with open(imbalance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['layer_idx', 'timestamp_sec', 'std_dev', 'max_min_ratio'])
                for layer_idx, ts, std_dev, ratio in self.stats['expert_load_imbalance']:
                    writer.writerow([layer_idx, f"{ts:.3f}", f"{std_dev:.3f}", f"{ratio:.3f}"])

        # Summary statistics
        summary_file = os.path.join(moe_dir, "moe_summary.json")
        summary = {}

        for layer_idx in sorted(self.stats['expert_activations'].keys()):
            expert_counts = self.stats['expert_activations'][layer_idx]
            counts_array = np.array(list(expert_counts.values()))
            total_activations = sum(expert_counts.values())
            num_experts = self.num_experts_per_layer.get(layer_idx, len(expert_counts))

            summary[f"layer_{layer_idx}"] = {
                'num_experts': num_experts,
                'top_k': self.top_k.get(layer_idx, 2),
                'total_activations': int(total_activations),
                'unique_experts_activated': len(expert_counts),
                'activation_coverage_pct': (len(expert_counts) / num_experts * 100) if num_experts > 0 else 0,
                'mean_activations_per_expert': float(np.mean(counts_array)),
                'std_dev_activations': float(np.std(counts_array)),
                'min_activations': int(np.min(counts_array)),
                'max_activations': int(np.max(counts_array)),
                'load_balance_ratio': float(np.max(counts_array) / max(np.min(counts_array), 1)),
            }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[MoE Expert Profiler] Saved statistics to {moe_dir}/")
        print(f"  - Layers tracked: {len(self.stats['expert_activations'])}")
        total_unique_experts = sum(len(experts) for experts in self.stats['expert_activations'].values())
        print(f"  - Total unique expert activations: {total_unique_experts}")
        print(f"  - Routing weight samples: {len(self.stats['routing_weights'])}")


# ============================================================================
# Global Profiler Instances
# ============================================================================

_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
_session_dir = os.path.join(ProfilingConfig.OUTPUT_DIR, f"session_{_session_id}")
os.makedirs(_session_dir, exist_ok=True)

_cuda_profiler = CUDAGraphProfiler(_session_dir) if ProfilingConfig.ENABLE_CUDA_GRAPH_TRACKING else None
_kv_profiler = KVCacheProfiler(_session_dir) if ProfilingConfig.ENABLE_KV_CACHE_TRACKING else None
_moe_profiler = MoEExpertProfiler(_session_dir) if ProfilingConfig.ENABLE_MOE_EXPERT_TRACKING else None


def save_all_stats():
    """Save all profiling statistics on exit"""
    if _cuda_profiler:
        _cuda_profiler.save_stats()
    if _kv_profiler:
        _kv_profiler.save_stats()
    if _moe_profiler:
        _moe_profiler.save_stats()

    # Write metadata
    metadata_file = os.path.join(_session_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'session_id': _session_id,
            'start_time': datetime.now().isoformat(),
            'output_dir': _session_dir,
            'cuda_graph_tracking': ProfilingConfig.ENABLE_CUDA_GRAPH_TRACKING,
            'kv_cache_tracking': ProfilingConfig.ENABLE_KV_CACHE_TRACKING,
            'moe_expert_tracking': ProfilingConfig.ENABLE_MOE_EXPERT_TRACKING,
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


def patch_fused_moe():
    """Patch FusedMoE layer to track expert activations"""
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.logger import init_logger

        logger = init_logger(__name__)

        # Store original forward method
        original_forward = FusedMoE.forward

        def instrumented_forward(self, hidden_states):
            """Instrumented forward pass with expert tracking"""
            # Call original forward to get router logits and expert selection
            # We need to intercept the router output before it goes to the experts

            # Get router outputs first
            if hasattr(self, 'gate') or hasattr(self, 'router'):
                router = getattr(self, 'gate', None) or getattr(self, 'router', None)

                if router is not None:
                    import torch

                    # Get router logits
                    router_logits = router(hidden_states)

                    # Get top-k selection
                    top_k = getattr(self, 'top_k', 2)
                    num_experts = getattr(self, 'num_experts', router_logits.shape[-1])

                    # Calculate routing weights and expert IDs
                    routing_weights = torch.softmax(router_logits, dim=-1)
                    topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)

                    # Normalize weights
                    if hasattr(self, 'normalize_expert_weights'):
                        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

                    # Track expert selection if profiler is enabled
                    if _moe_profiler:
                        layer_idx = getattr(self, 'layer_idx', -1)

                        # If layer_idx not set, try to infer from self
                        if layer_idx == -1 and hasattr(self, 'layer_id'):
                            layer_idx = self.layer_id

                        _moe_profiler.record_expert_selection(
                            layer_idx=layer_idx,
                            topk_ids=topk_ids,
                            topk_weights=topk_weights,
                            num_experts=num_experts,
                            top_k=top_k
                        )

            # Call original forward
            return original_forward(self, hidden_states)

        FusedMoE.forward = instrumented_forward

        logger.info("[Instrumentation] Successfully patched FusedMoE")

    except ImportError:
        pass
    except Exception as e:
        print(f"[Instrumentation] Failed to patch FusedMoE: {e}", file=sys.stderr)


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
                'vllm.model_executor.layers.fused_moe.layer',
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
            elif fullname == 'vllm.model_executor.layers.fused_moe.layer':
                patch_fused_moe()

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
print(f"  MoE expert tracking: {ProfilingConfig.ENABLE_MOE_EXPERT_TRACKING}", file=sys.stderr)
print("", file=sys.stderr)
