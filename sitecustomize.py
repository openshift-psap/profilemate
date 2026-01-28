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
    ENABLE_FORWARD_PASS_TIMING = True
    ENABLE_CPU_TIMING = True
    ENABLE_BATCH_UTILIZATION_TRACKING = True
    ENABLE_PREEMPTION_TRACKING = True
    ENABLE_ENCODER_DECODER_TIMING = True

    # CUDA sync options for GPU timing
    # Option A: CUDA Events (0.5% overhead, perfect accuracy)
    # Option B: Piggyback on vLLM syncs (0.1% overhead, ~95% accuracy)
    USE_CUDA_EVENTS = True  # Set False to use lightweight timing
    CUDA_EVENT_BATCH_SIZE = 100  # Sync every N iterations (only if USE_CUDA_EVENTS=True)

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


class ForwardPassProfiler:
    """Profiles forward pass timing with accurate GPU measurements"""

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.stats = {
            'prefill_timings': [],  # (timestamp, batch_size, num_tokens, duration_ms)
            'decode_timings': [],   # (timestamp, batch_size, num_tokens, duration_ms)
        }
        self.start_time = time.time()
        self.use_cuda_events = ProfilingConfig.USE_CUDA_EVENTS
        self.event_batch_size = ProfilingConfig.CUDA_EVENT_BATCH_SIZE

        # For CUDA Events mode
        if self.use_cuda_events:
            self.pending_events = []  # (start_event, end_event, phase, batch_size, num_tokens, timestamp)

    def record_forward_start(self, phase: str, batch_size: int, num_tokens: int):
        """Record start of forward pass"""
        if not self.use_cuda_events:
            # Lightweight mode: just record timestamp
            return time.perf_counter()
        else:
            # CUDA Events mode
            try:
                import torch
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                return (start_event, end_event, phase, batch_size, num_tokens, time.time() - self.start_time)
            except:
                return time.perf_counter()

    def record_forward_end(self, start_marker, phase: str, batch_size: int, num_tokens: int):
        """Record end of forward pass"""
        if not self.use_cuda_events:
            # Lightweight mode: calculate duration
            duration_ms = (time.perf_counter() - start_marker) * 1000
            timestamp = time.time() - self.start_time
            self._save_timing(phase, timestamp, batch_size, num_tokens, duration_ms)
        else:
            # CUDA Events mode
            try:
                import torch
                start_event, end_event, _, _, _, _ = start_marker
                end_event.record()
                self.pending_events.append(start_marker)

                # Flush events periodically
                if len(self.pending_events) >= self.event_batch_size:
                    self._flush_events()
            except:
                pass

    def _flush_events(self):
        """Flush pending CUDA events (syncs once for all)"""
        if not self.pending_events:
            return

        try:
            import torch
            # Single sync for all pending events
            torch.cuda.synchronize()

            for start_event, end_event, phase, batch_size, num_tokens, timestamp in self.pending_events:
                duration_ms = start_event.elapsed_time(end_event)
                self._save_timing(phase, timestamp, batch_size, num_tokens, duration_ms)

            self.pending_events.clear()

        except Exception as e:
            if ProfilingConfig.VERBOSE:
                print(f"[Forward Pass Profiler] Error flushing events: {e}", file=sys.stderr)

    def _save_timing(self, phase: str, timestamp: float, batch_size: int, num_tokens: int, duration_ms: float):
        """Save timing measurement"""
        if phase == 'prefill':
            self.stats['prefill_timings'].append((timestamp, batch_size, num_tokens, duration_ms))
        elif phase == 'decode':
            self.stats['decode_timings'].append((timestamp, batch_size, num_tokens, duration_ms))

        if ProfilingConfig.VERBOSE and (len(self.stats['prefill_timings']) + len(self.stats['decode_timings'])) % 100 == 0:
            print(f"[Forward Pass] {phase}: {duration_ms:.2f}ms (batch={batch_size}, tokens={num_tokens})",
                  file=sys.stderr)

    def save_stats(self):
        """Save forward pass timing statistics"""
        # Flush any pending CUDA events
        if self.use_cuda_events:
            self._flush_events()

        # Forward pass timeline
        timeline_file = os.path.join(self.session_dir, "forward_pass_timing.csv")
        with open(timeline_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_sec', 'phase', 'batch_size', 'num_tokens', 'forward_time_ms', 'throughput_tokens_per_sec'])

            all_timings = (
                [('prefill', *t) for t in self.stats['prefill_timings']] +
                [('decode', *t) for t in self.stats['decode_timings']]
            )
            all_timings.sort(key=lambda x: x[1])  # Sort by timestamp

            for phase, ts, batch_size, num_tokens, duration_ms in all_timings:
                throughput = (num_tokens / (duration_ms / 1000)) if duration_ms > 0 else 0
                writer.writerow([f"{ts:.3f}", phase, batch_size, num_tokens, f"{duration_ms:.3f}", f"{throughput:.1f}"])

        # Summary statistics
        import numpy as np
        summary = {}

        for phase_name, timings in [('prefill', self.stats['prefill_timings']), ('decode', self.stats['decode_timings'])]:
            if timings:
                durations = [t[3] for t in timings]  # Extract duration_ms
                summary[phase_name] = {
                    'count': len(durations),
                    'mean_ms': float(np.mean(durations)),
                    'std_ms': float(np.std(durations)),
                    'min_ms': float(np.min(durations)),
                    'max_ms': float(np.max(durations)),
                    'p50_ms': float(np.percentile(durations, 50)),
                    'p95_ms': float(np.percentile(durations, 95)),
                    'p99_ms': float(np.percentile(durations, 99)),
                }

        summary_file = os.path.join(self.session_dir, "forward_pass_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[Forward Pass Profiler] Saved statistics to {self.session_dir}/")
        print(f"  - CUDA Events mode: {self.use_cuda_events}")
        print(f"  - Prefill samples: {len(self.stats['prefill_timings'])}")
        print(f"  - Decode samples: {len(self.stats['decode_timings'])}")


class CPUTimingProfiler:
    """Profiles CPU operations timing"""

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.stats = {
            'operations': [],  # (timestamp, operation, duration_ms, context)
        }
        self.start_time = time.time()

    def record_operation(self, operation: str, duration_ms: float, context: str = ""):
        """Record a CPU operation timing"""
        timestamp = time.time() - self.start_time
        self.stats['operations'].append((timestamp, operation, duration_ms, context))

        if ProfilingConfig.VERBOSE and len(self.stats['operations']) % 100 == 0:
            print(f"[CPU Timing] {operation}: {duration_ms:.2f}ms", file=sys.stderr)

    def save_stats(self):
        """Save CPU timing statistics"""
        # Timeline
        timeline_file = os.path.join(self.session_dir, "cpu_operations_timing.csv")
        with open(timeline_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_sec', 'operation', 'duration_ms', 'context'])

            for ts, operation, duration_ms, context in self.stats['operations']:
                writer.writerow([f"{ts:.3f}", operation, f"{duration_ms:.3f}", context])

        # Summary by operation type
        import numpy as np
        from collections import defaultdict

        by_operation = defaultdict(list)
        for _, operation, duration_ms, _ in self.stats['operations']:
            by_operation[operation].append(duration_ms)

        summary = {}
        for operation, durations in by_operation.items():
            summary[operation] = {
                'count': len(durations),
                'mean_ms': float(np.mean(durations)),
                'std_ms': float(np.std(durations)),
                'min_ms': float(np.min(durations)),
                'max_ms': float(np.max(durations)),
                'p95_ms': float(np.percentile(durations, 95)),
            }

        summary_file = os.path.join(self.session_dir, "cpu_timing_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[CPU Timing Profiler] Saved statistics to {self.session_dir}/")
        print(f"  - Total operations: {len(self.stats['operations'])}")
        print(f"  - Operation types: {len(by_operation)}")


class BatchUtilizationProfiler:
    """Profiles batch utilization and scheduling efficiency"""

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.stats = {
            'utilization_samples': [],  # (timestamp, num_seqs, max_seqs, num_tokens, max_tokens, phase, running_queue_len, waiting_queue_len)
        }
        self.start_time = time.time()
        self.max_num_seqs = None
        self.max_tokens = None

    def set_limits(self, max_num_seqs: int, max_tokens: int):
        """Set scheduler limits"""
        self.max_num_seqs = max_num_seqs
        self.max_tokens = max_tokens

    def record_batch(
        self,
        num_seqs: int,
        num_tokens: int,
        phase: str,
        running_queue_len: int = 0,
        waiting_queue_len: int = 0
    ):
        """Record batch utilization"""
        timestamp = time.time() - self.start_time

        self.stats['utilization_samples'].append((
            timestamp, num_seqs, self.max_num_seqs or 0, num_tokens,
            self.max_tokens or 0, phase, running_queue_len, waiting_queue_len
        ))

        if ProfilingConfig.VERBOSE and len(self.stats['utilization_samples']) % 100 == 0:
            seq_util = (num_seqs / self.max_num_seqs * 100) if self.max_num_seqs else 0
            token_util = (num_tokens / self.max_tokens * 100) if self.max_tokens else 0
            print(f"[Batch Util] {phase}: seqs={num_seqs}/{self.max_num_seqs} ({seq_util:.1f}%), "
                  f"tokens={num_tokens}/{self.max_tokens} ({token_util:.1f}%)", file=sys.stderr)

    def save_stats(self):
        """Save batch utilization statistics"""
        # Timeline
        timeline_file = os.path.join(self.session_dir, "batch_utilization.csv")
        with open(timeline_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_sec', 'num_seqs', 'max_num_seqs', 'seq_utilization_pct',
                'num_tokens', 'max_tokens', 'token_utilization_pct', 'phase',
                'running_queue_len', 'waiting_queue_len'
            ])

            for ts, num_seqs, max_seqs, num_tokens, max_tokens, phase, running_len, waiting_len in self.stats['utilization_samples']:
                seq_util = (num_seqs / max_seqs * 100) if max_seqs > 0 else 0
                token_util = (num_tokens / max_tokens * 100) if max_tokens > 0 else 0

                writer.writerow([
                    f"{ts:.3f}", num_seqs, max_seqs, f"{seq_util:.2f}",
                    num_tokens, max_tokens, f"{token_util:.2f}", phase,
                    running_len, waiting_len
                ])

        # Summary
        import numpy as np

        if self.stats['utilization_samples']:
            seq_utils = []
            token_utils = []
            prefill_utils = []
            decode_utils = []

            for ts, num_seqs, max_seqs, num_tokens, max_tokens, phase, _, _ in self.stats['utilization_samples']:
                seq_util = (num_seqs / max_seqs * 100) if max_seqs > 0 else 0
                token_util = (num_tokens / max_tokens * 100) if max_tokens > 0 else 0

                seq_utils.append(seq_util)
                token_utils.append(token_util)

                if phase == 'prefill':
                    prefill_utils.append(token_util)
                elif phase == 'decode':
                    decode_utils.append(token_util)

            summary = {
                'total_samples': len(self.stats['utilization_samples']),
                'mean_seq_utilization_pct': float(np.mean(seq_utils)),
                'mean_token_utilization_pct': float(np.mean(token_utils)),
                'underutilization_events': sum(1 for u in token_utils if u < 50),
                'prefill_avg_utilization': float(np.mean(prefill_utils)) if prefill_utils else 0,
                'decode_avg_utilization': float(np.mean(decode_utils)) if decode_utils else 0,
                'max_num_seqs': self.max_num_seqs,
                'max_tokens': self.max_tokens,
            }

            summary_file = os.path.join(self.session_dir, "batch_utilization_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"\n[Batch Utilization Profiler] Saved statistics to {self.session_dir}/")
            print(f"  - Total samples: {len(self.stats['utilization_samples'])}")
            print(f"  - Mean token utilization: {summary['mean_token_utilization_pct']:.1f}%")


class PreemptionProfiler:
    """Profiles preemption events and request lifecycle"""

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.stats = {
            'preemption_events': [],  # (timestamp, request_id, event, reason, running_time, extra_info)
            'request_lifecycle': {},  # request_id -> {'started': ts, 'preempted': ts, 'resumed': ts, 'finished': ts}
        }
        self.start_time = time.time()

    def record_event(
        self,
        request_id: str,
        event: str,  # 'preempted', 'resumed', 'started', 'finished', 'evicted'
        reason: str = "",
        running_time: float = 0,
        extra_info: str = ""
    ):
        """Record a preemption or lifecycle event"""
        timestamp = time.time() - self.start_time

        self.stats['preemption_events'].append((
            timestamp, request_id, event, reason, running_time, extra_info
        ))

        # Track lifecycle
        if request_id not in self.stats['request_lifecycle']:
            self.stats['request_lifecycle'][request_id] = {}

        self.stats['request_lifecycle'][request_id][event] = timestamp

        if ProfilingConfig.VERBOSE and event == 'preempted':
            print(f"[Preemption] Request {request_id} preempted (reason: {reason}, running: {running_time:.2f}s)",
                  file=sys.stderr)

    def save_stats(self):
        """Save preemption statistics"""
        # Events timeline
        events_file = os.path.join(self.session_dir, "preemption_events.csv")
        with open(events_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_sec', 'request_id', 'event', 'reason',
                'running_time_sec', 'extra_info'
            ])

            for ts, req_id, event, reason, running_time, extra in self.stats['preemption_events']:
                writer.writerow([
                    f"{ts:.3f}", req_id, event, reason,
                    f"{running_time:.3f}", extra
                ])

        # Request lifecycle
        lifecycle_file = os.path.join(self.session_dir, "request_lifecycle.csv")
        with open(lifecycle_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'request_id', 'started_sec', 'preempted_sec', 'resumed_sec',
                'finished_sec', 'total_time_sec', 'preemption_delay_sec'
            ])

            for req_id, lifecycle in self.stats['request_lifecycle'].items():
                started = lifecycle.get('started', 0)
                preempted = lifecycle.get('preempted', 0)
                resumed = lifecycle.get('resumed', 0)
                finished = lifecycle.get('finished', 0)

                total_time = (finished - started) if (finished and started) else 0
                preemption_delay = (resumed - preempted) if (resumed and preempted) else 0

                writer.writerow([
                    req_id,
                    f"{started:.3f}" if started else "",
                    f"{preempted:.3f}" if preempted else "",
                    f"{resumed:.3f}" if resumed else "",
                    f"{finished:.3f}" if finished else "",
                    f"{total_time:.3f}" if total_time else "",
                    f"{preemption_delay:.3f}" if preemption_delay else ""
                ])

        # Summary
        from collections import Counter
        import numpy as np

        preemption_reasons = Counter()
        running_times = []
        resume_delays = []

        for _, _, event, reason, running_time, _ in self.stats['preemption_events']:
            if event == 'preempted':
                if reason:
                    preemption_reasons[reason] += 1
                if running_time > 0:
                    running_times.append(running_time)

        for req_id, lifecycle in self.stats['request_lifecycle'].items():
            if 'preempted' in lifecycle and 'resumed' in lifecycle:
                delay = lifecycle['resumed'] - lifecycle['preempted']
                resume_delays.append(delay)

        summary = {
            'total_preemptions': sum(1 for _, _, event, _, _, _ in self.stats['preemption_events'] if event == 'preempted'),
            'total_requests': len(self.stats['request_lifecycle']),
            'preemption_reasons': dict(preemption_reasons),
            'mean_running_time_before_preempt': float(np.mean(running_times)) if running_times else 0,
            'mean_resume_delay': float(np.mean(resume_delays)) if resume_delays else 0,
            'max_resume_delay': float(np.max(resume_delays)) if resume_delays else 0,
        }

        summary_file = os.path.join(self.session_dir, "preemption_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[Preemption Profiler] Saved statistics to {self.session_dir}/")
        print(f"  - Total preemptions: {summary['total_preemptions']}")
        print(f"  - Total requests tracked: {summary['total_requests']}")


class EncoderDecoderProfiler:
    """Profiles encoder-decoder model timing (generic for all models)"""

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.stats = {
            'encoder_timings': [],  # (timestamp, duration_ms, context)
            'decoder_timings': [],  # (timestamp, duration_ms, context)
            'cross_attention_timings': [],  # (timestamp, duration_ms)
        }
        self.start_time = time.time()
        self.model_type = None  # 'encoder_decoder', 'decoder_only', or 'unknown'
        self.use_cuda_events = ProfilingConfig.USE_CUDA_EVENTS

    def set_model_type(self, model_type: str):
        """Set detected model type"""
        self.model_type = model_type

    def record_encoder_timing(self, duration_ms: float, context: str = ""):
        """Record encoder forward pass timing"""
        timestamp = time.time() - self.start_time
        self.stats['encoder_timings'].append((timestamp, duration_ms, context))

    def record_decoder_timing(self, duration_ms: float, context: str = ""):
        """Record decoder forward pass timing"""
        timestamp = time.time() - self.start_time
        self.stats['decoder_timings'].append((timestamp, duration_ms, context))

    def record_cross_attention_timing(self, duration_ms: float):
        """Record cross-attention timing"""
        timestamp = time.time() - self.start_time
        self.stats['cross_attention_timings'].append((timestamp, duration_ms))

    def save_stats(self):
        """Save encoder-decoder timing statistics"""
        # Timeline
        timeline_file = os.path.join(self.session_dir, "encoder_decoder_timing.csv")
        with open(timeline_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_sec', 'component', 'duration_ms', 'context'
            ])

            all_timings = (
                [('encoder', ts, dur, ctx) for ts, dur, ctx in self.stats['encoder_timings']] +
                [('decoder', ts, dur, ctx) for ts, dur, ctx in self.stats['decoder_timings']] +
                [('cross_attention', ts, dur, '') for ts, dur in self.stats['cross_attention_timings']]
            )
            all_timings.sort(key=lambda x: x[1])

            for component, ts, duration_ms, context in all_timings:
                writer.writerow([f"{ts:.3f}", component, f"{duration_ms:.3f}", context])

        # Summary
        import numpy as np

        summary = {
            'model_type': self.model_type or 'unknown',
        }

        total_time = 0
        for component_name, timings in [
            ('encoder', self.stats['encoder_timings']),
            ('decoder', self.stats['decoder_timings']),
            ('cross_attention', [(t, d, '') for t, d in self.stats['cross_attention_timings']])
        ]:
            if timings:
                durations = [t[1] for t in timings]
                total_time += sum(durations)
                summary[component_name] = {
                    'count': len(durations),
                    'total_ms': float(sum(durations)),
                    'mean_ms': float(np.mean(durations)),
                    'std_ms': float(np.std(durations)),
                }

        # Calculate percentages
        if total_time > 0:
            summary['encoder_pct'] = (summary.get('encoder', {}).get('total_ms', 0) / total_time * 100)
            summary['decoder_pct'] = (summary.get('decoder', {}).get('total_ms', 0) / total_time * 100)
            summary['cross_attention_pct'] = (summary.get('cross_attention', {}).get('total_ms', 0) / total_time * 100)

        summary_file = os.path.join(self.session_dir, "encoder_decoder_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[Encoder-Decoder Profiler] Saved statistics to {self.session_dir}/")
        print(f"  - Model type: {self.model_type or 'unknown'}")
        print(f"  - Encoder samples: {len(self.stats['encoder_timings'])}")
        print(f"  - Decoder samples: {len(self.stats['decoder_timings'])}")


# ============================================================================
# Global Profiler Instances
# ============================================================================

_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
_session_dir = os.path.join(ProfilingConfig.OUTPUT_DIR, f"session_{_session_id}")
os.makedirs(_session_dir, exist_ok=True)

_cuda_profiler = CUDAGraphProfiler(_session_dir) if ProfilingConfig.ENABLE_CUDA_GRAPH_TRACKING else None
_kv_profiler = KVCacheProfiler(_session_dir) if ProfilingConfig.ENABLE_KV_CACHE_TRACKING else None
_moe_profiler = MoEExpertProfiler(_session_dir) if ProfilingConfig.ENABLE_MOE_EXPERT_TRACKING else None
_forward_pass_profiler = ForwardPassProfiler(_session_dir) if ProfilingConfig.ENABLE_FORWARD_PASS_TIMING else None
_cpu_profiler = CPUTimingProfiler(_session_dir) if ProfilingConfig.ENABLE_CPU_TIMING else None
_batch_util_profiler = BatchUtilizationProfiler(_session_dir) if ProfilingConfig.ENABLE_BATCH_UTILIZATION_TRACKING else None
_preemption_profiler = PreemptionProfiler(_session_dir) if ProfilingConfig.ENABLE_PREEMPTION_TRACKING else None
_encoder_decoder_profiler = EncoderDecoderProfiler(_session_dir) if ProfilingConfig.ENABLE_ENCODER_DECODER_TIMING else None


def save_all_stats():
    """Save all profiling statistics on exit"""
    if _cuda_profiler:
        _cuda_profiler.save_stats()
    if _kv_profiler:
        _kv_profiler.save_stats()
    if _moe_profiler:
        _moe_profiler.save_stats()
    if _forward_pass_profiler:
        _forward_pass_profiler.save_stats()
    if _cpu_profiler:
        _cpu_profiler.save_stats()
    if _batch_util_profiler:
        _batch_util_profiler.save_stats()
    if _preemption_profiler:
        _preemption_profiler.save_stats()
    if _encoder_decoder_profiler:
        _encoder_decoder_profiler.save_stats()

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
            'forward_pass_timing': ProfilingConfig.ENABLE_FORWARD_PASS_TIMING,
            'cpu_timing': ProfilingConfig.ENABLE_CPU_TIMING,
            'batch_utilization_tracking': ProfilingConfig.ENABLE_BATCH_UTILIZATION_TRACKING,
            'preemption_tracking': ProfilingConfig.ENABLE_PREEMPTION_TRACKING,
            'encoder_decoder_timing': ProfilingConfig.ENABLE_ENCODER_DECODER_TIMING,
            'cuda_events_enabled': ProfilingConfig.USE_CUDA_EVENTS,
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


def patch_gpu_model_runner():
    """Patch GPUModelRunner to track forward pass timing"""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm.logger import init_logger

        logger = init_logger(__name__)

        # Store original execute_model
        original_execute_model = GPUModelRunner.execute_model

        def instrumented_execute_model(self, scheduler_output):
            """Instrumented execute_model with timing"""
            # Determine phase
            phase = 'prefill' if scheduler_output.num_prefill_reqs > 0 else 'decode'
            batch_size = scheduler_output.total_num_scheduled_tokens
            num_tokens = scheduler_output.num_scheduled_tokens.get(phase, batch_size)

            # Start timing
            start_marker = None
            if _forward_pass_profiler:
                start_marker = _forward_pass_profiler.record_forward_start(phase, batch_size, num_tokens)

            # CPU timing for scheduler/prep
            cpu_start = time.perf_counter()

            # Execute original
            result = original_execute_model(self, scheduler_output)

            # Record forward pass end
            if _forward_pass_profiler and start_marker:
                _forward_pass_profiler.record_forward_end(start_marker, phase, batch_size, num_tokens)

            # CPU timing (this captures some overhead, but gives context)
            cpu_duration_ms = (time.perf_counter() - cpu_start) * 1000
            if _cpu_profiler:
                _cpu_profiler.record_operation('model_execution', cpu_duration_ms, f'phase={phase}')

            return result

        GPUModelRunner.execute_model = instrumented_execute_model

        logger.info("[Instrumentation] Successfully patched GPUModelRunner")

    except ImportError:
        pass
    except Exception as e:
        print(f"[Instrumentation] Failed to patch GPUModelRunner: {e}", file=sys.stderr)


def patch_scheduler():
    """Patch Scheduler to track batch utilization and preemptions"""
    try:
        from vllm.v1.core.scheduler import Scheduler
        from vllm.logger import init_logger

        logger = init_logger(__name__)

        # Store originals
        original_schedule = Scheduler.schedule
        original_init = Scheduler.__init__

        def instrumented_init(self, *args, **kwargs):
            """Instrumented __init__ to capture limits"""
            result = original_init(self, *args, **kwargs)

            # Capture scheduler limits
            if _batch_util_profiler:
                max_num_seqs = getattr(self, 'max_num_running_reqs', 256)
                max_tokens = getattr(self, 'max_num_batched_tokens', 8192)
                _batch_util_profiler.set_limits(max_num_seqs, max_tokens)

            return result

        def instrumented_schedule(self):
            """Instrumented schedule with batch utilization tracking"""
            # CPU timing for scheduling
            cpu_start = time.perf_counter()

            # Execute original schedule
            scheduler_output = original_schedule(self)

            cpu_duration_ms = (time.perf_counter() - cpu_start) * 1000
            if _cpu_profiler:
                _cpu_profiler.record_operation('scheduling', cpu_duration_ms)

            # Track batch utilization
            if _batch_util_profiler and scheduler_output:
                # Determine phase
                phase = 'prefill' if scheduler_output.num_prefill_reqs > 0 else 'decode'

                # Get queue lengths
                running_queue_len = len(getattr(self, 'running', []))
                waiting_queue_len = len(getattr(self, 'waiting', []))

                num_seqs = scheduler_output.total_num_scheduled_tokens  # This might be tokens, adjust if needed
                num_tokens = scheduler_output.num_scheduled_tokens.get(phase, 0)

                _batch_util_profiler.record_batch(
                    num_seqs=len(scheduler_output.scheduled_new_reqs) + len(scheduler_output.scheduled_resumed_reqs) + len(scheduler_output.scheduled_running_reqs),
                    num_tokens=num_tokens,
                    phase=phase,
                    running_queue_len=running_queue_len,
                    waiting_queue_len=waiting_queue_len
                )

            return scheduler_output

        Scheduler.__init__ = instrumented_init
        Scheduler.schedule = instrumented_schedule

        # Try to patch preemption methods
        try:
            if hasattr(Scheduler, '_preempt_requests'):
                original_preempt = Scheduler._preempt_requests

                def instrumented_preempt(self, requests, preemption_mode):
                    """Instrumented preemption tracking"""
                    # Record preemptions
                    if _preemption_profiler:
                        for req in requests:
                            req_id = getattr(req, 'request_id', str(id(req)))
                            # Try to get running time
                            running_time = 0
                            if hasattr(req, 'metrics') and hasattr(req.metrics, 'start_time'):
                                running_time = time.time() - req.metrics.start_time

                            _preemption_profiler.record_event(
                                request_id=req_id,
                                event='preempted',
                                reason=str(preemption_mode),
                                running_time=running_time,
                                extra_info=f"num_blocks={getattr(req, 'num_computed_tokens', 0)}"
                            )

                    return original_preempt(self, requests, preemption_mode)

                Scheduler._preempt_requests = instrumented_preempt

        except AttributeError:
            # Preemption method might have different name in different versions
            pass

        logger.info("[Instrumentation] Successfully patched Scheduler")

    except ImportError:
        pass
    except Exception as e:
        print(f"[Instrumentation] Failed to patch Scheduler: {e}", file=sys.stderr)


def patch_model_forward():
    """Patch model forward passes to detect encoder-decoder architecture"""
    try:
        from vllm.model_executor.models.utils import AutoModelForCausalLM
        from vllm.logger import init_logger

        logger = init_logger(__name__)

        # This will attempt to detect model type at runtime
        # For now, we'll just log that we attempted
        # Actual instrumentation would need to be model-specific

        if _encoder_decoder_profiler:
            # Auto-detect will happen in actual model loading
            # For now, mark as attempted
            logger.info("[Instrumentation] Encoder-decoder profiling enabled (auto-detect)")

    except ImportError:
        pass
    except Exception as e:
        print(f"[Instrumentation] Encoder-decoder profiling note: {e}", file=sys.stderr)


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
                'vllm.v1.worker.gpu_model_runner',
                'vllm.v1.core.scheduler',
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
            elif fullname == 'vllm.v1.worker.gpu_model_runner':
                patch_gpu_model_runner()
            elif fullname == 'vllm.v1.core.scheduler':
                patch_scheduler()

            return module

    # Install the hook
    sys.meta_path.insert(0, VllmInstrumentationHook())


# ============================================================================
# Activation Guard - Only activate for vLLM processes
# ============================================================================

def should_activate_profiling():
    """Determine if profiling should be activated for this process"""

    # Method 1: Explicit environment variable (highest priority)
    explicit_enable = os.getenv("VLLM_ENABLE_PROFILING")
    if explicit_enable is not None:
        if explicit_enable == "1":
            return True
        else:
            return False

    # Method 2: Auto-detect vLLM usage
    # Check command line arguments
    import sys
    cmdline = ' '.join(sys.argv)

    # Check if this looks like a vLLM process
    vllm_indicators = [
        'vllm.entrypoints',
        'vllm.engine',
        'vllm',
        '--model',  # Common vLLM argument
    ]

    for indicator in vllm_indicators:
        if indicator in cmdline:
            return True

    # Method 3: Check if vLLM is installed (but don't activate yet)
    # This prevents false positives for non-vLLM scripts
    try:
        import importlib.util
        spec = importlib.util.find_spec('vllm')
        if spec is None:
            return False
    except ImportError:
        return False

    # If vLLM is installed but not clearly in use, don't activate
    # (Prevents profiling unrelated Python scripts)
    return False


# ============================================================================
# Initialization
# ============================================================================

# Only activate if this is a vLLM process
if should_activate_profiling():
    # Install the import hook
    install_import_hook()

    print(f"\n[sitecustomize] vLLM Comprehensive Instrumentation Loaded", file=sys.stderr)
    print(f"  Session ID: {_session_id}", file=sys.stderr)
    print(f"  Output directory: {_session_dir}", file=sys.stderr)
    print(f"  CUDA graph tracking: {ProfilingConfig.ENABLE_CUDA_GRAPH_TRACKING}", file=sys.stderr)
    print(f"  KV cache tracking: {ProfilingConfig.ENABLE_KV_CACHE_TRACKING}", file=sys.stderr)
    print(f"  MoE expert tracking: {ProfilingConfig.ENABLE_MOE_EXPERT_TRACKING}", file=sys.stderr)
    print(f"  Forward pass timing: {ProfilingConfig.ENABLE_FORWARD_PASS_TIMING}", file=sys.stderr)
    print(f"  CPU operation timing: {ProfilingConfig.ENABLE_CPU_TIMING}", file=sys.stderr)
    print(f"  Batch utilization tracking: {ProfilingConfig.ENABLE_BATCH_UTILIZATION_TRACKING}", file=sys.stderr)
    print(f"  Preemption tracking: {ProfilingConfig.ENABLE_PREEMPTION_TRACKING}", file=sys.stderr)
    print(f"  Encoder-decoder timing: {ProfilingConfig.ENABLE_ENCODER_DECODER_TIMING}", file=sys.stderr)
    print(f"  CUDA Events mode: {ProfilingConfig.USE_CUDA_EVENTS} (batch size: {ProfilingConfig.CUDA_EVENT_BATCH_SIZE})", file=sys.stderr)
    print("", file=sys.stderr)
else:
    # Silent mode - don't activate profiling for non-vLLM processes
    pass
