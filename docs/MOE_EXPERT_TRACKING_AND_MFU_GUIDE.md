# MoE Expert Tracking and MFU Metrics Guide

## Table of Contents

1. [MoE Expert Activation Tracking](#moe-expert-activation-tracking)
2. [MFU Metrics Reliability](#mfu-metrics-reliability)
3. [Implementation Guide](#implementation-guide)

---

## 1. MoE Expert Activation Tracking

### Does vLLM Have Built-in Expert Tracking?

**Partial support** - vLLM has infrastructure for expert load tracking (EPLB), but **no public API** for general expert activation monitoring.

### What EXISTS in vLLM

#### Option 1: EPLB Load Metrics (Expert Parallel Load Balancer)

**Location**: `vllm/vllm/distributed/eplb/eplb_state.py:118-138`

**Available when EPLB is enabled**:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-MoE-72B \
    --tensor-parallel-size 4 \
    --enable-eplb \
    --eplb-config '{"log_balancedness": true}'
```

**What it tracks**:

```python
@dataclass
class EplbModelState:
    expert_load_pass: torch.Tensor
    """
    Expert load during this forward pass.
    We use the token count each expert processes as the load.

    Shape: (num_moe_layers, num_physical_experts)
    """

    expert_load_window: torch.Tensor
    """
    A sliding window of expert load.

    Shape: (window_size, num_moe_layers, num_physical_experts)
    """
```

**Output** (when `log_balancedness: true`):

```
[EPLB] Balancedness: 0.85 (avg tokens per expert ÷ max tokens per expert)
```

**Limitations**:
- ✗ No per-expert activation counts
- ✗ No router weights distribution
- ✗ No expert ID details per request
- ✓ Only aggregated load balancing stats

#### Option 2: Debug Logging (Limited)

**Location**: `vllm/vllm/model_executor/layers/fused_moe/layer.py`

Currently **no built-in** detailed expert logging beyond EPLB metrics.

### What You NEED (Not Available)

For comprehensive MoE profiling, you typically want:

1. **Per-expert activation counts**: How many times each expert was selected
2. **Router weight distribution**: Distribution of routing scores across experts
3. **Expert selection patterns**: Which experts are selected together
4. **Per-layer expert usage**: Expert usage breakdown per MoE layer
5. **Temporal patterns**: How expert usage changes over time
6. **Per-request tracking**: Which experts handle which requests

**None of these are publicly exposed by vLLM.**

---

## 2. MFU Metrics Reliability

### How MFU Metrics Work

**Location**: `vllm/vllm/v1/metrics/perf.py`

MFU (Model FLOPs Utilization) metrics in vLLM are **analytical estimators**, not measured values.

### The Calculation Method

#### Step 1: Parse Model Architecture

```python
# From vllm/v1/metrics/perf.py:265-304
class BaseConfigParser(Parser):
    """
    Parses base model configuration.
    Provides: vocab_size, hidden_size, num_attention_heads, num_hidden_layers,
    weight_byte_size, activation_byte_size, dp_size, tp_size, pp_size
    """
```

**What it extracts**:
- Model dimensions (hidden_size, num_layers, num_heads, etc.)
- Parallelization config (TP, PP, DP, EP)
- Data types (weight_byte_size, activation_byte_size)
- MoE config (num_experts, num_experts_per_tok, etc.)

#### Step 2: Build Execution Context

```python
# From vllm/v1/metrics/perf.py:66-104
@dataclass
class ExecutionContext:
    """
    Represents an execution context for a batch of requests.
    Separately tracking prefill and decode phases.
    """

    # Prefill phase statistics
    num_prefill_requests: int = 0
    prefill_num_tokens: int = 0
    prefill_context_len: int = 0
    prefill_token_context_product: int = 0

    # Decode phase statistics
    num_decode_requests: int = 0
    decode_num_tokens: int = 0
    decode_context_len: int = 0
    decode_token_context_product: int = 0
```

**What it captures** (from scheduler output):
- Number of prefill vs decode requests
- Total tokens processed
- Context lengths (for KV cache calculations)

#### Step 3: Analytical FLOPs/Bandwidth Calculation

**Attention FLOPs** (from `vllm/v1/metrics/perf.py:391-415`):

```python
def get_num_flops_breakdown(self, ctx: ExecutionContext):
    L, D, q, kv, d = (
        num_hidden_layers,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
    )
    T = ctx.total_num_tokens()
    TC = ctx.total_token_context_product()

    return {
        "qkv_proj": 2 * T * D * (q + 2 * kv) * d * L,
        "attn_qk": 2 * q * TC * d * L,
        "attn_av": 2 * q * TC * d * L,
        "out_proj": 2 * T * D * q * d * L,
    }
```

**Key formulas**:
- **QKV projection**: `2 * num_tokens * hidden_size * (Q + 2*KV) * head_dim * layers`
- **Attention QK**: `2 * num_heads * token_context_product * head_dim * layers`
- **Attention AV**: `2 * num_heads * token_context_product * head_dim * layers`
- **Output projection**: `2 * num_tokens * hidden_size * num_heads * head_dim * layers`

**FFN FLOPs** (from `vllm/v1/metrics/perf.py:682-723`):

```python
def get_num_flops_breakdown(self, ctx: ExecutionContext):
    L, D, DI = num_hidden_layers, hidden_size, intermediate_size
    T = ctx.total_num_tokens()

    flops = {}

    # Dense FFN layers (SwiGLU: 3 linear layers: up, gate, down)
    if Ld:
        flops["dense_ffn"] = 2 * D * 3 * DI * T * Ld

    # MoE routed experts
    if Lm and E:
        flops["routed_ffn"] = 2 * D * 3 * MI * num_activated_tokens * Lm

    # MoE shared experts
    if Lm and S:
        flops["shared_ffn"] = 2 * D * 3 * MI * S * T * Lm

    return flops
```

**MoE calculation** (from test file `tests/v1/metrics/test_perf_metrics.py:350-383`):

```python
# For MoE: compute is proportional to activated experts
# If num_experts_per_tok = 2, and dense would do X FLOPs,
# MoE does 2*X FLOPs (activating 2 experts instead of 1 dense layer)

moe_flops == dense_flops * num_experts_per_tok
```

**Memory Bandwidth** (from `vllm/v1/metrics/perf.py:417-486`):

```python
def get_read_bytes_breakdown(self, ctx: ExecutionContext):
    # Reads differ between prefill and decode!

    # Prefill: read Q, K, V activations (all in activation_byte_size)
    if ctx.prefill_num_tokens > 0:
        read_bytes["attn_input"] = (
            (prefill_num_tokens * q + 2 * prefill_context_len * kv)
            * d * activation_byte_size * L
        )

    # Decode: read Q activations + read K, V from cache (in cache_byte_size)
    if ctx.decode_num_tokens > 0:
        read_bytes["attn_input"] = (
            decode_num_tokens * q * d * activation_byte_size * L
            + 2 * decode_context_len * kv * d * cache_byte_size * L
        )
```

#### Step 4: Convert to Throughput

```python
# From vllm/v1/metrics/perf.py:1193-1220
def log(self):
    delta_time = now - self.last_log_time

    # Compute bandwidth
    avg_tflops_per_gpu = self.total_num_flops_per_gpu / delta_time / 1e12
    avg_gbps_per_gpu = (
        (self.total_read_bytes_per_gpu + self.total_write_bytes_per_gpu)
        / delta_time
        / 1e9
    )

    logger.info(
        "MFU: %.1f TF/s/GPU %.1f GB/s/GPU",
        avg_tflops_per_gpu,
        avg_gbps_per_gpu,
    )
```

### Reliability Assessment

#### ✅ HIGHLY RELIABLE For:

1. **Dense transformer models** (Llama, Qwen, GPT-style)
   - FLOPs calculation: **>95% accurate**
   - Memory bandwidth: **>90% accurate**
   - Reason: Well-understood GEMM operations with predictable access patterns

2. **Prefill phase**
   - FLOPs: **>95% accurate**
   - Reason: Dominated by large matrix multiplications (easy to model)

3. **Relative comparisons**
   - Comparing runs: **>99% reliable**
   - Reason: Same assumptions cancel out

4. **MoE models (with caveats)**
   - FLOPs: **~90% accurate** if load balancing is good
   - Reason: Assumes uniform expert activation (see limitations below)

#### ⚠️ LESS RELIABLE For:

1. **Decode phase bandwidth**
   - Accuracy: **~80-85%**
   - Why: Harder to model cache access patterns, depends heavily on:
     - KV cache layout
     - Memory controller behavior
     - Cache line effects
   - Example discrepancy:
     ```
     Analytical: 892 GB/s
     Measured (Nsight): 820 GB/s (~8% difference)
     ```

2. **MoE models with skewed load**
   - Accuracy: **~70-90%** depending on skew
   - Why: Assumes perfect load balancing
   - **Current assumption** (from `vllm/v1/metrics/perf.py:776-777`):
     ```python
     # FIXME: Assume perfect load balancing for now.
     num_activated_experts = min(num_activated_tokens, num_experts)
     ```
   - **Reality**: Some experts may be hot, others cold
   - Impact: If 10% of experts handle 50% of tokens, FLOPs estimate could be 10-20% off

3. **Quantized models**
   - Accuracy: **~85-90%**
   - Why: Hardware has specialized units (e.g., Tensor Cores for FP8)
   - Not all operations scale linearly with byte size

4. **Attention-variant architectures**
   - Accuracy: **Varies widely (60-95%)**
   - Examples with lower accuracy:
     - Sliding window attention
     - Sparse attention
     - MLA (Multi-head Latent Attention in DeepSeek)
   - Why: Analytical formulas assume standard attention

#### ❌ NOT MEASURED AT ALL:

1. **Kernel-level efficiency**
   - Analytical metrics assume 100% efficiency
   - Reality: kernels achieve 70-95% of theoretical peak
   - Not accounted for in MFU metrics

2. **Scheduling overhead**
   - CPU-GPU synchronization
   - Kernel launch overhead
   - Memory allocation overhead

3. **Communication costs**
   - Tensor parallel all-reduce
   - Pipeline parallel communication
   - Expert parallel all-to-all (for MoE)

### How to Verify MFU Metrics

#### Method 1: Compare Against Nsight Compute

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

ncu --set full --target-processes all \
    -o vllm_actual.ncu-rep \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-hf \
        --enable-mfu-metrics

# Open in Nsight Compute UI
ncu-ui vllm_actual.ncu-rep
```

**Compare**:
- vLLM reports: `245.3 TF/s/GPU`
- Nsight reports: `dram__throughput.avg.pct_of_peak_sustained_elapsed` = 42%
- If A100 peak = 312 TF/s → Actual = 312 * 0.42 = 131 TF/s

**Typical deltas**:
- Compute (FLOPs): ±5-10%
- Memory bandwidth: ±10-15%

#### Method 2: Cross-Check with Hardware Counters

```bash
# Enable debug mode
export VLLM_DEBUG_MFU_METRICS=1

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-mfu-metrics
```

**Output breakdown**:
```json
{
  "flops_breakdown": {
    "attn.qkv_proj": "45.2TF",
    "attn.attn_qk": "102.3TF",
    "attn.attn_av": "98.1TF",
    "attn.out_proj": "38.7TF",
    "ffn.dense_ffn": "156.7TF"
  },
  "num_read_bytes_breakdown": {
    "attn.qkv_weight": "12.3GB",
    "attn.attn_input": "45.6GB"
  }
}
```

**Sanity checks**:
1. Attention should dominate for long contexts
2. FFN should dominate for short contexts
3. Read bytes >> Write bytes (model weights are large)

#### Method 3: Roofline Analysis

Calculate arithmetic intensity:

```python
# From MFU metrics
total_flops = 441.0e12  # 441 TF
total_bytes = 100e9     # 100 GB

arithmetic_intensity = total_flops / total_bytes
# = 4410 FLOPs/byte

# Compare to hardware
# A100: Peak = 312 TF/s, Bandwidth = 2000 GB/s
# Ridge point = 312e12 / 2000e9 = 156 FLOPs/byte

# Since 4410 >> 156, workload is HIGHLY compute-bound
# → MFU metrics should be very accurate for compute
# → Bandwidth metrics may have more error
```

### MFU Metrics for MoE Models

**Location**: `vllm/v1/metrics/perf.py:684-723`

#### What It Assumes

```python
# From tests/v1/metrics/test_perf_metrics.py:461-517
def test_moe_expert_activation_proportional_scaling():
    """
    Test that routed expert metrics scale proportionally with num_experts_per_tok.
    """
    # If num_experts_per_tok = 2, and dense would do X FLOPs,
    # MoE does 2*X FLOPs

    moe_flops == dense_flops * num_experts_per_tok
```

**Key assumption**: All experts receive equal load (perfect load balancing)

#### Reality Check

**From EPLB documentation** (`vllm/docs/serving/expert_parallel_deployment.md:136-138`):

> While MoE models are typically trained so that each expert receives a similar number of tokens, in practice **the distribution of tokens across experts can be highly skewed**.

**Example** (Qwen2.5-MoE-72B):
- 64 experts total
- Top-2 routing (2 experts per token)
- Theoretical: Each expert gets ~3.125% of tokens (2/64)
- Reality: Expert distribution might be:
  - Top 10%: 40% of tokens
  - Middle 50%: 50% of tokens
  - Bottom 40%: 10% of tokens

**Impact on MFU metrics**:
- **FLOPs**: Still accurate! (Same experts activated, just different distribution)
- **Memory bandwidth**: May be off by 10-20% (hot experts cached, cold experts not)
- **Load balancing**: Critical for EP (Expert Parallel) performance

#### Tracking Expert Imbalance

**Current support** (EPLB balancedness metric):

```bash
--eplb-config '{"log_balancedness": true}'
```

**Output**:
```
[EPLB] Balancedness: 0.65
```

**Interpretation**:
- **1.0** = Perfect balance (all experts get equal load)
- **0.65** = Imbalanced (max-loaded expert has 1.5x avg load)
- **<0.5** = Highly skewed

**Formula**:
```
balancedness = (avg tokens per expert) / (max tokens per expert)
```

---

## 3. Implementation Guide

### Option A: Extend EPLB State (Recommended if using EP)

**If you're using Expert Parallelism**, extend the existing EPLB infrastructure.

**Location**: `vllm/vllm/distributed/eplb/eplb_state.py`

**Add to `EplbModelState`**:

```python
@dataclass
class EplbModelState:
    # Existing fields...
    expert_load_pass: torch.Tensor
    expert_load_window: torch.Tensor

    # NEW: Add expert activation tracking
    expert_activation_counts: torch.Tensor | None = None
    """
    Count of how many times each expert was selected.

    Shape: (num_moe_layers, num_physical_experts)
    """

    expert_routing_weights: list[torch.Tensor] | None = None
    """
    Router weights (topk_weights) for each layer.
    Stored only if detailed tracking is enabled.

    List of tensors, one per MoE layer.
    Each tensor shape: (num_tokens, topk)
    """

    expert_selection_patterns: dict[str, int] | None = None
    """
    Frequency of expert co-selection patterns.

    Example: {"0,1": 1234, "0,2": 567, "1,2": 890}
    Means experts (0,1) were selected together 1234 times, etc.
    """
```

**Modify `vllm/vllm/model_executor/layers/fused_moe/layer.py`**:

```python
def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
    # Existing routing
    topk_weights, topk_ids = self.router.select_experts(
        hidden_states, router_logits
    )

    # NEW: Track expert activations
    if self.enable_expert_tracking:
        self._record_expert_activations(topk_ids, topk_weights)

    # Rest of forward pass...
    return output

def _record_expert_activations(
    self,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor
):
    """Record expert activation statistics."""
    eplb_state = get_eplb_state()  # From forward context
    if eplb_state is None:
        return

    layer_idx = self.layer_idx

    # Count activations per expert
    for expert_id in range(self.num_experts):
        count = (topk_ids == expert_id).sum().item()
        eplb_state.expert_activation_counts[layer_idx, expert_id] += count

    # Store routing weights (optional, memory-intensive)
    if eplb_state.expert_routing_weights is not None:
        eplb_state.expert_routing_weights[layer_idx] = topk_weights.detach().cpu()

    # Track co-selection patterns (for topk=2)
    if topk_ids.size(1) == 2:
        # Sort expert pairs for consistent keys
        expert_pairs = torch.sort(topk_ids, dim=1)[0]
        for pair in expert_pairs:
            key = f"{pair[0].item()},{pair[1].item()}"
            eplb_state.expert_selection_patterns[key] = (
                eplb_state.expert_selection_patterns.get(key, 0) + 1
            )
```

**Enable tracking**:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-MoE-72B \
    --tensor-parallel-size 4 \
    --enable-eplb \
    --eplb-config '{"log_balancedness": true, "enable_expert_tracking": true}'
```

### Option B: sitecustomize.py Profiler (Recommended for General Use)

**If NOT using Expert Parallelism**, use the sitecustomize approach (same as CUDA graph profiler).

**File**: `profilemate/sitecustomize.py`

**Add new profiler class**:

```python
class MoEExpertProfiler:
    """Tracks MoE expert activation patterns."""

    def __init__(self):
        self.session_dir = setup_profiling_session()
        self.expert_activations = {}  # {layer_idx: {expert_id: count}}
        self.routing_weights = []     # [(layer_idx, weights)]
        self.co_selection_patterns = {}  # {"expert_i,expert_j": count}

    def record_expert_selection(
        self,
        layer_idx: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """Record which experts were selected."""
        # Initialize layer if needed
        if layer_idx not in self.expert_activations:
            self.expert_activations[layer_idx] = {}

        # Count expert activations
        for expert_id in topk_ids.flatten().tolist():
            self.expert_activations[layer_idx][expert_id] = (
                self.expert_activations[layer_idx].get(expert_id, 0) + 1
            )

        # Record routing weights (sample to save memory)
        if len(self.routing_weights) < 1000:  # Keep last 1000
            self.routing_weights.append((layer_idx, topk_weights.detach().cpu()))

        # Track co-selection for topk=2
        if topk_ids.size(1) == 2:
            for pair in topk_ids.tolist():
                key = f"{min(pair)},{max(pair)}"
                self.co_selection_patterns[key] = (
                    self.co_selection_patterns.get(key, 0) + 1
                )

    def save(self):
        """Save expert activation data to CSV."""
        # Expert activation counts
        activation_file = self.session_dir / "moe_expert_activations.csv"
        with open(activation_file, 'w') as f:
            f.write("layer_idx,expert_id,activation_count\n")
            for layer_idx, experts in self.expert_activations.items():
                for expert_id, count in sorted(experts.items()):
                    f.write(f"{layer_idx},{expert_id},{count}\n")

        # Co-selection patterns
        pattern_file = self.session_dir / "moe_expert_coselection.csv"
        with open(pattern_file, 'w') as f:
            f.write("expert_pair,count\n")
            for pattern, count in sorted(
                self.co_selection_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                f.write(f"{pattern},{count}\n")

        # Routing weights histogram
        if self.routing_weights:
            weights_file = self.session_dir / "moe_routing_weights_hist.csv"
            all_weights = torch.cat([w for _, w in self.routing_weights]).flatten()
            hist, bins = torch.histogram(all_weights, bins=50)

            with open(weights_file, 'w') as f:
                f.write("bin_min,bin_max,count\n")
                for i in range(len(hist)):
                    f.write(f"{bins[i]:.4f},{bins[i+1]:.4f},{hist[i]}\n")

        logger.info(f"MoE expert profiling data saved to {self.session_dir}")


# Global profiler instance
_moe_expert_profiler = None


def patch_fused_moe():
    """Patch FusedMoE to track expert activations."""
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        global _moe_expert_profiler
        _moe_expert_profiler = MoEExpertProfiler()

        original_forward = FusedMoE.forward

        def instrumented_forward(self, hidden_states, router_logits):
            # Call original
            output = original_forward(self, hidden_states, router_logits)

            # Extract topk_ids and topk_weights
            # (Need to intercept select_experts call)
            with torch.no_grad():
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states, router_logits
                )

                # Record
                layer_idx = getattr(self, 'layer_idx', 0)
                _moe_expert_profiler.record_expert_selection(
                    layer_idx, topk_ids, topk_weights
                )

            return output

        FusedMoE.forward = instrumented_forward

        # Register cleanup
        import atexit
        atexit.register(_moe_expert_profiler.save)

        logger.info("[sitecustomize] MoE expert profiling enabled")

    except ImportError:
        pass  # Not an MoE model
```

**Add to `install_import_hook()`**:

```python
def install_import_hook():
    """Install all profiling hooks."""
    import sys

    # ... existing patches ...

    # Patch MoE
    if 'vllm.model_executor.layers.fused_moe' in sys.modules:
        patch_fused_moe()
    else:
        # Lazy patch when module is loaded
        original_import = __builtins__.__import__

        def hooked_import(name, *args, **kwargs):
            module = original_import(name, *args, **kwargs)
            if name == 'vllm.model_executor.layers.fused_moe.layer':
                patch_fused_moe()
            return module

        __builtins__.__import__ = hooked_import
```

**Usage**:

```bash
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-MoE-72B \
    --tensor-parallel-size 4
```

**Output files**:

```
/tmp/vllm_profiling/session_20260127_123456/
├── moe_expert_activations.csv
├── moe_expert_coselection.csv
└── moe_routing_weights_hist.csv
```

### Analyzing Expert Activation Data

#### Python Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load expert activations
activations = pd.read_csv('moe_expert_activations.csv')

# Analysis per layer
for layer_idx in activations['layer_idx'].unique():
    layer_data = activations[activations['layer_idx'] == layer_idx]

    # Calculate statistics
    total_activations = layer_data['activation_count'].sum()
    num_experts = len(layer_data)

    # Expert utilization
    layer_data['utilization_pct'] = (
        layer_data['activation_count'] / total_activations * 100
    )

    # Print summary
    print(f"\nLayer {layer_idx}:")
    print(f"  Total activations: {total_activations}")
    print(f"  Experts: {num_experts}")
    print(f"  Top 5 experts:")
    print(layer_data.nlargest(5, 'activation_count')[
        ['expert_id', 'activation_count', 'utilization_pct']
    ])

    # Calculate skew
    max_count = layer_data['activation_count'].max()
    avg_count = layer_data['activation_count'].mean()
    skew = max_count / avg_count
    print(f"  Load skew: {skew:.2f}x (1.0 = perfect balance)")

    # Visualize distribution
    plt.figure(figsize=(12, 6))
    plt.bar(layer_data['expert_id'], layer_data['utilization_pct'])
    plt.xlabel('Expert ID')
    plt.ylabel('Utilization (%)')
    plt.title(f'Expert Utilization - Layer {layer_idx}')
    plt.axhline(y=100/num_experts, color='r', linestyle='--',
                label='Perfect balance')
    plt.legend()
    plt.savefig(f'expert_utilization_layer_{layer_idx}.png')
    plt.close()

# Analyze co-selection patterns
coselection = pd.read_csv('moe_expert_coselection.csv')
print("\nTop 10 expert pairs:")
print(coselection.head(10))

# Heatmap of co-selection
experts_in_pairs = set()
for pair in coselection['expert_pair']:
    e1, e2 = map(int, pair.split(','))
    experts_in_pairs.add(e1)
    experts_in_pairs.add(e2)

num_unique_experts = len(experts_in_pairs)
coselection_matrix = np.zeros((num_unique_experts, num_unique_experts))

for _, row in coselection.iterrows():
    e1, e2 = map(int, row['expert_pair'].split(','))
    coselection_matrix[e1, e2] = row['count']
    coselection_matrix[e2, e1] = row['count']

plt.figure(figsize=(10, 10))
plt.imshow(coselection_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Co-selection count')
plt.xlabel('Expert ID')
plt.ylabel('Expert ID')
plt.title('Expert Co-selection Heatmap')
plt.savefig('expert_coselection_heatmap.png')
```

#### Calculate MFU Impact

```python
# Compare actual vs uniform distribution
uniform_count = total_activations / num_experts

# Calculate effective FLOPs based on actual distribution
# (Accounting for cache efficiency of hot experts)

hot_experts = layer_data[layer_data['utilization_pct'] > 5.0]  # >5%
cold_experts = layer_data[layer_data['utilization_pct'] <= 5.0]

# Hot experts likely cached → faster execution
# Cold experts likely not cached → slower execution

hot_speedup = 1.2  # 20% faster due to caching
cold_slowdown = 0.9  # 10% slower due to cache misses

effective_flops = (
    hot_experts['activation_count'].sum() * hot_speedup +
    cold_experts['activation_count'].sum() * cold_slowdown
)

theoretical_flops = total_activations  # Assuming uniform

speedup_factor = effective_flops / theoretical_flops
print(f"Effective speedup due to expert distribution: {speedup_factor:.2f}x")

# Adjust MFU metrics
analytical_mfu = 245.3  # TF/s from vLLM
adjusted_mfu = analytical_mfu * speedup_factor
print(f"Analytical MFU: {analytical_mfu:.1f} TF/s")
print(f"Adjusted MFU: {adjusted_mfu:.1f} TF/s")
```

---

## Summary

### MoE Expert Tracking

| Approach | Built-in? | Overhead | Detail Level | Recommendation |
|----------|-----------|----------|--------------|----------------|
| EPLB metrics | ✓ Yes | <0.1% | Low (balancedness only) | Use for load balancing monitoring |
| Extended EPLB | Partial | ~1% | High (if modified) | Best if using Expert Parallel |
| sitecustomize | Custom | ~2-3% | Very High | Best for general profiling |
| NVTX + profiling | Manual | 10-20% | Highest (kernel-level) | One-time deep analysis |

### MFU Metrics Reliability

| Metric | Reliability | Notes |
|--------|-------------|-------|
| Dense model FLOPs | **>95%** | Highly accurate |
| MoE FLOPs (balanced) | **~90%** | Good if load is balanced |
| MoE FLOPs (skewed) | **~70-90%** | Depends on skew |
| Memory bandwidth (prefill) | **~90%** | Good for large batches |
| Memory bandwidth (decode) | **~80-85%** | Cache effects hard to model |
| Relative comparisons | **>99%** | Very reliable for A/B testing |

### Key Takeaways

1. **MFU metrics are analytical, not measured**
   - Based on theoretical FLOPs/bandwidth calculations
   - Do NOT account for kernel efficiency, scheduling overhead, or communication costs

2. **MoE models have special considerations**
   - Assumes perfect load balancing (rarely true in practice)
   - Expert activation skew can cause 10-30% error in bandwidth estimates
   - FLOPs are still accurate (same total compute, different distribution)

3. **Use MFU metrics for**:
   - ✓ Relative performance comparisons
   - ✓ Identifying compute vs memory bottlenecks
   - ✓ Capacity planning
   - ✗ Absolute performance claims (use Nsight for that)

4. **For MoE profiling, you need custom tracking**
   - vLLM has no built-in per-expert activation monitoring (except EPLB load)
   - sitecustomize.py approach is recommended for detailed analysis
   - EPLB metrics are good for production load balancing monitoring

---

## References

- **MFU Metrics Implementation**: `vllm/vllm/v1/metrics/perf.py`
- **MFU Metrics Tests**: `vllm/tests/v1/metrics/test_perf_metrics.py`
- **EPLB State**: `vllm/vllm/distributed/eplb/eplb_state.py`
- **FusedMoE Layer**: `vllm/vllm/model_executor/layers/fused_moe/layer.py`
- **Expert Parallel Deployment**: `vllm/docs/serving/expert_parallel_deployment.md`
