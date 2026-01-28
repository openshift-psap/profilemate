# ProfileMate Automation Scripts

Automated profiling scripts for vLLM using Nsight Systems and Nsight Compute.

## Scripts Overview

### Main Automation

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| **`profile_vllm.sh`** | All-in-one profiling pipeline | 5-60 min | Complete profiling report |
| `send_test_requests.py` | Send test traffic to vLLM server | Variable | Request logs |

### Profiling Tools

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| `nsys_quick_profile.sh` | Nsight Systems timeline profiling | ~5 min | nsys-rep + SQLite DB |
| `ncu_detailed_profile.sh` | Nsight Compute kernel profiling | ~30-60 min | ncu-rep + CSV |

### Analysis Tools

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| `parse_nsys_profile.py` | Extract metrics from nsys SQLite | <1 min | CSV files + summary.json |
| `parse_ncu_profile.py` | Extract metrics from ncu CSV | <1 min | CSV files + summary.json |
| `generate_profile_report.py` | Generate HTML report | <1 min | HTML with charts |
| `check_regression.py` | Compare vs baseline | <1 min | Regression report |

---

## Usage

### All-in-One Profiling

**Quick mode** (nsys only, ~5 min):
```bash
./profile_vllm.sh --model meta-llama/Llama-2-7b-hf --mode quick
```

**Full mode** (nsys + ncu, ~60 min):
```bash
./profile_vllm.sh --model meta-llama/Llama-2-7b-hf --mode full --with-ncu
```

**MoE mode** (with expert tracking):
```bash
# Requires sitecustomize.py to be in PYTHONPATH for expert tracking
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
./profile_vllm.sh --model Qwen/Qwen2.5-MoE-72B --mode moe
```

**Custom output directory**:
```bash
./profile_vllm.sh --model <model> --mode quick --output-dir ./my_results
```

---

### Individual Scripts

#### 1. nsys_quick_profile.sh

**Run Nsight Systems profiling**:

```bash
./nsys_quick_profile.sh [model_name] [output_dir]

# Example:
./nsys_quick_profile.sh meta-llama/Llama-2-7b-hf ./results
```

**What it does**:
1. Starts vLLM server with NVTX markers enabled
2. Sends test requests
3. Captures timeline with nsys
4. Exports to SQLite
5. Parses results automatically

**Outputs**:
- `{output_dir}/vllm_quick_profile.nsys-rep` - Binary report (open in `nsys-ui`)
- `{output_dir}/vllm_quick_profile.sqlite` - SQLite export
- `{output_dir}/parsed/` - Parsed CSV files

---

#### 2. ncu_detailed_profile.sh

**Run Nsight Compute kernel profiling**:

```bash
./ncu_detailed_profile.sh [model_name] [output_dir]

# Example:
./ncu_detailed_profile.sh meta-llama/Llama-2-7b-hf ./results
```

**What it does**:
1. Starts vLLM server (with `--enforce-eager` for easier profiling)
2. Profiles specific kernels (attention, gemm, moe, etc.)
3. Collects full metrics per kernel
4. Exports to CSV
5. Parses results automatically

**Warning**: NCU profiling has 10-100x overhead. Very slow!

**Outputs**:
- `{output_dir}/vllm_ncu_profile.ncu-rep` - Binary report (open in `ncu-ui`)
- `{output_dir}/vllm_ncu_profile.csv` - CSV export
- `{output_dir}/ncu_parsed/` - Parsed metrics

---

#### 3. parse_nsys_profile.py

**Parse existing nsys SQLite database**:

```bash
python parse_nsys_profile.py <sqlite_file> --output-dir <output_dir>

# Example:
python parse_nsys_profile.py vllm_profile.sqlite --output-dir ./results
```

**Extracts**:
- ✅ Prefill vs decode breakdown
- ✅ Component timing (attention, FFN, MoE, etc.)
- ✅ Top kernels by time
- ✅ CUDA graph coverage analysis
- ✅ Memory copy statistics

**Outputs**:
- `nvtx_ranges.csv` - All NVTX markers with timings
- `kernel_stats.csv` - Per-kernel statistics
- `memcpy_stats.csv` - Memory transfer stats
- `summary.json` - Aggregated metrics

---

#### 4. parse_ncu_profile.py

**Parse existing ncu CSV**:

```bash
# First export ncu report to CSV
ncu --csv --page raw vllm_kernels.ncu-rep > vllm_kernels.csv

# Then parse
python parse_ncu_profile.py vllm_kernels.csv --output-dir ./results
```

**Extracts**:
- ✅ Per-kernel memory bandwidth
- ✅ DRAM throughput utilization
- ✅ SM (compute) utilization
- ✅ Bottleneck identification (compute vs memory)
- ✅ Roofline analysis data

**Outputs**:
- `kernel_bandwidth_metrics.csv` - Bandwidth per kernel
- `roofline_data.csv` - Roofline analysis data
- `ncu_summary.json` - Aggregated metrics

---

#### 5. generate_profile_report.py

**Generate comprehensive HTML report**:

```bash
python generate_profile_report.py \
    --nsys-results <nsys_parsed_dir> \
    --ncu-results <ncu_parsed_dir> \  # Optional
    --output report.html

# Example:
python generate_profile_report.py \
    --nsys-results ./results/parsed \
    --ncu-results ./results/ncu_parsed \
    --output ./my_report.html
```

**Report includes**:
- Executive summary with key metrics
- Prefill/decode breakdown table
- Component time breakdown
- Top kernels by time
- Bandwidth analysis with bottleneck identification
- **Performance recommendations** (automated)

**Open report**:
```bash
xdg-open report.html  # Linux
open report.html      # macOS
```

---

#### 6. check_regression.py

**Compare current profile against baseline**:

```bash
python check_regression.py \
    --current <current_summary.json> \
    --baseline <baseline_summary.json> \
    --threshold 10  # 10% threshold

# Example:
python check_regression.py \
    --current ./new_run/parsed/summary.json \
    --baseline ./baseline_profile.json \
    --threshold 10
```

**Exit codes**:
- `0` - No regressions detected
- `1` - Performance regressions found

**Use in CI/CD**:
```bash
# In GitHub Actions or similar
./scripts/profile_vllm.sh --model <model> --mode quick
./scripts/check_regression.py \
    --current ./profiling_results_*/parsed/summary.json \
    --baseline ./baseline.json \
    --threshold 10 || exit 1
```

---

#### 7. send_test_requests.py

**Send test requests to vLLM server**:

```bash
python send_test_requests.py \
    --num-requests 100 \
    --port 8000 \
    --mode mixed  # prefill, decode, or mixed

# Example:
python send_test_requests.py --num-requests 50 --mode prefill
```

**Modes**:
- `prefill` - Long prompts, short completions (prefill-heavy)
- `decode` - Short prompts, long completions (decode-heavy)
- `mixed` - Mix of both (realistic workload)

**Features**:
- Waits for server to be ready
- Triggers CUDA profiler automatically
- Reports success/failure rates
- Calculates throughput

---

## Output Directory Structure

```
profiling_results_TIMESTAMP/
├── vllm_quick_profile.nsys-rep         # Nsys binary report
├── vllm_quick_profile.sqlite           # Nsys SQLite export
├── vllm_ncu_profile.ncu-rep            # NCU binary report
├── vllm_ncu_profile.csv                # NCU CSV export
├── parsed/                             # Nsys parsed results
│   ├── nvtx_ranges.csv
│   ├── kernel_stats.csv
│   ├── memcpy_stats.csv
│   └── summary.json
├── ncu_parsed/                         # NCU parsed results
│   ├── kernel_bandwidth_metrics.csv
│   ├── roofline_data.csv
│   └── ncu_summary.json
├── moe_expert_tracking/                # MoE tracking (if mode=moe and PYTHONPATH set)
│   ├── moe_expert_activations.csv      # Expert activation counts per layer
│   ├── moe_expert_coselection.csv      # Expert co-selection patterns
│   ├── moe_routing_weights_hist.csv    # Routing weight distributions
│   ├── moe_load_imbalance.csv          # Load balancing over time
│   └── moe_summary.json                # Aggregated MoE statistics
└── profile_report.html                 # Comprehensive report
```

---

## Requirements

### Software

- **NVIDIA Nsight Systems** (`nsys`) - Timeline profiling
  - Download: https://developer.nvidia.com/nsight-systems
  - Version: 2023.4 or later recommended

- **NVIDIA Nsight Compute** (`ncu`) - Kernel profiling
  - Download: https://developer.nvidia.com/nsight-compute
  - Version: 2023.3 or later recommended

- **Python 3.8+** with packages:
  ```bash
  pip install pandas numpy requests torch
  ```

### Hardware

- NVIDIA GPU (Compute Capability 7.0+)
- For NCU: GPU must support profiling (most datacenter GPUs)
- Recommended: 32GB+ GPU memory for large models

---

## Tips & Best Practices

### For Quick Analysis

1. **Start with nsys only** (`--mode quick`)
2. **Use representative workload** - match production traffic patterns
3. **Profile for 60+ seconds** to capture steady state
4. **Check HTML report first** before diving into raw data

### For Deep Analysis

1. **Run ncu sparingly** - very high overhead
2. **Profile specific kernels** - use `--kernel-name regex:...`
3. **Compare with baseline** regularly
4. **Document profiling conditions** (model, batch size, etc.)

### For MoE Models

1. **Use mode=moe** for expert tracking
2. **Enable EPLB metrics** in vLLM for load balancing
3. **Check expert activation patterns** for skew
4. **Verify kernel efficiency** with ncu

### Common Issues

**"No data in profile"**:
- Ensure server started and received requests
- Check `--delay` is sufficient for warmup
- Verify NVTX markers enabled: `export VLLM_NVTX_SCOPES_FOR_PROFILING=1`

**"NCU takes forever"**:
- Reduce `--launch-count`
- Use more specific `--kernel-name` filter
- Profile with smaller batch size or shorter sequences

**"Parsing script fails"**:
- Check nsys/ncu version compatibility
- Try updating parsing scripts for your version
- Check actual CSV/SQLite schema: `sqlite3 file.sqlite .tables`

---

## Examples

### Example 1: Quick Health Check

```bash
# Quick profiling
./profile_vllm.sh --model meta-llama/Llama-2-7b-hf --mode quick

# Check key metrics
cat ./profiling_results_*/parsed/summary.json | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Prefill: {d['prefill_decode']['prefill']['total_ms']:.1f}ms\")
print(f\"Decode: {d['prefill_decode']['decode']['total_ms']:.1f}ms\")
print(f\"CUDA graphs: {d['cuda_graphs']['graph_coverage_pct']:.1f}%\")
"
```

### Example 2: Performance Regression Testing

```bash
# Baseline
./profile_vllm.sh --model <model> --mode quick --output-dir ./baseline
cp ./baseline/parsed/summary.json baseline_profile.json

# After changes
./profile_vllm.sh --model <model> --mode quick --output-dir ./current

# Check regressions
./check_regression.py \
    --current ./current/parsed/summary.json \
    --baseline ./baseline_profile.json \
    --threshold 5
```

### Example 3: Detailed Kernel Analysis

```bash
# First, quick profile to identify hot kernels
./nsys_quick_profile.sh meta-llama/Llama-2-7b-hf ./quick_results

# Check top kernels
cat ./quick_results/parsed/summary.json | python -c "
import sys, json
d = json.load(sys.stdin)
print('Top 3 kernels:')
for k in d['top_kernels'][:3]:
    print(f\"  {k['kernel_name'][:50]} - {k['total_time_ms']:.2f}ms\")
"

# Then, detailed NCU profiling of specific kernels
ncu --set full \
    --kernel-name "flash_attention_v2" \
    --launch-skip 50 --launch-count 10 \
    --output ./detailed.ncu-rep \
    <vllm command>
```

### Example 4: MoE Expert Load Balancing Analysis

```bash
# Profile MoE model with expert tracking
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"
./profile_vllm.sh --model Qwen/Qwen2.5-MoE-72B --mode moe

# Navigate to MoE tracking directory
cd profiling_results_*/moe_expert_tracking/

# Check expert activation coverage
python -c "
import json
with open('moe_summary.json') as f:
    summary = json.load(f)
    for layer, stats in summary.items():
        print(f\"{layer}:\")
        print(f\"  Coverage: {stats['activation_coverage_pct']:.1f}%\")
        print(f\"  Load balance: {stats['load_balance_ratio']:.2f}\")
        print(f\"  Unique experts: {stats['unique_experts_activated']}/{stats['num_experts']}\")
"

# Find most imbalanced layers
python -c "
import pandas as pd
df = pd.read_csv('moe_load_imbalance.csv')
# Get average imbalance per layer
avg_imbalance = df.groupby('layer_idx')['max_min_ratio'].mean()
print('Most imbalanced layers:')
print(avg_imbalance.sort_values(ascending=False).head(5))
"

# Analyze expert co-selection patterns
python -c "
import pandas as pd
df = pd.read_csv('moe_expert_coselection.csv')
# Find top co-selection pairs for layer 0
layer_0 = df[df['layer_idx'] == 0].nlargest(10, 'coselection_count')
print('Top 10 expert pairs (Layer 0):')
for _, row in layer_0.iterrows():
    print(f\"  ({row['expert_id_1']}, {row['expert_id_2']}): {row['coselection_count']} times\")
"
```

---

## Contributing

To add new profiling capabilities:

1. Add parser logic to `parse_nsys_profile.py` or `parse_ncu_profile.py`
2. Update HTML template in `generate_profile_report.py`
3. Add new metrics to `check_regression.py`
4. Update this README with examples

---

## See Also

- [Nsight Automated Profiling Guide](../docs/NSIGHT_AUTOMATED_PROFILING_GUIDE.md) - Complete guide
- [Part 2: Integration & CI/CD](../docs/NSIGHT_AUTOMATED_PROFILING_GUIDE_PART2.md) - Advanced topics
- [QUICK_ANSWERS.md](../QUICK_ANSWERS.md) - Quick reference
