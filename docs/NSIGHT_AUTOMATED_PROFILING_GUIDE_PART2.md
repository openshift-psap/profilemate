# Automated vLLM Profiling - Part 2: Integration & Automation

*This is a continuation of NSIGHT_AUTOMATED_PROFILING_GUIDE.md*

---

## 6. Integration with profilemate

### Combining nsys/ncu with sitecustomize Profiling

**Best practice**: Use all three together for complete coverage.

```
┌─────────────────────────────────────────────────────────────┐
│  Profiling Layer Stack                                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: sitecustomize.py (Runtime tracking)               │
│   - CUDA graph captures/replays                             │
│   - KV cache usage                                          │
│   - MoE expert activations                                  │
│   - Overhead: <3%                                           │
│                                                              │
│  Layer 2: Nsight Systems (Timeline analysis)                │
│   - Prefill/decode breakdown                                │
│   - Component timing (attention, FFN, etc.)                 │
│   - CUDA graph coverage                                     │
│   - Overhead: ~5-10%                                        │
│                                                              │
│  Layer 3: Nsight Compute (Kernel deep-dive)                 │
│   - Per-kernel bandwidth & compute                          │
│   - Roofline analysis                                       │
│   - Memory access patterns                                  │
│   - Overhead: 10-100x (use sparingly)                       │
└─────────────────────────────────────────────────────────────┘
```

### Workflow: Integrated Profiling Session

**Step 1: Initial run with sitecustomize**

```bash
export PYTHONPATH="/path/to/profilemate:$PYTHONPATH"

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --cudagraph-metrics \
    --enable-mfu-metrics \
    --port 8000
```

**Outputs**:
- `/tmp/vllm_profiling/session_*/cuda_graph_usage.csv`
- `/tmp/vllm_profiling/session_*/kv_cache_usage.csv`
- vLLM logs with MFU metrics

**Analysis**: Identify areas needing deeper investigation.

**Step 2: Timeline profiling with nsys**

```bash
# Based on sitecustomize findings, profile specific scenarios
./profilemate/scripts/nsys_quick_profile.sh \
    meta-llama/Llama-2-7b-hf \
    ./profiling_results
```

**Outputs**:
- `./profiling_results/parsed/nvtx_ranges.csv` - Phase breakdowns
- `./profiling_results/parsed/kernel_stats.csv` - Kernel timings
- `./profiling_results/parsed/summary.json` - Overall metrics

**Analysis**: Understand where time is spent (prefill vs decode, attention vs FFN).

**Step 3: Kernel-level analysis with ncu (if needed)**

```bash
# If nsys shows specific kernels consuming >20% time, profile them
./profilemate/scripts/ncu_detailed_profile.sh \
    meta-llama/Llama-2-7b-hf \
    ./profiling_results
```

**Outputs**:
- `./profiling_results/ncu_parsed/kernel_bandwidth_metrics.csv`
- `./profiling_results/ncu_parsed/roofline_data.csv`

**Analysis**: Identify compute vs memory bottlenecks.

### Cross-Referencing Results

**Example workflow**:

```python
import pandas as pd
import json

# Load sitecustomize CUDA graph data
cudagraph_df = pd.read_csv('/tmp/vllm_profiling/session_*/cuda_graph_usage.csv')

# Load nsys summary
with open('./profiling_results/parsed/summary.json') as f:
    nsys_summary = json.load(f)

# Load ncu data
ncu_df = pd.read_csv('./profiling_results/ncu_parsed/kernel_bandwidth_metrics.csv')

# Analysis: Correlate CUDA graph usage with kernel performance

# Find most-used CUDA graph config
top_graph = cudagraph_df.nlargest(1, 'replay_count').iloc[0]
print(f"Most-used graph: num_tokens={top_graph['num_tokens']}, "
      f"replays={top_graph['replay_count']}")

# Check if corresponding kernels are efficient
graph_size = top_graph['num_tokens']

# Find kernels with similar grid sizes
similar_kernels = ncu_df[
    (ncu_df['grid_size'] >= graph_size * 0.9) &
    (ncu_df['grid_size'] <= graph_size * 1.1)
]

print(f"\nKernels for graph size ~{graph_size}:")
for _, kernel in similar_kernels.iterrows():
    print(f"  {kernel['kernel_name'][:60]}")
    print(f"    Bandwidth: {kernel.get('bandwidth_gbps', 0):.2f} GB/s")
    print(f"    DRAM util: {kernel.get('dram_throughput_pct', 0):.1f}%")

# Compare with MFU analytical metrics
# (from vLLM logs with --enable-mfu-metrics)
print(f"\nNsys measured decode time: {nsys_summary['prefill_decode']['decode']['total_ms']:.2f} ms")
print(f"Nsys measured prefill time: {nsys_summary['prefill_decode']['prefill']['total_ms']:.2f} ms")

# Calculate actual bandwidth from ncu
total_bandwidth = ncu_df['bandwidth_gbps'].mean()
print(f"NCU measured avg bandwidth: {total_bandwidth:.2f} GB/s")
print(f"Compare with MFU estimate: [check vLLM logs for 'GB/s/GPU']")
```

### Automated Report with All Data

**Modify `generate_profile_report.py`** to include sitecustomize data:

```python
def load_sitecustomize_data(session_dir: Path) -> dict:
    """Load sitecustomize profiling data."""
    data = {}

    # CUDA graphs
    cudagraph_file = session_dir / "cuda_graph_usage.csv"
    if cudagraph_file.exists():
        data['cudagraph'] = pd.read_csv(cudagraph_file)

    # KV cache
    kv_cache_file = session_dir / "kv_cache_usage.csv"
    if kv_cache_file.exists():
        data['kv_cache'] = pd.read_csv(kv_cache_file)

    # MoE experts (if exists)
    moe_file = session_dir / "moe_expert_activations.csv"
    if moe_file.exists():
        data['moe'] = pd.read_csv(moe_file)

    return data

# In main():
sitecustomize_data = load_sitecustomize_data(sitecustomize_dir)

# Add sections to HTML report
# ... (modify HTML template to include these)
```

---

## 7. Complete Automation Pipeline

### All-in-One Profiling Script

**File**: `profilemate/scripts/profile_vllm.sh` (already created)

**Usage**:

```bash
# Quick profiling (nsys only, ~5 min)
./profilemate/scripts/profile_vllm.sh \
    --model meta-llama/Llama-2-7b-hf \
    --mode quick

# Full profiling (nsys + ncu, ~60 min)
./profilemate/scripts/profile_vllm.sh \
    --model meta-llama/Llama-2-7b-hf \
    --mode full \
    --with-ncu

# MoE-specific profiling
./profilemate/scripts/profile_vllm.sh \
    --model Qwen/Qwen2.5-MoE-72B \
    --mode moe
```

**What it does**:
1. ✅ Runs nsys profiling with NVTX markers
2. ✅ Parses nsys SQLite output
3. ✅ Optionally runs ncu profiling (slow)
4. ✅ Parses ncu CSV output
5. ✅ For MoE: Enables sitecustomize expert tracking
6. ✅ Generates comprehensive HTML report
7. ✅ Provides performance recommendations

### CI/CD Integration

**For automated performance regression detection**:

```yaml
# .github/workflows/profile.yml
name: Performance Profiling

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  profile:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          pip install -e .
          # Install nsys, ncu (if not on runner)

      - name: Run profiling
        run: |
          ./profilemate/scripts/profile_vllm.sh \
            --model meta-llama/Llama-2-7b-hf \
            --mode quick \
            --output-dir ./profile_results

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: profile-results
          path: ./profile_results/

      - name: Check for regressions
        run: |
          python ./profilemate/scripts/check_regression.py \
            --current ./profile_results/parsed/summary.json \
            --baseline ./baseline_profile.json \
            --threshold 10  # 10% regression threshold
```

### Baseline Comparison Script

**File**: `profilemate/scripts/check_regression.py`

```python
#!/usr/bin/env python3
"""
Compare current profile against baseline to detect regressions.

Usage:
    python check_regression.py \
        --current ./current_summary.json \
        --baseline ./baseline_summary.json \
        --threshold 10
"""

import argparse
import json
import sys


def compare_profiles(current: dict, baseline: dict, threshold: float) -> list[str]:
    """Compare profiles and return list of regressions."""
    regressions = []

    # Compare prefill/decode times
    for phase in ['prefill', 'decode']:
        current_time = current['prefill_decode'][phase]['total_ms']
        baseline_time = baseline['prefill_decode'][phase]['total_ms']

        if baseline_time > 0:
            change_pct = (current_time - baseline_time) / baseline_time * 100

            if change_pct > threshold:
                regressions.append(
                    f"❌ {phase.capitalize()} time regression: "
                    f"{change_pct:+.1f}% ({baseline_time:.2f}ms → {current_time:.2f}ms)"
                )
            elif change_pct < -threshold:
                regressions.append(
                    f"✅ {phase.capitalize()} time improvement: "
                    f"{change_pct:+.1f}% ({baseline_time:.2f}ms → {current_time:.2f}ms)"
                )

    # Compare CUDA graph coverage
    current_coverage = current['cuda_graphs']['graph_coverage_pct']
    baseline_coverage = baseline['cuda_graphs']['graph_coverage_pct']

    if current_coverage < baseline_coverage - threshold:
        regressions.append(
            f"❌ CUDA graph coverage decreased: "
            f"{baseline_coverage:.1f}% → {current_coverage:.1f}%"
        )

    # Compare top kernels
    current_top_kernel_time = current['top_kernels'][0]['total_time_ms']
    baseline_top_kernel_time = baseline['top_kernels'][0]['total_time_ms']

    if baseline_top_kernel_time > 0:
        change_pct = (current_top_kernel_time - baseline_top_kernel_time) / baseline_top_kernel_time * 100

        if change_pct > threshold:
            regressions.append(
                f"❌ Top kernel time regression: "
                f"{change_pct:+.1f}% ({current['top_kernels'][0]['kernel_name'][:40]})"
            )

    return regressions


def main():
    parser = argparse.ArgumentParser(
        description='Check for performance regressions'
    )
    parser.add_argument('--current', required=True,
                        help='Current profile summary JSON')
    parser.add_argument('--baseline', required=True,
                        help='Baseline profile summary JSON')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='Regression threshold percentage (default: 10)')

    args = parser.parse_args()

    # Load profiles
    with open(args.current) as f:
        current = json.load(f)

    with open(args.baseline) as f:
        baseline = json.load(f)

    # Compare
    results = compare_profiles(current, baseline, args.threshold)

    # Print results
    print("=== Performance Comparison ===")
    if not results:
        print("✅ No significant changes detected")
        sys.exit(0)
    else:
        for result in results:
            print(result)

        # Check if any regressions (vs improvements)
        has_regression = any('❌' in r for r in results)
        sys.exit(1 if has_regression else 0)


if __name__ == '__main__':
    main()
```

### Scheduled Profiling Dashboard

**For tracking performance over time**:

```python
# scripts/profile_dashboard.py
"""
Simple dashboard to track profiling metrics over time.

Saves profiling runs to a database and generates trend visualizations.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


class ProfileDatabase:
    """Store and query profiling results."""

    def __init__(self, db_path: str = "./profile_history.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profile_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                git_hash TEXT,
                prefill_time_ms REAL,
                decode_time_ms REAL,
                cuda_graph_coverage_pct REAL,
                avg_kernel_time_ms REAL,
                notes TEXT
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS kernel_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                kernel_name TEXT NOT NULL,
                total_time_ms REAL,
                invocations INTEGER,
                FOREIGN KEY (run_id) REFERENCES profile_runs(id)
            )
        """)

        self.conn.commit()

    def add_run(self, summary: dict, model: str, git_hash: str = None, notes: str = None):
        """Add a profiling run to the database."""
        cursor = self.conn.execute("""
            INSERT INTO profile_runs
            (timestamp, model, git_hash, prefill_time_ms, decode_time_ms,
             cuda_graph_coverage_pct, avg_kernel_time_ms, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            model,
            git_hash,
            summary['prefill_decode']['prefill']['total_ms'],
            summary['prefill_decode']['decode']['total_ms'],
            summary['cuda_graphs']['graph_coverage_pct'],
            summary['top_kernels'][0]['total_time_ms'] if summary['top_kernels'] else None,
            notes,
        ))

        run_id = cursor.lastrowid

        # Add kernel metrics
        for kernel in summary['top_kernels'][:20]:
            self.conn.execute("""
                INSERT INTO kernel_metrics
                (run_id, kernel_name, total_time_ms, invocations)
                VALUES (?, ?, ?, ?)
            """, (
                run_id,
                kernel['kernel_name'][:100],
                kernel['total_time_ms'],
                kernel['invocations'],
            ))

        self.conn.commit()
        return run_id

    def get_trends(self, model: str, days: int = 30) -> pd.DataFrame:
        """Get profiling trends for a model."""
        query = """
            SELECT timestamp, prefill_time_ms, decode_time_ms, cuda_graph_coverage_pct
            FROM profile_runs
            WHERE model = ?
            AND timestamp >= datetime('now', ? || ' days')
            ORDER BY timestamp
        """
        return pd.read_sql_query(query, self.conn, params=(model, -days))

    def plot_trends(self, model: str, output_dir: Path):
        """Generate trend plots."""
        output_dir.mkdir(parents=True, exist_ok=True)
        trends = self.get_trends(model)

        if trends.empty:
            print(f"No data for model: {model}")
            return

        trends['timestamp'] = pd.to_datetime(trends['timestamp'])

        # Plot time trends
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        ax1.plot(trends['timestamp'], trends['prefill_time_ms'], label='Prefill', marker='o')
        ax1.plot(trends['timestamp'], trends['decode_time_ms'], label='Decode', marker='o')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title(f'Prefill/Decode Time Trends - {model}')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(trends['timestamp'], trends['cuda_graph_coverage_pct'], marker='o', color='green')
        ax2.set_ylabel('Coverage (%)')
        ax2.set_xlabel('Date')
        ax2.set_title('CUDA Graph Coverage Trend')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(output_dir / f'trends_{model.replace("/", "_")}.png', dpi=150)
        print(f"Trend plot saved: {output_dir / f'trends_{model.replace(\"/\", \"_\")}.png'}")

# Usage:
# from profile_dashboard import ProfileDatabase
#
# db = ProfileDatabase()
# with open('./profiling_results/parsed/summary.json') as f:
#     summary = json.load(f)
# db.add_run(summary, model='meta-llama/Llama-2-7b-hf', git_hash='abc123')
# db.plot_trends('meta-llama/Llama-2-7b-hf', Path('./dashboard'))
```

---

## Summary

### Complete Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: Initial Profiling (sitecustomize)                      │
│  ├─ Enable: export PYTHONPATH=/path/to/profilemate:$PYTHONPATH  │
│  ├─ Run: vllm serve --cudagraph-metrics --enable-mfu-metrics    │
│  └─ Analyze: CUDA graph usage, KV cache, MFU estimates          │
├──────────────────────────────────────────────────────────────────┤
│  Step 2: Timeline Analysis (nsys)                               │
│  ├─ Run: ./scripts/nsys_quick_profile.sh                        │
│  ├─ Parse: ./scripts/parse_nsys_profile.py                      │
│  └─ Analyze: Prefill/decode split, component breakdown          │
├──────────────────────────────────────────────────────────────────┤
│  Step 3: Kernel Analysis (ncu) - Optional                       │
│  ├─ Run: ./scripts/ncu_detailed_profile.sh                      │
│  ├─ Parse: ./scripts/parse_ncu_profile.py                       │
│  └─ Analyze: Per-kernel bandwidth, roofline, bottlenecks        │
├──────────────────────────────────────────────────────────────────┤
│  Step 4: Report Generation                                      │
│  ├─ Run: ./scripts/generate_profile_report.py                   │
│  └─ Output: Comprehensive HTML report with recommendations      │
├──────────────────────────────────────────────────────────────────┤
│  Step 5: Trend Tracking (Optional)                              │
│  ├─ Store: Add results to profile database                      │
│  ├─ Compare: Check for regressions vs baseline                  │
│  └─ Visualize: Generate trend plots over time                   │
└──────────────────────────────────────────────────────────────────┘
```

### Quick Reference Commands

```bash
# Quick profiling (all-in-one, ~5 min)
./profilemate/scripts/profile_vllm.sh --model meta-llama/Llama-2-7b-hf --mode quick

# Full profiling (nsys + ncu, ~60 min)
./profilemate/scripts/profile_vllm.sh --model meta-llama/Llama-2-7b-hf --mode full --with-ncu

# MoE-specific
./profilemate/scripts/profile_vllm.sh --model Qwen/Qwen2.5-MoE-72B --mode moe

# Manual nsys profiling
./profilemate/scripts/nsys_quick_profile.sh meta-llama/Llama-2-7b-hf ./results

# Manual ncu profiling
./profilemate/scripts/ncu_detailed_profile.sh meta-llama/Llama-2-7b-hf ./results

# Parse existing nsys SQLite
python ./profilemate/scripts/parse_nsys_profile.py vllm_profile.sqlite --output-dir ./parsed

# Parse existing ncu CSV
ncu --csv --page raw vllm_kernels.ncu-rep > vllm_kernels.csv
python ./profilemate/scripts/parse_ncu_profile.py vllm_kernels.csv --output-dir ./ncu_parsed

# Generate HTML report
python ./profilemate/scripts/generate_profile_report.py \
    --nsys-results ./parsed \
    --ncu-results ./ncu_parsed \
    --output report.html

# Check for regressions
python ./profilemate/scripts/check_regression.py \
    --current ./results/parsed/summary.json \
    --baseline ./baseline.json \
    --threshold 10
```

### Output Files Summary

```
profiling_results_TIMESTAMP/
├── vllm_quick_profile.nsys-rep         # Nsys binary (open in nsys-ui)
├── vllm_quick_profile.sqlite           # Nsys SQLite export
├── vllm_ncu_profile.ncu-rep            # NCU binary (open in ncu-ui)
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
├── moe_expert_tracking/                # MoE sitecustomize results (if mode=moe)
│   ├── moe_expert_activations.csv
│   ├── moe_expert_coselection.csv
│   └── moe_routing_weights_hist.csv
└── profile_report.html                 # Comprehensive HTML report
```

---

## Best Practices

### For Development

1. **Start with quick nsys profiling** to understand overall performance
2. **Use sitecustomize** for continuous monitoring during development
3. **Run ncu sparingly** - only when you need kernel-level details
4. **Compare against baseline** regularly to catch regressions early

### For Production Analysis

1. **Run profiling on representative workload** - use realistic request patterns
2. **Profile for sufficient duration** - at least 60s to capture steady state
3. **Profile different scenarios**:
   - Prefill-heavy (long prompts, short outputs)
   - Decode-heavy (short prompts, long outputs)
   - Mixed workload (realistic)
4. **Document profiling conditions** - model, batch size, sequence lengths, etc.

### For MoE Models

1. **Always enable EPLB load balancing metrics** (`--eplb-config '{"log_balancedness": true}'`)
2. **Use sitecustomize for expert tracking** to understand activation patterns
3. **Profile with ncu** to verify expert kernel efficiency
4. **Check for expert imbalance** - high skew can invalidate MFU estimates

### Common Pitfalls

❌ **Don't**:
- Profile during warmup (use `--delay` with nsys)
- Run ncu on full production workloads (too slow)
- Trust single-run results (variance can be high)
- Mix profiling with other workloads on same GPU

✅ **Do**:
- Profile with realistic batch sizes
- Run multiple iterations and average results
- Use dedicated profiling environment
- Document exact command used for reproducibility

---

## Troubleshooting

### nsys Issues

**Problem**: "Profile contains no data"
- **Solution**: Ensure vLLM server started properly and received requests
- Check logs for `torch.cuda.cudart().cudaProfilerStart()` calls

**Problem**: NVTX ranges not showing
- **Solution**: Set `export VLLM_NVTX_SCOPES_FOR_PROFILING=1` before starting

**Problem**: SQLite export failed
- **Solution**: Use `--export=sqlite` flag explicitly

### ncu Issues

**Problem**: "No kernels matched regex"
- **Solution**: Run without `--kernel-name` filter first to see all kernels
- Use broader regex: `regex:".*"` to capture everything

**Problem**: ncu takes too long
- **Solution**: Reduce `--launch-count` or use more specific `--kernel-name` filter
- Profile with `--enforce-eager` to avoid CUDA graphs (easier to profile)

**Problem**: CSV export is empty
- **Solution**: Try `--csv --page raw` instead of just `--csv`
- Some metrics require specific sections: `--page details`

### Parsing Script Issues

**Problem**: "Column not found" errors
- **Solution**: NCU/nsys output format may vary by version
- Update parsing scripts to handle different column names
- Check actual CSV headers: `head -20 file.csv`

**Problem**: No prefill/decode detected
- **Solution**: Verify NVTX markers are enabled
- Check for marker names in SQLite: `SELECT DISTINCT value FROM StringIds WHERE value LIKE '%prefill%'`

---

## Future Enhancements

### Planned Features

1. **Real-time profiling dashboard** - Live visualization of running profile
2. **Automated bottleneck identification** - ML-based performance analysis
3. **Multi-GPU profiling** - Coordinated profiling across TP/PP ranks
4. **Cloud integration** - Upload results to centralized performance database
5. **A/B testing framework** - Automated comparison of different configs

### Contributing

To add new profiling capabilities:

1. Add parser to `parse_nsys_profile.py` or `parse_ncu_profile.py`
2. Update HTML template in `generate_profile_report.py`
3. Add new metrics to regression checker
4. Update documentation with examples

---

## References

- **Nsight Systems Documentation**: https://docs.nvidia.com/nsight-systems/
- **Nsight Compute Documentation**: https://docs.nvidia.com/nsight-compute/
- **vLLM NVTX Implementation**: `vllm/vllm/profiler.py`
- **CUDA Profiler API**: https://docs.nvidia.com/cuda/profiler-users-guide/

---

*End of Automated Profiling Guide Part 2*
