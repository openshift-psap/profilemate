# Automated vLLM Profiling with Nsight Systems and Nsight Compute

## Table of Contents

1. [Overview](#overview)
2. [Profiling Workflow](#profiling-workflow)
3. [Nsight Systems (nsys) - Timeline Analysis](#nsight-systems-nsys---timeline-analysis)
4. [Nsight Compute (ncu) - Kernel Analysis](#nsight-compute-ncu---kernel-analysis)
5. [Automated Analysis Scripts](#automated-analysis-scripts)
6. [Integration with profilemate](#integration-with-profilemate)
7. [Complete Automation Pipeline](#complete-automation-pipeline)

---

## Overview

### What We'll Extract

**From Nsight Systems (nsys)**:
- ✅ Prefill vs Decode phase breakdown (using NVTX markers)
- ✅ Attention vs FFN vs other component timing
- ✅ Per-layer execution time
- ✅ CUDA graph replays
- ✅ CPU-GPU synchronization overhead
- ✅ Kernel launch overhead

**From Nsight Compute (ncu)**:
- ✅ Per-kernel memory bandwidth utilization
- ✅ Compute throughput (SM efficiency, achieved occupancy)
- ✅ Memory access patterns (cache hit rates)
- ✅ Tensor Core utilization
- ✅ MoE expert kernel patterns

**Combined Analysis**:
- ✅ Roofline analysis
- ✅ Bottleneck identification
- ✅ Performance recommendations

### Tools Required

```bash
# Check NVIDIA tools are installed
nsys --version  # Nsight Systems
ncu --version   # Nsight Compute

# If not installed (on Linux):
# Download from: https://developer.nvidia.com/nsight-systems
# Download from: https://developer.nvidia.com/nsight-compute
```

---

## Profiling Workflow

### Phase 1: Quick Profile (5-10 min)

**Goal**: Get high-level timeline, identify phases

```bash
./profilemate/scripts/nsys_quick_profile.sh
```

**Extracts**:
- Prefill/decode time split
- Major component breakdown
- CUDA graph coverage

### Phase 2: Detailed Kernel Analysis (30-60 min)

**Goal**: Deep dive into specific kernels

```bash
./profilemate/scripts/ncu_detailed_profile.sh
```

**Extracts**:
- Per-kernel bandwidth
- Compute utilization
- Roofline data

### Phase 3: Automated Report (1 min)

**Goal**: Generate comprehensive performance report

```bash
./profilemate/scripts/generate_profile_report.py
```

**Outputs**:
- HTML report with charts
- CSV data files
- Performance recommendations

---

## Nsight Systems (nsys) - Timeline Analysis

### Setup: Enable NVTX Markers

**vLLM has built-in NVTX support**:

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING=1
```

**What this enables**:
- Layer-by-layer markers (when `--enable-layerwise-nvtx-tracing`)
- Prefill/decode phase markers
- Component markers (attention, FFN, etc.)

### Basic nsys Command

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --output=vllm_profile \
    --force-overwrite=true \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-hf \
        --max-num-seqs 64 \
        --max-model-len 2048 &

# Wait for server to start
sleep 30

# Send requests to trigger profiling
python profilemate/scripts/send_test_requests.py --num-requests 100

# Stop profiling (server will shut down after requests)
```

### Advanced nsys Command (Recommended)

```bash
nsys profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --cuda-graph-trace=node \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --sample=cpu \
    --cpuctxsw=none \
    --backtrace=dwarf \
    --delay=30 \
    --duration=60 \
    --output=vllm_detailed_profile \
    --export=sqlite,text \
    --force-overwrite=true \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-hf \
        --max-num-seqs 256 \
        --max-model-len 2048
```

**Flags explained**:
- `--trace=cuda,nvtx,osrt,cublas,cudnn`: Trace CUDA kernels, NVTX ranges, cuBLAS, cuDNN
- `--cuda-graph-trace=node`: Trace CUDA graph node execution
- `--cuda-memory-usage=true`: Track GPU memory allocations
- `--capture-range=cudaProfilerApi`: Only profile when explicitly triggered (reduces overhead)
- `--sample=cpu`: Sample CPU activity
- `--delay=30`: Wait 30s before starting (let server initialize)
- `--duration=60`: Profile for 60s
- `--export=sqlite,text`: Export to SQLite DB and text files

### Triggering Profiling Programmatically

**In your workload script**:

```python
import torch

# Start profiling
torch.cuda.cudart().cudaProfilerStart()

# Run your workload
# ... send requests ...

# Stop profiling
torch.cuda.cudart().cudaProfilerStop()
```

### nsys Output Files

```
vllm_detailed_profile.nsys-rep    # Binary report (open in nsys-ui)
vllm_detailed_profile.sqlite      # SQLite database (for scripting)
vllm_detailed_profile_*.txt       # Text exports
```

### Extracting Data from SQLite

**Schema exploration**:

```bash
sqlite3 vllm_detailed_profile.sqlite ".tables"
```

**Key tables**:
- `NVTX_EVENTS` - NVTX range markers
- `CUPTI_ACTIVITY_KIND_KERNEL` - Kernel launches
- `CUPTI_ACTIVITY_KIND_MEMCPY` - Memory copies
- `StringIds` - String lookup table

**Example query** (prefill vs decode time):

```sql
-- Extract NVTX ranges
SELECT
    s.value as name,
    (n.end - n.start) / 1000000.0 as duration_ms,
    n.start,
    n.end
FROM NVTX_EVENTS n
JOIN StringIds s ON n.textId = s.id
WHERE s.value LIKE '%prefill%' OR s.value LIKE '%decode%'
ORDER BY n.start;
```

---

## Nsight Compute (ncu) - Kernel Analysis

### When to Use NCU

**Use NCU for**:
- ✅ Per-kernel bandwidth and compute analysis
- ✅ Identifying inefficient kernels
- ✅ Roofline analysis
- ✅ Memory access pattern analysis

**Don't use NCU for**:
- ✗ Timeline profiling (use nsys)
- ✗ Full runs (very high overhead 10-100x)
- ✗ Multi-GPU profiling (limited support)

### Basic ncu Command

```bash
ncu \
    --set full \
    --target-processes all \
    --kernel-name regex:"attention|gemm|moe" \
    --launch-skip 100 \
    --launch-count 10 \
    --output vllm_kernels \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-hf \
        --max-num-seqs 32
```

**Flags**:
- `--set full`: Collect all metrics (use `--set detailed` for even more)
- `--target-processes all`: Profile all processes (important for multi-GPU)
- `--kernel-name regex:"attention|gemm|moe"`: Only profile specific kernels
- `--launch-skip 100`: Skip first 100 kernel launches (warmup)
- `--launch-count 10`: Profile next 10 launches

### Advanced ncu - Roofline Analysis

```bash
ncu \
    --set roofline \
    --target-processes all \
    --kernel-name regex:"gemm|attention" \
    --launch-skip 50 \
    --launch-count 20 \
    --cache-control all \
    --clock-control base \
    --output vllm_roofline \
    python script.py
```

**Metrics collected**:
- DRAM bandwidth
- L2 cache bandwidth
- L1 cache bandwidth
- Arithmetic intensity
- FP32/FP16/INT8 throughput

### MoE-Specific Profiling

```bash
ncu \
    --set full \
    --target-processes all \
    --kernel-name regex:"moe|expert|topk|routing" \
    --launch-skip 100 \
    --launch-count 50 \
    --output vllm_moe_kernels \
    --csv \
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-MoE-72B \
        --tensor-parallel-size 4
```

### ncu Output Files

```
vllm_kernels.ncu-rep    # Binary report (open in ncu-ui)
vllm_kernels.csv        # CSV export (with --csv flag)
```

### Key Metrics to Extract

**From ncu CSV or API**:

```python
# Bandwidth metrics
"dram__bytes_read.sum"                    # Total DRAM reads
"dram__bytes_write.sum"                   # Total DRAM writes
"dram__throughput.avg.pct_of_peak_sustained_elapsed"  # % of peak bandwidth

# Compute metrics
"sm__throughput.avg.pct_of_peak_sustained_elapsed"    # SM utilization
"sm__sass_thread_inst_executed_op_fadd_pred_on.sum"   # FP32 add ops
"sm__sass_thread_inst_executed_op_fmul_pred_on.sum"   # FP32 mul ops

# Tensor Core metrics (if applicable)
"sm__inst_executed_pipe_tensor.sum"                   # Tensor Core instructions

# Memory efficiency
"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"     # L1 load sectors
"l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"     # L1 store sectors
"smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"  # Load efficiency
"smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct"  # Store efficiency

# Occupancy
"sm__warps_active.avg.pct_of_peak_sustained_active"   # Achieved occupancy
```

---

## Automated Analysis Scripts

### Script 1: nsys SQLite Parser

**File**: `profilemate/scripts/parse_nsys_profile.py`

```python
#!/usr/bin/env python3
"""
Parse Nsight Systems SQLite export and extract performance metrics.

Usage:
    python parse_nsys_profile.py vllm_profile.sqlite --output-dir ./results
"""

import argparse
import sqlite3
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd


class NsysProfileParser:
    """Parse nsys SQLite database."""

    def __init__(self, sqlite_path: str):
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.row_factory = sqlite3.Row

    def get_nvtx_ranges(self) -> pd.DataFrame:
        """Extract NVTX range timings."""
        query = """
        SELECT
            s.value as name,
            (n.end - n.start) / 1000000.0 as duration_ms,
            n.start as start_ns,
            n.end as end_ns,
            n.globalTid as thread_id
        FROM NVTX_EVENTS n
        JOIN StringIds s ON n.textId = s.id
        ORDER BY n.start
        """
        return pd.read_sql_query(query, self.conn)

    def get_kernel_stats(self) -> pd.DataFrame:
        """Extract kernel execution statistics."""
        query = """
        SELECT
            s.value as kernel_name,
            (k.end - k.start) / 1000.0 as duration_us,
            k.start as start_ns,
            k.gridX * k.gridY * k.gridZ as grid_size,
            k.blockX * k.blockY * k.blockZ as block_size,
            k.staticSharedMemory + k.dynamicSharedMemory as shared_mem_bytes,
            k.registersPerThread
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
        """
        return pd.read_sql_query(query, self.conn)

    def get_memcpy_stats(self) -> pd.DataFrame:
        """Extract memory copy statistics."""
        query = """
        SELECT
            (m.end - m.start) / 1000.0 as duration_us,
            m.bytes,
            m.bytes / ((m.end - m.start) / 1000000000.0) / 1e9 as bandwidth_gbps,
            CASE m.copyKind
                WHEN 1 THEN 'HtoD'
                WHEN 2 THEN 'DtoH'
                WHEN 8 THEN 'DtoD'
                ELSE 'Other'
            END as direction
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        ORDER BY m.start
        """
        return pd.read_sql_query(query, self.conn)

    def analyze_prefill_decode(self, nvtx_df: pd.DataFrame) -> dict:
        """Analyze prefill vs decode breakdown."""
        results = {
            'prefill': {'count': 0, 'total_ms': 0.0, 'avg_ms': 0.0},
            'decode': {'count': 0, 'total_ms': 0.0, 'avg_ms': 0.0},
            'other': {'count': 0, 'total_ms': 0.0, 'avg_ms': 0.0}
        }

        for _, row in nvtx_df.iterrows():
            name = row['name'].lower()
            duration = row['duration_ms']

            if 'prefill' in name:
                results['prefill']['count'] += 1
                results['prefill']['total_ms'] += duration
            elif 'decode' in name or 'generation' in name:
                results['decode']['count'] += 1
                results['decode']['total_ms'] += duration
            else:
                results['other']['count'] += 1
                results['other']['total_ms'] += duration

        # Calculate averages
        for phase in results:
            if results[phase]['count'] > 0:
                results[phase]['avg_ms'] = (
                    results[phase]['total_ms'] / results[phase]['count']
                )

        return results

    def analyze_component_breakdown(self, nvtx_df: pd.DataFrame) -> dict:
        """Analyze time spent in different model components."""
        components = defaultdict(lambda: {'count': 0, 'total_ms': 0.0})

        component_patterns = {
            'attention': ['attention', 'attn', 'qkv', 'softmax'],
            'ffn': ['ffn', 'mlp', 'feed_forward'],
            'moe': ['moe', 'expert', 'routing', 'topk'],
            'layernorm': ['layernorm', 'rms_norm', 'norm'],
            'embedding': ['embed', 'lm_head', 'token'],
        }

        for _, row in nvtx_df.iterrows():
            name = row['name'].lower()
            duration = row['duration_ms']

            matched = False
            for component, patterns in component_patterns.items():
                if any(pattern in name for pattern in patterns):
                    components[component]['count'] += 1
                    components[component]['total_ms'] += duration
                    matched = True
                    break

            if not matched:
                components['other']['count'] += 1
                components['other']['total_ms'] += duration

        # Calculate percentages
        total_time = sum(c['total_ms'] for c in components.values())
        for component in components:
            components[component]['percent'] = (
                components[component]['total_ms'] / total_time * 100
                if total_time > 0 else 0
            )

        return dict(components)

    def analyze_kernel_efficiency(self, kernel_df: pd.DataFrame) -> dict:
        """Analyze kernel execution efficiency."""
        # Group by kernel name
        kernel_groups = kernel_df.groupby('kernel_name')

        results = []
        for name, group in kernel_groups:
            results.append({
                'kernel_name': name[:80],  # Truncate long names
                'invocations': len(group),
                'total_time_ms': group['duration_us'].sum() / 1000,
                'avg_time_us': group['duration_us'].mean(),
                'min_time_us': group['duration_us'].min(),
                'max_time_us': group['duration_us'].max(),
                'avg_grid_size': int(group['grid_size'].mean()),
                'avg_block_size': int(group['block_size'].mean()),
            })

        # Sort by total time
        results.sort(key=lambda x: x['total_time_ms'], reverse=True)
        return results

    def analyze_cuda_graphs(self, kernel_df: pd.DataFrame) -> dict:
        """Analyze CUDA graph usage."""
        # Identify CUDA graph kernels (heuristic: kernels with identical grid/block sizes)
        kernel_signatures = defaultdict(list)

        for _, row in kernel_df.iterrows():
            signature = (
                row['kernel_name'],
                row['grid_size'],
                row['block_size'],
                row['shared_mem_bytes'],
            )
            kernel_signatures[signature].append(row['duration_us'])

        # Find repeated patterns (likely CUDA graphs)
        cuda_graph_candidates = {
            k: v for k, v in kernel_signatures.items()
            if len(v) > 10  # Repeated more than 10 times
        }

        total_kernels = len(kernel_df)
        graph_kernel_count = sum(len(v) for v in cuda_graph_candidates.values())

        return {
            'total_kernel_launches': total_kernels,
            'graph_kernel_launches': graph_kernel_count,
            'graph_coverage_pct': graph_kernel_count / total_kernels * 100 if total_kernels > 0 else 0,
            'num_unique_graphs': len(cuda_graph_candidates),
        }

    def generate_report(self, output_dir: Path):
        """Generate comprehensive profiling report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Extracting NVTX ranges...")
        nvtx_df = self.get_nvtx_ranges()
        nvtx_df.to_csv(output_dir / "nvtx_ranges.csv", index=False)

        print("Extracting kernel statistics...")
        kernel_df = self.get_kernel_stats()
        kernel_df.to_csv(output_dir / "kernel_stats.csv", index=False)

        print("Extracting memory copy statistics...")
        memcpy_df = self.get_memcpy_stats()
        memcpy_df.to_csv(output_dir / "memcpy_stats.csv", index=False)

        print("\nAnalyzing performance...")

        # Prefill vs Decode
        prefill_decode = self.analyze_prefill_decode(nvtx_df)
        print("\n=== Prefill vs Decode Breakdown ===")
        for phase, stats in prefill_decode.items():
            print(f"{phase.capitalize():10} | "
                  f"Count: {stats['count']:4} | "
                  f"Total: {stats['total_ms']:8.2f} ms | "
                  f"Avg: {stats['avg_ms']:8.2f} ms")

        # Component breakdown
        components = self.analyze_component_breakdown(nvtx_df)
        print("\n=== Component Time Breakdown ===")
        for component, stats in sorted(
            components.items(),
            key=lambda x: x[1]['total_ms'],
            reverse=True
        ):
            print(f"{component.capitalize():15} | "
                  f"{stats['total_ms']:8.2f} ms | "
                  f"{stats['percent']:5.1f}%")

        # Kernel efficiency
        kernel_efficiency = self.analyze_kernel_efficiency(kernel_df)
        print("\n=== Top 10 Kernels by Time ===")
        for i, kernel in enumerate(kernel_efficiency[:10], 1):
            print(f"{i:2}. {kernel['kernel_name'][:60]}")
            print(f"    Invocations: {kernel['invocations']}, "
                  f"Total: {kernel['total_time_ms']:.2f} ms, "
                  f"Avg: {kernel['avg_time_us']:.2f} us")

        # CUDA graphs
        cuda_graphs = self.analyze_cuda_graphs(kernel_df)
        print("\n=== CUDA Graph Analysis ===")
        print(f"Total kernel launches: {cuda_graphs['total_kernel_launches']}")
        print(f"Graph kernel launches: {cuda_graphs['graph_kernel_launches']}")
        print(f"Graph coverage: {cuda_graphs['graph_coverage_pct']:.1f}%")
        print(f"Unique graphs: {cuda_graphs['num_unique_graphs']}")

        # Memory bandwidth
        if not memcpy_df.empty:
            print("\n=== Memory Copy Statistics ===")
            for direction in ['HtoD', 'DtoH', 'DtoD']:
                subset = memcpy_df[memcpy_df['direction'] == direction]
                if not subset.empty:
                    total_mb = subset['bytes'].sum() / 1e6
                    avg_bw = subset['bandwidth_gbps'].mean()
                    print(f"{direction}: {total_mb:.2f} MB, Avg BW: {avg_bw:.2f} GB/s")

        # Save summary
        summary = {
            'prefill_decode': prefill_decode,
            'components': components,
            'top_kernels': kernel_efficiency[:20],
            'cuda_graphs': cuda_graphs,
        }

        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {output_dir}")

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Parse Nsight Systems profile and extract metrics'
    )
    parser.add_argument('sqlite_file', help='Path to .sqlite file from nsys')
    parser.add_argument('--output-dir', default='./nsys_results',
                        help='Output directory for results')

    args = parser.parse_args()

    nsys_parser = NsysProfileParser(args.sqlite_file)
    nsys_parser.generate_report(Path(args.output_dir))
    nsys_parser.close()


if __name__ == '__main__':
    main()
```

### Script 2: ncu CSV Parser

**File**: `profilemate/scripts/parse_ncu_profile.py`

```python
#!/usr/bin/env python3
"""
Parse Nsight Compute CSV export and extract kernel metrics.

Usage:
    # Export ncu report to CSV first:
    ncu --csv --page raw vllm_kernels.ncu-rep > vllm_kernels.csv

    # Then parse:
    python parse_ncu_profile.py vllm_kernels.csv --output-dir ./results
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd


class NcuProfileParser:
    """Parse NCU CSV export."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = self._load_csv()

    def _load_csv(self) -> pd.DataFrame:
        """Load NCU CSV file."""
        # NCU CSV has a specific format with metadata at the top
        # Skip lines until we find the header
        with open(self.csv_path, 'r') as f:
            lines = f.readlines()

        # Find header line
        header_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('"ID"') or line.startswith('ID'):
                header_idx = i
                break

        # Read from header
        df = pd.read_csv(self.csv_path, skiprows=header_idx)
        return df

    def extract_bandwidth_metrics(self) -> list[dict]:
        """Extract memory bandwidth metrics per kernel."""
        results = []

        # Group by kernel name
        if 'Kernel Name' in self.data.columns:
            kernel_col = 'Kernel Name'
        elif 'Name' in self.data.columns:
            kernel_col = 'Name'
        else:
            print("Warning: Could not find kernel name column")
            return results

        for kernel_name in self.data[kernel_col].unique():
            kernel_data = self.data[self.data[kernel_col] == kernel_name]

            # Extract relevant metrics (column names may vary)
            metrics = {}

            # Memory bandwidth
            for col in kernel_data.columns:
                col_lower = col.lower()
                if 'dram' in col_lower and 'read' in col_lower and 'byte' in col_lower:
                    metrics['dram_read_bytes'] = kernel_data[col].iloc[0]
                elif 'dram' in col_lower and 'write' in col_lower and 'byte' in col_lower:
                    metrics['dram_write_bytes'] = kernel_data[col].iloc[0]
                elif 'dram' in col_lower and 'throughput' in col_lower:
                    metrics['dram_throughput_pct'] = kernel_data[col].iloc[0]
                elif 'sm' in col_lower and 'throughput' in col_lower:
                    metrics['sm_throughput_pct'] = kernel_data[col].iloc[0]
                elif 'duration' in col_lower:
                    metrics['duration_ns'] = kernel_data[col].iloc[0]

            if metrics:
                # Calculate bandwidth
                if 'dram_read_bytes' in metrics and 'dram_write_bytes' in metrics and 'duration_ns' in metrics:
                    total_bytes = metrics.get('dram_read_bytes', 0) + metrics.get('dram_write_bytes', 0)
                    duration_s = metrics['duration_ns'] / 1e9
                    metrics['bandwidth_gbps'] = total_bytes / duration_s / 1e9

                results.append({
                    'kernel_name': kernel_name[:80],
                    **metrics
                })

        return results

    def calculate_roofline_data(self, bandwidth_metrics: list[dict]) -> list[dict]:
        """Calculate arithmetic intensity for roofline analysis."""
        roofline_data = []

        for kernel in bandwidth_metrics:
            # Arithmetic intensity = FLOPs / Bytes
            # We need FLOPs estimate (not always available in CSV)

            # If we have throughput percentages, estimate
            if 'sm_throughput_pct' in kernel and 'bandwidth_gbps' in kernel:
                # Rough estimate based on throughput
                kernel['compute_bound'] = (
                    kernel.get('sm_throughput_pct', 0) >
                    kernel.get('dram_throughput_pct', 0)
                )

                roofline_data.append({
                    'kernel_name': kernel['kernel_name'],
                    'sm_utilization': kernel.get('sm_throughput_pct', 0),
                    'memory_utilization': kernel.get('dram_throughput_pct', 0),
                    'bottleneck': 'compute' if kernel.get('compute_bound', False) else 'memory',
                    'bandwidth_gbps': kernel.get('bandwidth_gbps', 0),
                })

        return roofline_data

    def analyze_moe_kernels(self, bandwidth_metrics: list[dict]) -> dict:
        """Analyze MoE-specific kernels."""
        moe_kernels = []

        for kernel in bandwidth_metrics:
            name = kernel['kernel_name'].lower()
            if any(pattern in name for pattern in ['moe', 'expert', 'topk', 'routing']):
                moe_kernels.append(kernel)

        if not moe_kernels:
            return {'moe_kernels_found': False}

        total_time = sum(k.get('duration_ns', 0) for k in moe_kernels)
        total_bandwidth = sum(k.get('bandwidth_gbps', 0) for k in moe_kernels)

        return {
            'moe_kernels_found': True,
            'num_moe_kernels': len(moe_kernels),
            'total_time_ms': total_time / 1e6,
            'avg_bandwidth_gbps': total_bandwidth / len(moe_kernels) if moe_kernels else 0,
            'kernels': moe_kernels[:10],  # Top 10
        }

    def generate_report(self, output_dir: Path):
        """Generate NCU analysis report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Extracting bandwidth metrics...")
        bandwidth_metrics = self.extract_bandwidth_metrics()

        if not bandwidth_metrics:
            print("Warning: No bandwidth metrics found. Check CSV format.")
            return

        # Save raw metrics
        pd.DataFrame(bandwidth_metrics).to_csv(
            output_dir / "kernel_bandwidth_metrics.csv",
            index=False
        )

        print("\n=== Top 10 Kernels by Bandwidth ===")
        sorted_kernels = sorted(
            bandwidth_metrics,
            key=lambda x: x.get('bandwidth_gbps', 0),
            reverse=True
        )

        for i, kernel in enumerate(sorted_kernels[:10], 1):
            print(f"{i:2}. {kernel['kernel_name'][:60]}")
            print(f"    Bandwidth: {kernel.get('bandwidth_gbps', 0):.2f} GB/s, "
                  f"DRAM Util: {kernel.get('dram_throughput_pct', 0):.1f}%, "
                  f"SM Util: {kernel.get('sm_throughput_pct', 0):.1f}%")

        # Roofline analysis
        print("\nCalculating roofline data...")
        roofline_data = self.calculate_roofline_data(bandwidth_metrics)
        pd.DataFrame(roofline_data).to_csv(
            output_dir / "roofline_data.csv",
            index=False
        )

        compute_bound = sum(1 for k in roofline_data if k['bottleneck'] == 'compute')
        memory_bound = len(roofline_data) - compute_bound

        print(f"\n=== Bottleneck Analysis ===")
        print(f"Compute-bound kernels: {compute_bound}")
        print(f"Memory-bound kernels: {memory_bound}")

        # MoE analysis
        moe_analysis = self.analyze_moe_kernels(bandwidth_metrics)
        if moe_analysis['moe_kernels_found']:
            print(f"\n=== MoE Kernel Analysis ===")
            print(f"MoE kernels found: {moe_analysis['num_moe_kernels']}")
            print(f"Total MoE time: {moe_analysis['total_time_ms']:.2f} ms")
            print(f"Avg MoE bandwidth: {moe_analysis['avg_bandwidth_gbps']:.2f} GB/s")

        # Save summary
        summary = {
            'total_kernels': len(bandwidth_metrics),
            'top_kernels_by_bandwidth': sorted_kernels[:20],
            'roofline_summary': {
                'compute_bound': compute_bound,
                'memory_bound': memory_bound,
            },
            'moe_analysis': moe_analysis,
        }

        with open(output_dir / "ncu_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse Nsight Compute CSV and extract kernel metrics'
    )
    parser.add_argument('csv_file', help='Path to CSV file from ncu')
    parser.add_argument('--output-dir', default='./ncu_results',
                        help='Output directory for results')

    args = parser.parse_args()

    ncu_parser = NcuProfileParser(args.csv_file)
    ncu_parser.generate_report(Path(args.output_dir))


if __name__ == '__main__':
    main()
```

### Script 3: Combined Report Generator

**File**: `profilemate/scripts/generate_profile_report.py`

```python
#!/usr/bin/env python3
"""
Generate comprehensive profiling report from nsys and ncu results.

Usage:
    python generate_profile_report.py \
        --nsys-results ./nsys_results \
        --ncu-results ./ncu_results \
        --output report.html
"""

import argparse
import json
from pathlib import Path
import pandas as pd


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>vLLM Profiling Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .warning {{ color: #ff9800; }}
        .error {{ color: #f44336; }}
        .good {{ color: #4CAF50; }}
    </style>
</head>
<body>
    <h1>vLLM Profiling Report</h1>
    <p>Generated: {timestamp}</p>

    <h2>Executive Summary</h2>
    <div>
        <div class="metric">
            <div class="metric-value">{prefill_pct:.1f}%</div>
            <div class="metric-label">Prefill Time</div>
        </div>
        <div class="metric">
            <div class="metric-value">{decode_pct:.1f}%</div>
            <div class="metric-label">Decode Time</div>
        </div>
        <div class="metric">
            <div class="metric-value">{cuda_graph_coverage:.1f}%</div>
            <div class="metric-label">CUDA Graph Coverage</div>
        </div>
        <div class="metric">
            <div class="metric-value">{avg_bandwidth:.1f} GB/s</div>
            <div class="metric-label">Avg Bandwidth</div>
        </div>
    </div>

    <h2>Prefill vs Decode Breakdown</h2>
    <table>
        <tr>
            <th>Phase</th>
            <th>Count</th>
            <th>Total Time (ms)</th>
            <th>Avg Time (ms)</th>
            <th>Percentage</th>
        </tr>
        {prefill_decode_table}
    </table>

    <h2>Component Time Breakdown</h2>
    <table>
        <tr>
            <th>Component</th>
            <th>Total Time (ms)</th>
            <th>Percentage</th>
        </tr>
        {component_table}
    </table>

    <h2>Top Kernels by Time</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Kernel Name</th>
            <th>Invocations</th>
            <th>Total Time (ms)</th>
            <th>Avg Time (μs)</th>
        </tr>
        {kernel_table}
    </table>

    <h2>Bandwidth Analysis</h2>
    <table>
        <tr>
            <th>Kernel Name</th>
            <th>Bandwidth (GB/s)</th>
            <th>DRAM Util (%)</th>
            <th>SM Util (%)</th>
            <th>Bottleneck</th>
        </tr>
        {bandwidth_table}
    </table>

    <h2>Performance Recommendations</h2>
    <ul>
        {recommendations}
    </ul>

    <h2>Raw Data Files</h2>
    <ul>
        <li>Nsys results: {nsys_dir}</li>
        <li>NCU results: {ncu_dir}</li>
    </ul>
</body>
</html>
"""


def generate_recommendations(nsys_summary, ncu_summary) -> list[str]:
    """Generate performance recommendations based on profiling data."""
    recommendations = []

    # Check CUDA graph coverage
    cuda_graph_coverage = nsys_summary.get('cuda_graphs', {}).get('graph_coverage_pct', 0)
    if cuda_graph_coverage < 70:
        recommendations.append(
            f"<li class='warning'>CUDA graph coverage is {cuda_graph_coverage:.1f}%. "
            f"Consider increasing coverage for better performance.</li>"
        )

    # Check prefill/decode balance
    prefill_decode = nsys_summary.get('prefill_decode', {})
    total_time = sum(phase['total_ms'] for phase in prefill_decode.values())
    prefill_pct = prefill_decode.get('prefill', {}).get('total_ms', 0) / total_time * 100 if total_time > 0 else 0

    if prefill_pct > 60:
        recommendations.append(
            f"<li class='warning'>Prefill takes {prefill_pct:.1f}% of time. "
            f"Consider chunked prefill or increasing max-num-batched-tokens.</li>"
        )

    # Check bandwidth utilization
    if ncu_summary and 'top_kernels_by_bandwidth' in ncu_summary:
        avg_dram_util = sum(
            k.get('dram_throughput_pct', 0)
            for k in ncu_summary['top_kernels_by_bandwidth'][:10]
        ) / 10

        if avg_dram_util < 50:
            recommendations.append(
                f"<li class='warning'>Average DRAM utilization is {avg_dram_util:.1f}%. "
                f"Workload is compute-bound, but may benefit from better batching.</li>"
            )
        elif avg_dram_util > 85:
            recommendations.append(
                f"<li class='warning'>Average DRAM utilization is {avg_dram_util:.1f}%. "
                f"Workload is memory-bound. Consider quantization or larger batch sizes.</li>"
            )

    # Component-specific recommendations
    components = nsys_summary.get('components', {})
    attn_pct = components.get('attention', {}).get('percent', 0)
    ffn_pct = components.get('ffn', {}).get('percent', 0)

    if attn_pct > 60:
        recommendations.append(
            f"<li>Attention takes {attn_pct:.1f}% of time. "
            f"Flash Attention or other optimizations may help.</li>"
        )

    if not recommendations:
        recommendations.append("<li class='good'>No major performance issues detected!</li>")

    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive profiling report'
    )
    parser.add_argument('--nsys-results', required=True,
                        help='Directory with nsys results')
    parser.add_argument('--ncu-results',
                        help='Directory with ncu results (optional)')
    parser.add_argument('--output', default='profile_report.html',
                        help='Output HTML file')

    args = parser.parse_args()

    nsys_dir = Path(args.nsys_results)
    ncu_dir = Path(args.ncu_results) if args.ncu_results else None

    # Load nsys summary
    with open(nsys_dir / 'summary.json') as f:
        nsys_summary = json.load(f)

    # Load ncu summary if available
    ncu_summary = None
    if ncu_dir and (ncu_dir / 'ncu_summary.json').exists():
        with open(ncu_dir / 'ncu_summary.json') as f:
            ncu_summary = json.load(f)

    # Calculate metrics
    prefill_decode = nsys_summary['prefill_decode']
    total_time = sum(phase['total_ms'] for phase in prefill_decode.values())

    prefill_pct = prefill_decode['prefill']['total_ms'] / total_time * 100 if total_time > 0 else 0
    decode_pct = prefill_decode['decode']['total_ms'] / total_time * 100 if total_time > 0 else 0

    cuda_graph_coverage = nsys_summary['cuda_graphs']['graph_coverage_pct']

    avg_bandwidth = 0
    if ncu_summary and 'top_kernels_by_bandwidth' in ncu_summary:
        avg_bandwidth = sum(
            k.get('bandwidth_gbps', 0)
            for k in ncu_summary['top_kernels_by_bandwidth'][:10]
        ) / 10

    # Generate tables
    prefill_decode_rows = ""
    for phase, stats in prefill_decode.items():
        pct = stats['total_ms'] / total_time * 100 if total_time > 0 else 0
        prefill_decode_rows += f"""
        <tr>
            <td>{phase.capitalize()}</td>
            <td>{stats['count']}</td>
            <td>{stats['total_ms']:.2f}</td>
            <td>{stats['avg_ms']:.2f}</td>
            <td>{pct:.1f}%</td>
        </tr>
        """

    component_rows = ""
    for component, stats in sorted(
        nsys_summary['components'].items(),
        key=lambda x: x[1]['total_ms'],
        reverse=True
    ):
        component_rows += f"""
        <tr>
            <td>{component.capitalize()}</td>
            <td>{stats['total_ms']:.2f}</td>
            <td>{stats['percent']:.1f}%</td>
        </tr>
        """

    kernel_rows = ""
    for i, kernel in enumerate(nsys_summary['top_kernels'][:10], 1):
        kernel_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{kernel['kernel_name'][:80]}</td>
            <td>{kernel['invocations']}</td>
            <td>{kernel['total_time_ms']:.2f}</td>
            <td>{kernel['avg_time_us']:.2f}</td>
        </tr>
        """

    bandwidth_rows = ""
    if ncu_summary and 'top_kernels_by_bandwidth' in ncu_summary:
        for kernel in ncu_summary['top_kernels_by_bandwidth'][:10]:
            bottleneck_class = 'warning' if kernel.get('bottleneck') == 'memory' else 'good'
            bandwidth_rows += f"""
            <tr>
                <td>{kernel['kernel_name'][:80]}</td>
                <td>{kernel.get('bandwidth_gbps', 0):.2f}</td>
                <td>{kernel.get('dram_throughput_pct', 0):.1f}</td>
                <td>{kernel.get('sm_throughput_pct', 0):.1f}</td>
                <td class='{bottleneck_class}'>{kernel.get('bottleneck', 'unknown').upper()}</td>
            </tr>
            """

    # Generate recommendations
    recommendations = generate_recommendations(nsys_summary, ncu_summary)

    # Generate HTML
    html = HTML_TEMPLATE.format(
        timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        prefill_pct=prefill_pct,
        decode_pct=decode_pct,
        cuda_graph_coverage=cuda_graph_coverage,
        avg_bandwidth=avg_bandwidth,
        prefill_decode_table=prefill_decode_rows,
        component_table=component_rows,
        kernel_table=kernel_rows,
        bandwidth_table=bandwidth_rows if bandwidth_rows else "<tr><td colspan='5'>NCU data not available</td></tr>",
        recommendations='\n'.join(recommendations),
        nsys_dir=nsys_dir,
        ncu_dir=ncu_dir if ncu_dir else 'Not provided',
    )

    # Write output
    with open(args.output, 'w') as f:
        f.write(html)

    print(f"Report generated: {args.output}")


if __name__ == '__main__':
    main()
```

*This is getting long - I'll continue in a follow-up file with the automation scripts and integration plan.*
