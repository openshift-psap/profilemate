# VLLM Log Analyzer

This tool parses vLLM server logs to extract and analyze iteration details, request lifecycle, and performance metrics.

## Setup

The tool uses `uv` for environment management. The environment has already been created at `.venv/`.

To activate the environment:
```bash
source .venv/bin/activate
```

## Usage

### Basic Usage

```bash
python parse_vllm_log.py vllm_server.log --output-dir ./analysis_output
```

### With Iteration Range Filter

To analyze only a specific range of iterations:

```bash
python parse_vllm_log.py vllm_server.log --iteration-range 0 100 --output-dir ./analysis_output
```

### Command Line Options

- `log_file`: Path to the vLLM server log file (required)
- `--iteration-range START END`: Filter iterations by range (optional)
- `--output-dir DIR`: Output directory for plots and summary (default: current directory)
- `--summary-file FILE`: Output file name for summary (default: summary.txt)

## Output Files

The tool generates the following visualization files (each as a separate PNG):

1. **01_iteration_elapsed_time.png**: Iteration elapsed time by type (scatter plot)
2. **02_context_vs_generation_tokens.png**: Context vs Generation tokens scatter plot (colored by elapsed time)
3. **03_request_counts.png**: Request counts over iterations
4. **04_token_counts.png**: Token counts over iterations
5. **05_iteration_type_distribution.png**: Iteration type distribution (bar chart)
6. **06_decode_only_tokens.png**: Decode-only iterations: decode tokens vs iteration number
7. **07_prefill_only_tokens.png**: Prefill-only iterations: prefill tokens vs iteration number
8. **08_mixed_tokens.png**: Mixed iterations: both prefill and decode tokens vs iteration number
9. **09_throughput_timeline.png**: Average prompt/s and decode/s over timeline
10. **10_request_timeline.png**: Timeline visualization showing request queue and processing times (if available)

11. **summary.txt**: Text summary with:
   - Iteration statistics
   - Performance metrics
   - Token statistics
   - Request statistics
   - Performance breakdown by iteration type
   - Request lifecycle statistics

12. **iterations.csv**: CSV file with all parsed iteration data

13. **requests.csv**: CSV file with all parsed request data

## Iteration Classification

Iterations are automatically classified into four types:

- **prefill-only**: Has context requests but no generation requests (colored blue)
- **decode-only**: Has generation requests but no context requests (colored green)
- **mixed**: Has both context and generation requests (colored orange)
- **idle**: Has neither context nor generation requests (colored gray)

The same consistent colors are used across all visualizations for easy identification.

## Features

- Parses iteration details (context/generation requests, tokens, elapsed time)
- Tracks request lifecycle (received, added, completed)
- Classifies iterations by type
- Generates comprehensive visualizations
- Creates detailed summary reports
- Exports data to CSV for further analysis

## Example Output

The summary includes:
- Total iterations and time range
- Iteration type breakdown with percentages
- Performance metrics (average, median, min, max, std dev)
- Token statistics (total, average, max)
- Request statistics
- Performance breakdown by iteration type
- Request lifecycle timing (queue time, processing time, total time)
