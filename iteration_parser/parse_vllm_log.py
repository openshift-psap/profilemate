#!/usr/bin/env python3
"""
VLLM Server Log Parser and Analyzer

This script parses vLLM server logs to extract:
- Iteration details (context requests, generation requests, tokens, elapsed time)
- Request lifecycle (received, added, completed)
- Classifies iterations as prefill-only, mixed, or decode-only
- Generates plots and timeline visualizations
- Creates a summary report
"""

import re
import argparse
from datetime import datetime
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path


class VLLMLogParser:
    def __init__(self, log_file):
        self.log_file = log_file
        self.iterations = []
        self.requests = {}
        self.request_added = {}
        self.request_completed = {}
        
    def parse_log(self, iteration_range=None):
        """Parse the log file and extract all relevant information."""
        print(f"Parsing log file: {self.log_file}")
        
        # Patterns for different log entries
        # Handle ANSI escape codes - they may appear as literal [0;36m or as escape sequences
        iteration_pattern = re.compile(
            r'.*?\(EngineCore_DP0 pid=\d+\).*?INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[core\.py:348\] '
            r'Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, '
            r'(\d+) generation requests, (\d+) generation tokens, iteration elapsed time: ([\d.]+) ms'
        )
        
        request_received_pattern = re.compile(
            r'.*?\(APIServer pid=\d+\).*?INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[logger\.py:49\] '
            r'Received request ([^:]+):'
        )
        
        request_added_pattern = re.compile(
            r'.*?\(APIServer pid=\d+\).*?INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[async_llm\.py:419\] '
            r'Added request ([^\s]+)\.'
        )
        
        request_completed_pattern = re.compile(
            r'.*?\(APIServer pid=\d+\).*?INFO:.*"POST /v1/completions HTTP/1.1" 200 OK'
        )
        
        # Track the last timestamp for completion (since HTTP logs don't have request IDs)
        last_completion_time = None
        
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Parse iteration
                match = iteration_pattern.search(line)
                if match:
                    timestamp_str, iter_num, ctx_req, ctx_tokens, gen_req, gen_tokens, elapsed = match.groups()
                    iter_num = int(iter_num)
                    
                    # Apply iteration range filter if specified
                    if iteration_range:
                        start, end = iteration_range
                        if iter_num < start or iter_num > end:
                            continue
                    
                    # Parse timestamp (assuming format: MM-DD HH:MM:SS)
                    # We'll use a reference year since it's not in the log
                    timestamp = datetime.strptime(f"2026-{timestamp_str}", "%Y-%m-%d %H:%M:%S")
                    
                    iteration = {
                        'iteration': iter_num,
                        'timestamp': timestamp,
                        'context_requests': int(ctx_req),
                        'context_tokens': int(ctx_tokens),
                        'generation_requests': int(gen_req),
                        'generation_tokens': int(gen_tokens),
                        'elapsed_time_ms': float(elapsed),
                        'line_number': line_num
                    }
                    
                    # Classify iteration type
                    if iteration['context_requests'] > 0 and iteration['generation_requests'] > 0:
                        iteration['type'] = 'mixed'
                    elif iteration['context_requests'] > 0:
                        iteration['type'] = 'prefill-only'
                    elif iteration['generation_requests'] > 0:
                        iteration['type'] = 'decode-only'
                    else:
                        iteration['type'] = 'idle'
                    
                    self.iterations.append(iteration)
                    continue
                
                # Parse request received
                match = request_received_pattern.search(line)
                if match:
                    timestamp_str, request_id = match.groups()
                    timestamp = datetime.strptime(f"2026-{timestamp_str}", "%Y-%m-%d %H:%M:%S")
                    
                    if request_id not in self.requests:
                        self.requests[request_id] = {}
                    self.requests[request_id]['received_time'] = timestamp
                    self.requests[request_id]['request_id'] = request_id
                    continue
                
                # Parse request added
                match = request_added_pattern.search(line)
                if match:
                    timestamp_str, request_id = match.groups()
                    timestamp = datetime.strptime(f"2026-{timestamp_str}", "%Y-%m-%d %H:%M:%S")
                    
                    if request_id not in self.requests:
                        self.requests[request_id] = {}
                    self.requests[request_id]['added_time'] = timestamp
                    self.requests[request_id]['request_id'] = request_id
                    self.request_added[request_id] = timestamp
                    continue
                
                # Parse request completion (HTTP 200 OK)
                if request_completed_pattern.search(line):
                    # Extract timestamp from the line
                    time_match = re.search(r'(\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if time_match:
                        timestamp_str = time_match.group(1)
                        timestamp = datetime.strptime(f"2026-{timestamp_str}", "%Y-%m-%d %H:%M:%S")
                        # Store completion times - we'll match them to requests later
                        if not hasattr(self, 'completion_times'):
                            self.completion_times = []
                        self.completion_times.append(timestamp)
                        continue
        
        # Match completions to requests (approximate - matches oldest uncompleted request to each completion)
        if hasattr(self, 'completion_times') and self.completion_times:
            # Sort requests by added_time
            uncompleted_requests = [
                (req_id, req_data) 
                for req_id, req_data in self.requests.items()
                if 'added_time' in req_data and 'completed_time' not in req_data
            ]
            uncompleted_requests.sort(key=lambda x: x[1]['added_time'])
            
            # Match each completion to the oldest uncompleted request at that time
            for completion_time in sorted(self.completion_times):
                for req_id, req_data in uncompleted_requests:
                    if 'completed_time' not in req_data and req_data['added_time'] <= completion_time:
                        req_data['completed_time'] = completion_time
                        break
        
        print(f"Parsed {len(self.iterations)} iterations")
        print(f"Parsed {len(self.requests)} requests")
        
        return self
    
    def get_dataframe(self):
        """Convert parsed data to pandas DataFrames."""
        if not self.iterations:
            return None, None
        
        iter_df = pd.DataFrame(self.iterations)
        iter_df = iter_df.sort_values('iteration')
        
        # Create requests DataFrame
        requests_list = []
        for req_id, req_data in self.requests.items():
            requests_list.append({
                'request_id': req_id,
                'received_time': req_data.get('received_time'),
                'added_time': req_data.get('added_time'),
                'completed_time': req_data.get('completed_time'),
                'queue_time_ms': None,
                'processing_time_ms': None,
                'total_time_ms': None
            })
            
            # Calculate timing metrics
            if req_data.get('received_time') and req_data.get('added_time'):
                queue_time = (req_data['added_time'] - req_data['received_time']).total_seconds() * 1000
                requests_list[-1]['queue_time_ms'] = queue_time
            
            if req_data.get('added_time') and req_data.get('completed_time'):
                processing_time = (req_data['completed_time'] - req_data['added_time']).total_seconds() * 1000
                requests_list[-1]['processing_time_ms'] = processing_time
            
            if req_data.get('received_time') and req_data.get('completed_time'):
                total_time = (req_data['completed_time'] - req_data['received_time']).total_seconds() * 1000
                requests_list[-1]['total_time_ms'] = total_time
        
        req_df = pd.DataFrame(requests_list)
        
        return iter_df, req_df
    
    def create_plots(self, output_dir='.'):
        """Create visualization plots - each in a separate PNG file."""
        iter_df, req_df = self.get_dataframe()
        
        if iter_df is None or len(iter_df) == 0:
            print("No iteration data to plot")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('default')
        
        # Constant colors for iteration types
        COLOR_PREFILL_ONLY = '#1f77b4'  # blue
        COLOR_DECODE_ONLY = '#2ca02c'   # green
        COLOR_MIXED = '#ff7f0e'         # orange
        COLOR_IDLE = '#7f7f7f'          # gray
        
        # Prepare time data
        start_time = iter_df['timestamp'].min()
        iter_df = iter_df.copy()
        iter_df['time_seconds'] = (iter_df['timestamp'] - start_time).dt.total_seconds()
        
        # 1. Iteration elapsed time over iterations
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = iter_df['type'].map({
            'prefill-only': COLOR_PREFILL_ONLY,
            'decode-only': COLOR_DECODE_ONLY,
            'mixed': COLOR_MIXED,
            'idle': COLOR_IDLE
        })
        ax.scatter(iter_df['iteration'], iter_df['elapsed_time_ms'], c=colors, alpha=0.6, s=20)
        ax.set_xlabel('Iteration Number', fontsize=12)
        ax.set_ylabel('Elapsed Time (ms)', fontsize=12)
        ax.set_title('Iteration Elapsed Time by Type', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_PREFILL_ONLY, markersize=8, label='Prefill-only'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_DECODE_ONLY, markersize=8, label='Decode-only'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MIXED, markersize=8, label='Mixed'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_IDLE, markersize=8, label='Idle')
        ], loc='upper right')
        plt.tight_layout()
        plt.savefig(output_dir / '01_iteration_elapsed_time.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_dir / '01_iteration_elapsed_time.png'}")
        plt.close()
        
        # 2. Context vs Generation tokens
        fig, ax = plt.subplots(figsize=(14, 8))
        scatter = ax.scatter(iter_df['context_tokens'], iter_df['generation_tokens'], 
                  c=iter_df['elapsed_time_ms'], cmap='viridis', alpha=0.6, s=30)
        ax.set_xlabel('Context Tokens', fontsize=12)
        ax.set_ylabel('Generation Tokens', fontsize=12)
        ax.set_title('Context vs Generation Tokens (colored by elapsed time)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Elapsed Time (ms)')
        plt.tight_layout()
        plt.savefig(output_dir / '02_context_vs_generation_tokens.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_dir / '02_context_vs_generation_tokens.png'}")
        plt.close()
        
        # 3. Request counts over iterations
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(iter_df['iteration'], iter_df['context_requests'], label='Context Requests', marker='o', markersize=3, linewidth=1.5)
        ax.plot(iter_df['iteration'], iter_df['generation_requests'], label='Generation Requests', marker='s', markersize=3, linewidth=1.5)
        ax.set_xlabel('Iteration Number', fontsize=12)
        ax.set_ylabel('Number of Requests', fontsize=12)
        ax.set_title('Request Counts Over Iterations', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '03_request_counts.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_dir / '03_request_counts.png'}")
        plt.close()
        
        # 4. Token counts over iterations
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(iter_df['iteration'], iter_df['context_tokens'], label='Context Tokens', marker='o', markersize=3, linewidth=1.5)
        ax.plot(iter_df['iteration'], iter_df['generation_tokens'], label='Generation Tokens', marker='s', markersize=3, linewidth=1.5)
        ax.set_xlabel('Iteration Number', fontsize=12)
        ax.set_ylabel('Number of Tokens', fontsize=12)
        ax.set_title('Token Counts Over Iterations', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '04_token_counts.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_dir / '04_token_counts.png'}")
        plt.close()
        
        # 5. Iteration type distribution
        fig, ax = plt.subplots(figsize=(10, 8))
        type_counts = iter_df['type'].value_counts()
        color_map = {
            'prefill-only': COLOR_PREFILL_ONLY,
            'decode-only': COLOR_DECODE_ONLY,
            'mixed': COLOR_MIXED,
            'idle': COLOR_IDLE
        }
        colors_bar = [color_map.get(itype, COLOR_IDLE) for itype in type_counts.index]
        ax.bar(type_counts.index, type_counts.values, color=colors_bar)
        ax.set_xlabel('Iteration Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Iteration Type Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / '05_iteration_type_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_dir / '05_iteration_type_distribution.png'}")
        plt.close()
        
        # 6. Decode-only iterations: iteration number vs decode tokens
        decode_df = iter_df[iter_df['type'] == 'decode-only'].copy()
        if len(decode_df) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.scatter(decode_df['iteration'], decode_df['generation_tokens'], 
                      c=COLOR_DECODE_ONLY, alpha=0.6, s=30)
            ax.set_xlabel('Iteration Number', fontsize=12)
            ax.set_ylabel('Decode Tokens', fontsize=12)
            ax.set_title('Decode-Only Iterations: Decode Tokens vs Iteration Number', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / '06_decode_only_tokens.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot to {output_dir / '06_decode_only_tokens.png'}")
            plt.close()
        
        # 7. Prefill-only iterations: iteration number vs prefill tokens
        prefill_df = iter_df[iter_df['type'] == 'prefill-only'].copy()
        if len(prefill_df) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.scatter(prefill_df['iteration'], prefill_df['context_tokens'], 
                      c=COLOR_PREFILL_ONLY, alpha=0.6, s=30)
            ax.set_xlabel('Iteration Number', fontsize=12)
            ax.set_ylabel('Prefill Tokens (Context Tokens)', fontsize=12)
            ax.set_title('Prefill-Only Iterations: Prefill Tokens vs Iteration Number', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / '07_prefill_only_tokens.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot to {output_dir / '07_prefill_only_tokens.png'}")
            plt.close()
        
        # 8. Mixed iterations: iteration number vs both prefill and decode tokens
        mixed_df = iter_df[iter_df['type'] == 'mixed'].copy()
        if len(mixed_df) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.scatter(mixed_df['iteration'], mixed_df['context_tokens'], 
                      c=COLOR_PREFILL_ONLY, alpha=0.6, s=30, label='Prefill Tokens', marker='o')
            ax.scatter(mixed_df['iteration'], mixed_df['generation_tokens'], 
                      c=COLOR_DECODE_ONLY, alpha=0.6, s=30, label='Decode Tokens', marker='s')
            ax.set_xlabel('Iteration Number', fontsize=12)
            ax.set_ylabel('Tokens', fontsize=12)
            ax.set_title('Mixed Iterations: Prefill and Decode Tokens vs Iteration Number', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / '08_mixed_tokens.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot to {output_dir / '08_mixed_tokens.png'}")
            plt.close()
        
        # 9. Throughput over timeline: prompt/s and decode/s
        self._plot_throughput(iter_df, output_dir, start_time)
        
        # Create request timeline plot if we have request data
        if req_df is not None and len(req_df) > 0:
            self._plot_request_timeline(req_df, output_dir)
    
    def _plot_throughput(self, iter_df, output_dir, start_time):
        """Plot average prompt/s and decode/s over timeline."""
        # Calculate throughput over time windows
        # Use a rolling window approach
        window_size_seconds = 1.0  # 1 second windows
        
        # Get time range
        max_time = iter_df['time_seconds'].max()
        time_bins = np.arange(0, max_time + window_size_seconds, window_size_seconds)
        
        prompt_throughput = []
        decode_throughput = []
        time_points = []
        
        for i in range(len(time_bins) - 1):
            bin_start = time_bins[i]
            bin_end = time_bins[i + 1]
            
            # Get iterations in this time window
            mask = (iter_df['time_seconds'] >= bin_start) & (iter_df['time_seconds'] < bin_end)
            window_df = iter_df[mask]
            
            if len(window_df) > 0:
                # Calculate total tokens processed in this window
                total_context_tokens = window_df['context_tokens'].sum()
                total_generation_tokens = window_df['generation_tokens'].sum()
                
                # Calculate throughput (tokens per second)
                window_duration = bin_end - bin_start
                if window_duration > 0:
                    prompt_tps = total_context_tokens / window_duration
                    decode_tps = total_generation_tokens / window_duration
                    
                    time_points.append((bin_start + bin_end) / 2)
                    prompt_throughput.append(prompt_tps)
                    decode_throughput.append(decode_tps)
        
        if len(time_points) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.plot(time_points, prompt_throughput, label='Prompt Throughput (tokens/s)', 
                   color='#1f77b4', linewidth=2, marker='o', markersize=4)
            ax.plot(time_points, decode_throughput, label='Decode Throughput (tokens/s)', 
                   color='#2ca02c', linewidth=2, marker='s', markersize=4)
            ax.set_xlabel('Time (seconds from start)', fontsize=12)
            ax.set_ylabel('Throughput (tokens/second)', fontsize=12)
            ax.set_title('Average Prompt/s and Decode/s Over Timeline', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / '09_throughput_timeline.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot to {output_dir / '09_throughput_timeline.png'}")
            plt.close()
    
    def _plot_request_timeline(self, req_df, output_dir):
        """Create a timeline plot for requests."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Filter requests with timing data
        req_df_filtered = req_df.dropna(subset=['received_time', 'added_time'])
        
        if len(req_df_filtered) == 0:
            print("No request timing data available for timeline plot")
            return
        
        # Normalize timestamps
        start_time = req_df_filtered['received_time'].min()
        req_df_filtered = req_df_filtered.copy()
        req_df_filtered['received_time_norm'] = (req_df_filtered['received_time'] - start_time).dt.total_seconds()
        req_df_filtered['added_time_norm'] = (req_df_filtered['added_time'] - start_time).dt.total_seconds()
        
        if 'completed_time' in req_df_filtered.columns:
            req_df_filtered['completed_time_norm'] = (req_df_filtered['completed_time'] - start_time).dt.total_seconds()
        
        # Plot request lifecycle
        y_pos = np.arange(len(req_df_filtered))
        
        for idx, row in req_df_filtered.iterrows():
            y = idx
            # Queue time (received to added)
            if pd.notna(row['received_time_norm']) and pd.notna(row['added_time_norm']):
                ax.plot([row['received_time_norm'], row['added_time_norm']], [y, y], 
                       'b-', linewidth=2, label='Queue' if idx == 0 else '')
            
            # Processing time (added to completed)
            if 'completed_time_norm' in row and pd.notna(row['completed_time_norm']):
                if pd.notna(row['added_time_norm']):
                    ax.plot([row['added_time_norm'], row['completed_time_norm']], [y, y], 
                           'g-', linewidth=2, label='Processing' if idx == 0 else '')
        
        ax.set_xlabel('Time (seconds from start)')
        ax.set_ylabel('Request Index')
        ax.set_title('Request Timeline (Queue and Processing)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / '10_request_timeline.png', dpi=300, bbox_inches='tight')
        print(f"Saved request timeline plot to {output_dir / '10_request_timeline.png'}")
        plt.close()
    
    def generate_summary(self, output_file='summary.txt'):
        """Generate a text summary of the analysis."""
        iter_df, req_df = self.get_dataframe()
        
        if iter_df is None or len(iter_df) == 0:
            print("No data to summarize")
            return
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("VLLM Server Log Analysis Summary")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        # Iteration statistics
        summary_lines.append("ITERATION STATISTICS")
        summary_lines.append("-" * 80)
        summary_lines.append(f"Total Iterations: {len(iter_df)}")
        summary_lines.append(f"Iteration Range: {iter_df['iteration'].min()} - {iter_df['iteration'].max()}")
        summary_lines.append(f"Time Range: {iter_df['timestamp'].min()} to {iter_df['timestamp'].max()}")
        summary_lines.append("")
        
        # Iteration type breakdown
        summary_lines.append("Iteration Type Breakdown:")
        type_counts = iter_df['type'].value_counts()
        for itype, count in type_counts.items():
            percentage = (count / len(iter_df)) * 100
            summary_lines.append(f"  {itype:20s}: {count:6d} ({percentage:5.2f}%)")
        summary_lines.append("")
        
        # Performance metrics
        summary_lines.append("PERFORMANCE METRICS")
        summary_lines.append("-" * 80)
        summary_lines.append(f"Average Elapsed Time: {iter_df['elapsed_time_ms'].mean():.2f} ms")
        summary_lines.append(f"Median Elapsed Time: {iter_df['elapsed_time_ms'].median():.2f} ms")
        summary_lines.append(f"Min Elapsed Time: {iter_df['elapsed_time_ms'].min():.2f} ms")
        summary_lines.append(f"Max Elapsed Time: {iter_df['elapsed_time_ms'].max():.2f} ms")
        summary_lines.append(f"Std Dev Elapsed Time: {iter_df['elapsed_time_ms'].std():.2f} ms")
        summary_lines.append("")
        
        # Token statistics
        summary_lines.append("TOKEN STATISTICS")
        summary_lines.append("-" * 80)
        summary_lines.append(f"Total Context Tokens: {iter_df['context_tokens'].sum():,}")
        summary_lines.append(f"Total Generation Tokens: {iter_df['generation_tokens'].sum():,}")
        summary_lines.append(f"Average Context Tokens per Iteration: {iter_df['context_tokens'].mean():.2f}")
        summary_lines.append(f"Average Generation Tokens per Iteration: {iter_df['generation_tokens'].mean():.2f}")
        summary_lines.append(f"Max Context Tokens in Single Iteration: {iter_df['context_tokens'].max():,}")
        summary_lines.append(f"Max Generation Tokens in Single Iteration: {iter_df['generation_tokens'].max():,}")
        summary_lines.append("")
        
        # Request statistics
        summary_lines.append(f"Total Context Requests: {iter_df['context_requests'].sum():,}")
        summary_lines.append(f"Total Generation Requests: {iter_df['generation_requests'].sum():,}")
        summary_lines.append(f"Average Context Requests per Iteration: {iter_df['context_requests'].mean():.2f}")
        summary_lines.append(f"Average Generation Requests per Iteration: {iter_df['generation_requests'].mean():.2f}")
        summary_lines.append("")
        
        # Performance by iteration type
        summary_lines.append("PERFORMANCE BY ITERATION TYPE")
        summary_lines.append("-" * 80)
        for itype in ['prefill-only', 'mixed', 'decode-only', 'idle']:
            type_df = iter_df[iter_df['type'] == itype]
            if len(type_df) > 0:
                summary_lines.append(f"\n{itype.upper()}:")
                summary_lines.append(f"  Count: {len(type_df)}")
                summary_lines.append(f"  Avg Elapsed Time: {type_df['elapsed_time_ms'].mean():.2f} ms")
                summary_lines.append(f"  Avg Context Tokens: {type_df['context_tokens'].mean():.2f}")
                summary_lines.append(f"  Avg Generation Tokens: {type_df['generation_tokens'].mean():.2f}")
        summary_lines.append("")
        
        # Request lifecycle statistics
        if req_df is not None and len(req_df) > 0:
            summary_lines.append("REQUEST LIFECYCLE STATISTICS")
            summary_lines.append("-" * 80)
            summary_lines.append(f"Total Requests Tracked: {len(req_df)}")
            
            queue_times = req_df['queue_time_ms'].dropna()
            if len(queue_times) > 0:
                summary_lines.append(f"\nQueue Time (Received -> Added):")
                summary_lines.append(f"  Average: {queue_times.mean():.2f} ms")
                summary_lines.append(f"  Median: {queue_times.median():.2f} ms")
                summary_lines.append(f"  Min: {queue_times.min():.2f} ms")
                summary_lines.append(f"  Max: {queue_times.max():.2f} ms")
            
            proc_times = req_df['processing_time_ms'].dropna()
            if len(proc_times) > 0:
                summary_lines.append(f"\nProcessing Time (Added -> Completed):")
                summary_lines.append(f"  Average: {proc_times.mean():.2f} ms")
                summary_lines.append(f"  Median: {proc_times.median():.2f} ms")
                summary_lines.append(f"  Min: {proc_times.min():.2f} ms")
                summary_lines.append(f"  Max: {proc_times.max():.2f} ms")
            
            total_times = req_df['total_time_ms'].dropna()
            if len(total_times) > 0:
                summary_lines.append(f"\nTotal Time (Received -> Completed):")
                summary_lines.append(f"  Average: {total_times.mean():.2f} ms")
                summary_lines.append(f"  Median: {total_times.median():.2f} ms")
                summary_lines.append(f"  Min: {total_times.min():.2f} ms")
                summary_lines.append(f"  Max: {total_times.max():.2f} ms")
            summary_lines.append("")
        
        summary_lines.append("=" * 80)
        
        summary_text = "\n".join(summary_lines)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(summary_text)
        
        # Also print to console
        print(summary_text)
        print(f"\nSummary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Parse and analyze vLLM server logs')
    parser.add_argument('log_file', help='Path to the vLLM server log file')
    parser.add_argument('--iteration-range', nargs=2, type=int, metavar=('START', 'END'),
                       help='Filter iterations by range (e.g., --iteration-range 0 100)')
    parser.add_argument('--output-dir', default='.', help='Output directory for plots and summary')
    parser.add_argument('--summary-file', default='summary.txt', help='Output file for summary')
    
    args = parser.parse_args()
    
    # Create parser and parse log
    log_parser = VLLMLogParser(args.log_file)
    iteration_range = tuple(args.iteration_range) if args.iteration_range else None
    log_parser.parse_log(iteration_range=iteration_range)
    
    # Generate outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    log_parser.create_plots(output_dir=output_dir)
    log_parser.generate_summary(output_file=str(output_dir / args.summary_file))
    
    # Save data to CSV
    iter_df, req_df = log_parser.get_dataframe()
    if iter_df is not None:
        iter_df.to_csv(output_dir / 'iterations.csv', index=False)
        print(f"Saved iterations data to {output_dir / 'iterations.csv'}")
    if req_df is not None and len(req_df) > 0:
        req_df.to_csv(output_dir / 'requests.csv', index=False)
        print(f"Saved requests data to {output_dir / 'requests.csv'}")


if __name__ == '__main__':
    main()
