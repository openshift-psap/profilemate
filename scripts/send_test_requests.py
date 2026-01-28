#!/usr/bin/env python3
"""
Send test requests to vLLM server for profiling.

Usage:
    python send_test_requests.py --num-requests 100 --port 8000
"""

import argparse
import requests
import time
import torch
from typing import List


def wait_for_server(port: int, timeout: int = 60) -> bool:
    """Wait for vLLM server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f'http://localhost:{port}/health', timeout=1)
            if response.status_code == 200:
                print(f"Server ready on port {port}")
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)

    print(f"Server failed to start within {timeout}s")
    return False


def generate_test_prompts(num_requests: int, mode: str = 'mixed') -> List[dict]:
    """Generate test prompts with varying lengths."""
    prompts = []

    if mode == 'prefill':
        # Long prompts for prefill-heavy workload
        for i in range(num_requests):
            prompts.append({
                'prompt': f"Explain the history of artificial intelligence in detail. " * 20,
                'max_tokens': 50,
            })

    elif mode == 'decode':
        # Short prompts for decode-heavy workload
        for i in range(num_requests):
            prompts.append({
                'prompt': f"Hello world {i}",
                'max_tokens': 200,
            })

    elif mode == 'mixed':
        # Mix of short and long prompts
        for i in range(num_requests):
            if i % 3 == 0:
                # Long prefill
                prompt = f"Write a detailed essay about climate change. " * 15
                max_tokens = 100
            elif i % 3 == 1:
                # Medium
                prompt = f"Explain quantum computing. " * 5
                max_tokens = 150
            else:
                # Short
                prompt = f"Hello, how are you?"
                max_tokens = 50

            prompts.append({
                'prompt': prompt,
                'max_tokens': max_tokens,
            })

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return prompts


def send_requests(
    port: int,
    num_requests: int,
    mode: str = 'mixed',
    trigger_profiler: bool = True,
    verbose: bool = True,
):
    """Send test requests to vLLM server."""

    # Wait for server
    if not wait_for_server(port):
        return

    # Generate prompts
    prompts = generate_test_prompts(num_requests, mode)

    # Trigger CUDA profiler if requested
    if trigger_profiler:
        try:
            torch.cuda.cudart().cudaProfilerStart()
            print("CUDA profiler started")
        except Exception as e:
            print(f"Warning: Could not start profiler: {e}")

    # Send requests
    print(f"Sending {num_requests} requests in {mode} mode...")
    start_time = time.time()
    successful = 0
    failed = 0

    for i, prompt_config in enumerate(prompts):
        try:
            response = requests.post(
                f'http://localhost:{port}/v1/completions',
                json={
                    'prompt': prompt_config['prompt'],
                    'max_tokens': prompt_config['max_tokens'],
                    'temperature': 0.8,
                },
                timeout=120,
            )

            if response.status_code == 200:
                successful += 1
                if verbose and (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{num_requests} requests")
            else:
                failed += 1
                if verbose:
                    print(f"Request {i + 1} failed with status {response.status_code}")

        except Exception as e:
            failed += 1
            if verbose:
                print(f"Request {i + 1} error: {e}")

    elapsed = time.time() - start_time

    # Stop profiler
    if trigger_profiler:
        try:
            torch.cuda.cudart().cudaProfilerStop()
            print("CUDA profiler stopped")
        except Exception as e:
            print(f"Warning: Could not stop profiler: {e}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {successful / elapsed:.2f} req/s")


def main():
    parser = argparse.ArgumentParser(
        description='Send test requests to vLLM server for profiling'
    )
    parser.add_argument('--num-requests', type=int, default=100,
                        help='Number of requests to send')
    parser.add_argument('--port', type=int, default=8000,
                        help='vLLM server port')
    parser.add_argument('--mode', choices=['prefill', 'decode', 'mixed'],
                        default='mixed',
                        help='Request mode: prefill (long prompts), decode (short prompts), mixed')
    parser.add_argument('--no-profiler', action='store_true',
                        help='Do not trigger CUDA profiler')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    send_requests(
        port=args.port,
        num_requests=args.num_requests,
        mode=args.mode,
        trigger_profiler=not args.no_profiler,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
