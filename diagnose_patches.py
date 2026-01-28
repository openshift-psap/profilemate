#!/usr/bin/env python3
"""
ProfileMate Patch Activation Diagnostic Tool

This script helps diagnose why patches aren't activating during vLLM runs.
Run this BEFORE starting vLLM to check your configuration.
"""

import os
import sys
import importlib.util
from pathlib import Path


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_pythonpath():
    """Check if ProfileMate is in PYTHONPATH."""
    print_header("1. PYTHONPATH Configuration")

    pythonpath = os.environ.get('PYTHONPATH', '')
    print(f"PYTHONPATH = {pythonpath}")

    if not pythonpath:
        print("❌ PYTHONPATH is not set!")
        print("\nFIX: Add this to your shell startup file (~/.bashrc or ~/.zshrc):")
        print(f"export PYTHONPATH=\"{Path(__file__).parent}:$PYTHONPATH\"")
        return False

    profilemate_in_path = any('profilemate' in p for p in pythonpath.split(':'))
    if profilemate_in_path:
        print("✅ ProfileMate directory appears in PYTHONPATH")
    else:
        print("⚠️  'profilemate' not found in PYTHONPATH")
        print("\nFIX: Add ProfileMate to PYTHONPATH:")
        print(f"export PYTHONPATH=\"{Path(__file__).parent}:$PYTHONPATH\"")
        return False

    # Check if sitecustomize.py is findable
    for path in pythonpath.split(':'):
        sitecustomize_path = Path(path) / 'sitecustomize.py'
        if sitecustomize_path.exists():
            print(f"✅ Found sitecustomize.py at: {sitecustomize_path}")
            return True

    print("❌ sitecustomize.py not found in any PYTHONPATH directory")
    return False


def check_activation_env():
    """Check activation environment variable."""
    print_header("2. Activation Environment Variable")

    vllm_enable = os.environ.get('VLLM_ENABLE_PROFILING')
    if vllm_enable is None:
        print("VLLM_ENABLE_PROFILING = <not set> (will auto-detect)")
        print("✅ Auto-detection will activate for vLLM commands")
        print("\nTo force enable:")
        print("export VLLM_ENABLE_PROFILING=1")
        print("\nTo force disable:")
        print("export VLLM_ENABLE_PROFILING=0")
    elif vllm_enable == "1":
        print("VLLM_ENABLE_PROFILING = 1")
        print("✅ Profiling will ALWAYS activate")
    elif vllm_enable == "0":
        print("VLLM_ENABLE_PROFILING = 0")
        print("❌ Profiling is DISABLED")
        print("\nFIX: Remove or change the environment variable:")
        print("unset VLLM_ENABLE_PROFILING")
    else:
        print(f"VLLM_ENABLE_PROFILING = {vllm_enable}")
        print("⚠️  Invalid value (should be '0' or '1')")


def check_vllm_installation():
    """Check if vLLM is installed."""
    print_header("3. vLLM Installation")

    try:
        spec = importlib.util.find_spec('vllm')
        if spec is None:
            print("❌ vLLM is not installed")
            print("\nFIX: Install vLLM:")
            print("pip install vllm")
            return False
        else:
            print(f"✅ vLLM is installed at: {spec.origin}")

            # Try to get vLLM version
            try:
                import vllm
                if hasattr(vllm, '__version__'):
                    print(f"   Version: {vllm.__version__}")
            except Exception as e:
                print(f"   (Could not determine version: {e})")

            return True
    except ImportError:
        print("❌ vLLM is not installed")
        return False


def check_output_directory():
    """Check if output directory is writable."""
    print_header("4. Output Directory")

    # Default output directory (from ProfilingConfig)
    output_dir = Path("/tmp/vllm_profiling")

    print(f"Default output directory: {output_dir}")

    if output_dir.exists():
        print(f"✅ Directory exists")
        if os.access(output_dir, os.W_OK):
            print(f"✅ Directory is writable")
        else:
            print(f"❌ Directory is not writable")
            print(f"\nFIX: Make directory writable:")
            print(f"sudo chmod a+w {output_dir}")
    else:
        print(f"⚠️  Directory does not exist (will be created on first run)")
        parent = output_dir.parent
        if os.access(parent, os.W_OK):
            print(f"✅ Parent directory {parent} is writable (can create)")
        else:
            print(f"❌ Cannot create directory (parent not writable)")
            print(f"\nFIX: Create directory manually:")
            print(f"mkdir -p {output_dir}")


def simulate_activation_check():
    """Simulate the should_activate_profiling() logic."""
    print_header("5. Activation Simulation")

    print("Simulating activation logic with current command line:")
    print(f"sys.argv = {sys.argv}")

    # Priority 1: Explicit enable
    explicit_enable = os.getenv("VLLM_ENABLE_PROFILING")
    if explicit_enable is not None:
        if explicit_enable == "1":
            print("\n✅ WILL ACTIVATE (VLLM_ENABLE_PROFILING=1)")
            return True
        else:
            print("\n❌ WILL NOT ACTIVATE (VLLM_ENABLE_PROFILING=0)")
            return False

    # Priority 2: Auto-detect from command line
    cmdline = ' '.join(sys.argv)
    vllm_indicators = ['vllm.entrypoints', 'vllm.engine', 'vllm', '--model']

    print("\nChecking for vLLM indicators in command line:")
    found_indicators = []
    for indicator in vllm_indicators:
        if indicator in cmdline:
            print(f"  ✅ Found: '{indicator}'")
            found_indicators.append(indicator)
        else:
            print(f"  ❌ Not found: '{indicator}'")

    if found_indicators:
        print(f"\n✅ WILL ACTIVATE (found indicators: {found_indicators})")
        return True

    # Priority 3: Check vLLM installation
    try:
        spec = importlib.util.find_spec('vllm')
        if spec is None:
            print("\n❌ WILL NOT ACTIVATE (vLLM not installed)")
            return False
    except ImportError:
        print("\n❌ WILL NOT ACTIVATE (vLLM not installed)")
        return False

    print("\n❌ WILL NOT ACTIVATE (no vLLM indicators found)")
    print("\nNOTE: This is a diagnostic script, not a vLLM command.")
    print("Activation logic will behave differently with an actual vLLM command.")
    return False


def check_vllm_module_paths():
    """Check if target vLLM modules exist."""
    print_header("6. Target vLLM Module Availability")

    try:
        import vllm
    except ImportError:
        print("❌ vLLM not installed, skipping module checks")
        return

    target_modules = {
        'vllm.compilation.cuda_graph': 'CUDA Graph Wrapper',
        'vllm.v1.core.kv_cache_manager': 'KV Cache Manager',
        'vllm.v1.core.sched.scheduler': 'Scheduler (new path)',
        'vllm.v1.core.scheduler': 'Scheduler (old path)',
        'vllm.v1.worker.gpu_model_runner': 'GPU Model Runner',
        'vllm.model_executor.layers.fused_moe.layer': 'Fused MoE Layer',
        'vllm.v1.core.block_pool': 'Block Pool',
    }

    print("Checking which target modules exist in your vLLM version:\n")

    available_count = 0
    for module_path, description in target_modules.items():
        try:
            spec = importlib.util.find_spec(module_path)
            if spec is not None:
                print(f"✅ {description:<30} ({module_path})")
                available_count += 1
            else:
                print(f"❌ {description:<30} ({module_path})")
        except (ImportError, ModuleNotFoundError, ValueError):
            print(f"❌ {description:<30} ({module_path})")

    print(f"\n{available_count}/{len(target_modules)} target modules available")

    if available_count == 0:
        print("\n⚠️  WARNING: No target modules found!")
        print("This could indicate:")
        print("  1. Your vLLM version has different module paths")
        print("  2. vLLM installation is incomplete")
        print("  3. ProfileMate needs updating for your vLLM version")


def provide_test_command():
    """Provide a test command to verify activation."""
    print_header("7. Test Command")

    print("To test if ProfileMate activates correctly, run:\n")
    print("VLLM_ENABLE_PROFILING=1 python -c \"import sys; print('Args:', sys.argv)\"")
    print("\nYou should see:")
    print("  [sitecustomize] vLLM Comprehensive Instrumentation Loaded")
    print("  Session ID: YYYYMMDD_HHMMSS")
    print("  Output directory: /tmp/vllm_profiling/session_YYYYMMDD_HHMMSS")
    print("\nThen run your actual vLLM command:")
    print("python -m vllm.entrypoints.openai.api_server --model <your-model> 2>&1 | grep sitecustomize")


def main():
    """Run all diagnostic checks."""
    print("\n" + "█" * 70)
    print("  ProfileMate Patch Activation Diagnostic Tool")
    print("█" * 70)

    results = {
        'pythonpath': check_pythonpath(),
        'vllm': check_vllm_installation(),
    }

    check_activation_env()
    check_output_directory()
    simulate_activation_check()

    if results['vllm']:
        check_vllm_module_paths()

    provide_test_command()

    # Summary
    print_header("Summary")

    if all(results.values()):
        print("✅ All critical checks passed!")
        print("\nProfileMate should activate when you run vLLM.")
        print("\nTo verify, run your vLLM command and check for:")
        print("  [sitecustomize] vLLM Comprehensive Instrumentation Loaded")
    else:
        print("❌ Some checks failed. Please address the issues above.")
        print("\nMost common fixes:")
        print("  1. Set PYTHONPATH:")
        print(f"     export PYTHONPATH=\"{Path(__file__).parent}:$PYTHONPATH\"")
        print("  2. Install vLLM:")
        print("     pip install vllm")
        print("  3. Force enable profiling:")
        print("     export VLLM_ENABLE_PROFILING=1")


if __name__ == "__main__":
    main()
