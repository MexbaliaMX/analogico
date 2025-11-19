#!/usr/bin/env python3
"""
Simple benchmark runner for the Analogico project.

This script provides an easy way to run different benchmarks.
"""
import sys
import os
import argparse

# Add the project directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.benchmark import BenchmarkSuite, run_comprehensive_benchmarks


def main():
    parser = argparse.ArgumentParser(description='Analogico Benchmark Runner')
    parser.add_argument('--type', type=str, default='comprehensive',
                        choices=['comprehensive', 'scalability', 'rram_effects', 'hp_inv_vs_numpy'],
                        help='Type of benchmark to run')
    parser.add_argument('--sizes', type=int, nargs='+', default=[4, 8, 12],
                        help='Matrix sizes to benchmark (default: 4 8 12)')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials for each benchmark (default: 3)')
    
    args = parser.parse_args()
    
    print(f"Running {args.type} benchmark with sizes: {args.sizes}, trials: {args.trials}")
    
    benchmark_suite = BenchmarkSuite()
    
    if args.type == 'comprehensive':
        run_comprehensive_benchmarks()
    elif args.type == 'scalability':
        results = benchmark_suite.benchmark_scalability(args.sizes, args.trials)
        print(benchmark_suite.generate_performance_report())
    elif args.type == 'rram_effects':
        # For RRAM effects, we'll run it for the first size
        results = benchmark_suite.benchmark_rram_effects(args.sizes[0], args.trials)
        print(benchmark_suite.generate_performance_report())
    elif args.type == 'hp_inv_vs_numpy':
        # For HP-INV vs NumPy, run for each size
        for size in args.sizes:
            results = benchmark_suite.benchmark_hp_inv_vs_numpy(size, args.trials)
        print(benchmark_suite.generate_performance_report())
    
    # Optionally plot results
    try:
        import matplotlib.pyplot as plt
        benchmark_suite.plot_benchmark_results()
    except ImportError:
        print("Matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()