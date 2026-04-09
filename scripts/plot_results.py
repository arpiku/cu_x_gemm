#!/usr/bin/env python3
"""Plot GEMM benchmark results from CSV output."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def safe_float(val, default=0.0):
    """Safely convert to float, returning default for empty/None values."""
    if val is None or val == '':
        return default
    return float(val)


def load_results(csv_path):
    """Load benchmark results from CSV file."""
    bf16_results = []
    fp32_results = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['dtype'] == 'BF16':
                bf16_results.append({
                    'dim': int(row['dim']),
                    'variant': row['variant'],
                    'desc': row['desc'],
                    'custom_ms': float(row['custom_ms']),
                    'cublas_32f_ms': safe_float(row.get('cublas_32f_ms')),
                    'cublas_tc_ms': safe_float(row.get('cublas_tc_ms')),
                    'l2_error': safe_float(row.get('l2_error')),
                })
            elif row['dtype'] == 'FP32':
                fp32_results.append({
                    'dim': int(row['dim']),
                    'variant': row['variant'],
                    'desc': row['desc'],
                    'custom_ms': float(row['custom_ms']),
                    'sgemm_ms': safe_float(row.get('sgemm_ms')),
                    'cuda_ms': safe_float(row.get('cuda_ms')),
                    'pedantic_ms': safe_float(row.get('pedantic_ms')),
                    'tc_ms': safe_float(row.get('tc_ms')),
                    'l2_error': safe_float(row.get('l2_error')),
                })
    
    return bf16_results, fp32_results


def plot_bf16(results, output_dir):
    """Plot BF16 performance comparison."""
    if not results:
        return
    
    dims = np.array([r['dim'] for r in results])
    custom = np.array([r['custom_ms'] for r in results])
    cublas_32f = np.array([r['cublas_32f_ms'] for r in results])
    cublas_tc = np.array([r['cublas_tc_ms'] for r in results])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Log-log runtime plot
    ax = axes[0]
    ax.loglog(dims, custom, 'o-', linewidth=2, markersize=8, label='Custom', color='#2E86AB')
    ax.loglog(dims, cublas_32f, 's--', linewidth=2, markersize=8, label='CuBLAS 32F', color='#A23B72')
    ax.loglog(dims, cublas_tc, '^-.', linewidth=2, markersize=8, label='CuBLAS TC', color='#F18F01')
    
    ax.set_xlabel('Matrix Dimension (N)', fontsize=11)
    ax.set_ylabel('Runtime (ms)', fontsize=11)
    ax.set_title('BF16 Runtime Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims], rotation=45, ha='right')
    
    # Right: Ratio bar chart
    ax = axes[1]
    x = np.arange(len(dims))
    width = 0.35
    
    ratio_32f = (cublas_32f / custom) * 100
    ratio_tc = (cublas_tc / custom) * 100
    
    bars1 = ax.bar(x - width/2, ratio_32f, width, label='vs CuBLAS 32F', color='#A23B72')
    bars2 = ax.bar(x + width/2, ratio_tc, width, label='vs CuBLAS TC', color='#F18F01')
    
    ax.set_xlabel('Matrix Dimension', fontsize=11)
    ax.set_ylabel('Performance Ratio (%)', fontsize=11)
    ax.set_title('BF16: Custom / CuBLAS Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.axhline(y=70, color='green', linestyle=':', linewidth=1.5, label='70% target')
    
    # Add value labels
    for bar, val in zip(bars1, ratio_32f):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / 'gemm_bf16_performance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_fp32(results, output_dir):
    """Plot FP32 performance comparison."""
    if not results:
        return
    
    dims = np.array([r['dim'] for r in results])
    custom = np.array([r['custom_ms'] for r in results])
    sgemm = np.array([r['sgemm_ms'] for r in results])
    cuda = np.array([r['cuda_ms'] for r in results])
    pedantic = np.array([r['pedantic_ms'] for r in results])
    tc = np.array([r['tc_ms'] for r in results])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Log-log runtime plot
    ax = axes[0]
    ax.loglog(dims, custom, 'o-', linewidth=2, markersize=8, label='Custom', color='#2E86AB')
    ax.loglog(dims, sgemm, 's--', linewidth=2, markersize=6, label='Sgemm', color='#A23B72')
    ax.loglog(dims, cuda, '^-.', linewidth=2, markersize=6, label='CUDA 32F', color='#F18F01')
    ax.loglog(dims, pedantic, 'd:', linewidth=2, markersize=6, label='Pedantic', color='#3A86FF')
    ax.loglog(dims, tc, 'v-', linewidth=2, markersize=6, label='TF32 (TC)', color='#8338EC')
    
    ax.set_xlabel('Matrix Dimension (N)', fontsize=11)
    ax.set_ylabel('Runtime (ms)', fontsize=11)
    ax.set_title('FP32 Runtime Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims], rotation=45, ha='right')
    
    # Right: Ratio bar chart (grouped)
    ax = axes[1]
    x = np.arange(len(dims))
    width = 0.2
    
    ratio_sgemm = (sgemm / custom) * 100
    ratio_cuda = (cuda / custom) * 100
    ratio_pedantic = (pedantic / custom) * 100
    ratio_tc = (tc / custom) * 100
    
    ax.bar(x - 1.5*width, ratio_sgemm, width, label='Sgemm', color='#A23B72')
    ax.bar(x - 0.5*width, ratio_cuda, width, label='CUDA', color='#F18F01')
    ax.bar(x + 0.5*width, ratio_pedantic, width, label='Pedantic', color='#3A86FF')
    ax.bar(x + 1.5*width, ratio_tc, width, label='TC', color='#8338EC')
    
    ax.set_xlabel('Matrix Dimension', fontsize=11)
    ax.set_ylabel('Performance Ratio (%)', fontsize=11)
    ax.set_title('FP32: Custom / CuBLAS Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.axhline(y=70, color='green', linestyle=':', linewidth=1.5)
    
    plt.tight_layout()
    output_path = output_dir / 'gemm_fp32_performance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary(bf16_results, fp32_results, output_dir):
    """Plot summary comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # BF16: compare against CuBLAS TC (primary target for Tensor Core kernels)
    if bf16_results:
        dims_bf16 = np.array([r['dim'] for r in bf16_results])
        ratio_bf16 = np.array([(r['cublas_tc_ms'] / r['custom_ms']) * 100 for r in bf16_results])
        x_bf16 = np.arange(len(dims_bf16))
        ax.bar(x_bf16 - 0.2, ratio_bf16, 0.4, label='BF16 vs CuBLAS TC', color='#2E86AB')
        
        # Use BF16 dimensions as x-axis base
        ax.set_xticks(x_bf16)
        ax.set_xticklabels([str(d) for d in dims_bf16], rotation=45, ha='right')
    
    # FP32: compare against Pedantic (primary target for CUDA core kernels)
    if fp32_results:
        dims_fp32 = np.array([r['dim'] for r in fp32_results])
        ratio_fp32 = np.array([(r['pedantic_ms'] / r['custom_ms']) * 100 for r in fp32_results])
        x_fp32 = np.arange(len(dims_fp32))
        ax.bar(x_fp32 + 0.2, ratio_fp32, 0.4, label='FP32 vs CuBLAS Pedantic', color='#A23B72')
    
    ax.axhline(y=70, color='green', linestyle='--', linewidth=2, label='70% Target')
    
    ax.set_xlabel('Matrix Dimension', fontsize=11)
    ax.set_ylabel('Performance Ratio (%)', fontsize=11)
    ax.set_title('GEMM Performance Summary: Custom vs Primary CuBLAS Reference', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = output_dir / 'gemm_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot GEMM benchmark results')
    parser.add_argument('csv_file', help='Path to benchmark results CSV')
    parser.add_argument('--output', '-o', default='results/', help='Output directory for plots')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bf16_results, fp32_results = load_results(args.csv_file)
    
    print(f"Loaded {len(bf16_results)} BF16 results, {len(fp32_results)} FP32 results")
    
    plot_bf16(bf16_results, output_dir)
    plot_fp32(fp32_results, output_dir)
    plot_summary(bf16_results, fp32_results, output_dir)
    
    print("\nPerformance Summary:")
    print("=" * 70)
    
    if bf16_results:
        print("\nBF16 (vs CuBLAS Tensor Core):")
        print(f"{'Dim':<8} {'Custom (ms)':<12} {'CuBLAS TC (ms)':<15} {'Ratio':<10}")
        print("-" * 50)
        for r in bf16_results:
            ratio = (r['cublas_tc_ms'] / r['custom_ms']) * 100
            print(f"{r['dim']:<8} {r['custom_ms']:<12.4f} {r['cublas_tc_ms']:<15.4f} {ratio:>8.1f}%")
    
    if fp32_results:
        print("\nFP32 (vs CuBLAS Pedantic - CUDA cores):")
        print(f"{'Dim':<8} {'Custom (ms)':<12} {'Pedantic (ms)':<14} {'Ratio':<10}")
        print("-" * 50)
        for r in fp32_results:
            ratio = (r['pedantic_ms'] / r['custom_ms']) * 100
            print(f"{r['dim']:<8} {r['custom_ms']:<12.4f} {r['pedantic_ms']:<14.4f} {ratio:>8.1f}%")


if __name__ == '__main__':
    main()
