#!/usr/bin/env python3

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(csv_path):
    results = {'BF16': {'dims': [], 'custom': [], 'cublas': []},
               'FP32': {'dims': [], 'custom': [], 'cublas': []}}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dtype = row['dtype']
            results[dtype]['dims'].append(int(row['dim']))
            results[dtype]['custom'].append(float(row['custom_time_ms']))
            results[dtype]['cublas'].append(float(row['cublas_time_ms']))

    return results


def plot_results(results, output_dir):
    output_dir = Path(output_dir)

    for dtype in ['BF16', 'FP32']:
        if not results[dtype]['dims']:
            continue

        dims = np.array(results[dtype]['dims'])
        custom = np.array(results[dtype]['custom'])
        cublas = np.array(results[dtype]['cublas'])

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.loglog(dims, custom, 'o-', linewidth=2, markersize=8,
                  label='Custom GEMM', color='#2E86AB')
        ax.loglog(dims, cublas, 's--', linewidth=2, markersize=8,
                  label='CuBLAS', color='#A23B72')

        ax.set_xlabel('Matrix Dimension (N x N)', fontsize=12)
        ax.set_ylabel('Runtime (ms)', fontsize=12)
        ax.set_title(f'GEMM Performance Comparison - {dtype}', fontsize=14, fontweight='bold')

        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)

        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims], rotation=45, ha='right')

        for i, (d, c, cu) in enumerate(zip(dims, custom, cublas)):
            ratio = (cu / c) * 100
            ax.annotate(f'{ratio:.0f}%',
                       xy=(d, c),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9,
                       color='#2E86AB')

        plt.tight_layout()

        output_path = output_dir / f'gemm_performance_{dtype}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

        plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'BF16': ('#2E86AB', '#1E5F74'), 'FP32': ('#A23B72', '#7B2D54')}

    for dtype in ['BF16', 'FP32']:
        if not results[dtype]['dims']:
            continue

        dims = np.array(results[dtype]['dims'])
        custom = np.array(results[dtype]['custom'])
        cublas = np.array(results[dtype]['cublas'])

        ax.loglog(dims, custom, 'o-', linewidth=2, markersize=8,
                  label=f'{dtype} Custom', color=colors[dtype][0])
        ax.loglog(dims, cublas, 's--', linewidth=2, markersize=8,
                  label=f'{dtype} CuBLAS', color=colors[dtype][1])

    ax.set_xlabel('Matrix Dimension (N x N)', fontsize=12)
    ax.set_ylabel('Runtime (ms)', fontsize=12)
    ax.set_title('GEMM Performance Comparison - All Types', fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    all_dims = sorted(set(results['BF16']['dims'] + results['FP32']['dims']))
    ax.set_xticks(all_dims)
    ax.set_xticklabels([str(d) for d in all_dims], rotation=45, ha='right')

    plt.tight_layout()

    output_path = output_dir / 'gemm_performance_all.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, dtype in enumerate(['BF16', 'FP32']):
        if not results[dtype]['dims']:
            continue

        dims = np.array(results[dtype]['dims'])
        custom = np.array(results[dtype]['custom'])
        cublas = np.array(results[dtype]['cublas'])
        ratio = (cublas / custom) * 100

        ax = axes[idx]
        x = np.arange(len(dims))
        width = 0.35

        bars1 = ax.bar(x - width/2, custom, width, label='Custom', color='#2E86AB')
        bars2 = ax.bar(x + width/2, cublas, width, label='CuBLAS', color='#A23B72')

        ax.set_xlabel('Matrix Dimension', fontsize=11)
        ax.set_ylabel('Runtime (ms)', fontsize=11)
        ax.set_title(f'{dtype} Runtime Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for i, (c, cu, r) in enumerate(zip(custom, cublas, ratio)):
            ax.annotate(f'{r:.0f}%',
                       xy=(i - width/2, c),
                       xytext=(0, 3),
                       textcoords='offset points',
                       ha='center', fontsize=8, color='#2E86AB')

    plt.tight_layout()

    output_path = output_dir / 'gemm_performance_bar.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot GEMM benchmark results')
    parser.add_argument('csv_file', help='Path to benchmark results CSV')
    parser.add_argument('--output', '-o', default='results/',
                        help='Output directory for plots')

    args = parser.parse_args()

    results = load_results(args.csv_file)
    plot_results(results, args.output)

    print("\nPerformance Summary:")
    print("=" * 60)

    for dtype in ['BF16', 'FP32']:
        if not results[dtype]['dims']:
            continue

        print(f"\n{dtype}:")
        print(f"{'Dim':<8} {'Custom (ms)':<12} {'CuBLAS (ms)':<12} {'Ratio':<8}")
        print("-" * 44)

        for d, c, cu in zip(results[dtype]['dims'],
                            results[dtype]['custom'],
                            results[dtype]['cublas']):
            ratio = (cu / c) * 100
            print(f"{d:<8} {c:<12.4f} {cu:<12.4f} {ratio:>6.1f}%")


if __name__ == '__main__':
    main()
