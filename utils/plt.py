#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

def plot_results(results: list, output_filename: str = "benchmark_results.png"):
    
    if not results:
        print("No results to plot.", file=sys.stderr)
        return

    try:
        df = pd.DataFrame(results)
    except Exception as e:
        print(f"Error creating DataFrame for plotting: {e}", file=sys.stderr)
        return
        
    if not all(col in df.columns for col in ['test', 'cores', 'tflops']):
        print("DataFrame is missing required columns (test, cores, tflops).", file=sys.stderr)
        return
        
    plot_avg_ms = 'avg_ms' in df.columns

    df_2d = df[df['test'] == '2D'].sort_values('cores')
    df_3d = df[df['test'] == '3D'].sort_values('cores')

    has_2d = not df_2d.empty
    has_3d = not df_3d.empty
    
    if not has_2d and not has_3d:
        print("No 2D or 3D data found in results.", file=sys.stderr)
        return
        
    fig, ax_tflops = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('JAX TPU Benchmark Performance (2D vs 3D)', fontsize=18, weight='bold')

    all_cores = pd.concat([df_2d['cores'], df_3d['cores']]).unique()
    all_cores.sort()
    ax_tflops.set_xlabel('Number of Cores', fontsize=12)
    ax_tflops.set_xticks(all_cores)
    ax_tflops.grid(True, linestyle='--', alpha=0.6, which='major')

    color_2d_tflops = 'tab:blue'
    color_3d_tflops = 'tab:green'
    ax_tflops.set_ylabel('Total TFLOPS', fontsize=12)
    
    lns = []

    if has_2d:
        cores_2d = df_2d['cores']
        tflops_2d = df_2d['tflops']
        ln1 = ax_tflops.plot(cores_2d, tflops_2d, marker='o', linestyle='-', color=color_2d_tflops, label='2D TFLOPS')
        ax_tflops.tick_params(axis='y', labelcolor=color_2d_tflops)
        for i, row in df_2d.iterrows():
            ax_tflops.text(row['cores'], row['tflops'], f' {row["tflops"]:.2f} T', va='bottom', ha='center', color=color_2d_tflops, size='small')
        lns.extend(ln1)

    if has_3d:
        cores_3d = df_3d['cores']
        tflops_3d = df_3d['tflops']
        ln2 = ax_tflops.plot(cores_3d, tflops_3d, marker='s', linestyle='-', color=color_3d_tflops, label='3D TFLOPS')
        if not has_2d:
             ax_tflops.tick_params(axis='y', labelcolor=color_3d_tflops)
        
        for i, row in df_3d.iterrows():
            ax_tflops.text(row['cores'], row['tflops'], f' {row["tflops"]:.2f} T', va='bottom', ha='right', color=color_3d_tflops, size='small')
        lns.extend(ln2)

    if plot_avg_ms:
        ax_time = ax_tflops.twinx()
        ax_time.set_ylabel('Avg. Time (ms)', fontsize=12)
        
        color_2d_time = 'tab:red'
        color_3d_time = 'tab:orange'

        if has_2d and 'avg_ms' in df_2d.columns:
            avg_ms_2d = df_2d['avg_ms']
            ln3 = ax_time.plot(cores_2d, avg_ms_2d, marker='x', linestyle='--', color=color_2d_time, label='2D Avg. Time (ms)')
            ax_time.tick_params(axis='y', labelcolor=color_2d_time)
            for i, row in df_2d.iterrows():
                ax_time.text(row['cores'], row['avg_ms'], f' {row["avg_ms"]:.2f} ms', va='top', ha='center', color=color_2d_time, size='small')
            lns.extend(ln3)

        if has_3d and 'avg_ms' in df_3d.columns:
            avg_ms_3d = df_3d['avg_ms']
            ln4 = ax_time.plot(cores_3d, avg_ms_3d, marker='d', linestyle='--', color=color_3d_time, label='3D Avg. Time (ms)')
            if not has_2d:
                ax_time.tick_params(axis='y', labelcolor=color_3d_time)
            
            for i, row in df_3d.iterrows():
                ax_time.text(row['cores'], row['avg_ms'], f' {row["avg_ms"]:.2f} ms', va='top', ha='right', color=color_3d_time, size='small')
            lns.extend(ln4)

    labs = [l.get_label() for l in lns]
    ax_tflops.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=min(len(lns), 4))

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"Plot saved to {output_filename}", file=sys.stdout)
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}", file=sys.stderr)


if __name__ == "__main__":
    # for test this file
    print("Testing 4-value combined plot generation...")
    test_data = [
        {'test': '2D', 'cores': 1, 'tflops': 150.1, 'avg_ms': 50.2},
        {'test': '2D', 'cores': 4, 'tflops': 580.5, 'avg_ms': 12.1},
        {'test': '2D', 'cores': 8, 'tflops': 1100.9, 'avg_ms': 6.5},
        {'test': '3D', 'cores': 1, 'tflops': 200.3, 'avg_ms': 40.1},
        {'test': '3D', 'cores': 4, 'tflops': 790.0, 'avg_ms': 10.1},
        {'test': '3D', 'cores': 8, 'tflops': 1550.0, 'avg_ms': 5.2},
    ]
    plot_results(test_data, "test_plot_4_value_combined.png")
    print("Test plot saved to test_plot_4_value_combined.png")
