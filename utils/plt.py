#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

STYLE_MAP = {
    '2D':        {'color': 'tab:blue',   'marker': 'o', 'label': '2D MatMul'},
    '3D':        {'color': 'tab:green',  'marker': 's', 'label': '3D MatMul'},
    '2D_FFT':    {'color': 'tab:red',    'marker': '^', 'label': '2D FFT'},
    '3D_FFT':    {'color': 'tab:orange', 'marker': 'v', 'label': '3D FFT'},
    'Bandwidth': {'color': 'tab:purple', 'marker': 'd', 'label': 'Bandwidth'}
}

LINESTYLE_PERF = '-'  
LINESTYLE_LATENCY = '--' 

def plot_results(results: list, output_filename: str = "benchmark_results.png"):
    
    if not results:
        print("No results to plot.", file=sys.stderr)
        return

    try:
        df = pd.DataFrame(results)
    except Exception as e:
        print(f"Error creating DataFrame for plotting: {e}", file=sys.stderr)
        return
        
    all_tests = sorted(df['test'].unique())
    if not all_tests:
        print("No test data found in results.", file=sys.stderr)
        return


    data_by_test = {
        test: df[df['test'] == test].sort_values('cores') for test in all_tests
    }


    has_tflops = 'tflops' in df.columns
    has_bandwidth = 'bandwidth_gbs' in df.columns
    has_time = 'avg_ms' in df.columns

    if not (has_tflops or has_bandwidth or has_time):
        print("No plottable metrics (tflops, bandwidth_gbs, avg_ms) found.", file=sys.stderr)
        return

    fig, (ax_perf, ax_time) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('JAX Benchmark Results', fontsize=18, weight='bold')

    all_cores = sorted(df['cores'].unique())
    
    ax_perf.set_title('Performance (Higher is Better)', fontsize=14)
    ax_perf.set_ylabel('TFLOPS', fontsize=12, color='tab:cyan')
    ax_perf.tick_params(axis='y', labelcolor='tab:cyan')
    ax_perf.grid(True, linestyle='--', alpha=0.6)
    ax_perf.set_xticks(all_cores)
    
    ax_bw = ax_perf.twinx()
    ax_bw.set_ylabel('Bandwidth (GB/s)', fontsize=12, color='tab:purple')
    ax_bw.tick_params(axis='y', labelcolor='tab:purple')

    ax_time.set_title('Latency (Lower is Better)', fontsize=14)
    ax_time.set_ylabel('Avg. Time (ms)', fontsize=12)
    ax_time.grid(True, linestyle='--', alpha=0.6)
    ax_time.set_xlabel('Number of Cores', fontsize=12)
    
    legend_handles = []

    for test in all_tests:
        if test not in data_by_test:
            continue
            
        data = data_by_test[test]
        style = STYLE_MAP.get(test, {'color': 'gray', 'marker': 'x', 'label': test})
        
        legend_handles.append(
            plt.Line2D([0], [0], 
                       color=style.get('color'), 
                       marker=style.get('marker'), 
                       linestyle='-', 
                       label=style.get('label', test))
        )
        
        if test == 'Bandwidth' and has_bandwidth and 'bandwidth_gbs' in data.columns:

            ax_bw.plot(
                data['cores'], data['bandwidth_gbs'],
                marker=style.get('marker'),
                color=style.get('color'),
                linestyle=LINESTYLE_PERF,
            )
            for _, row in data.iterrows():
                ax_bw.text(row['cores'], row['bandwidth_gbs'], f' {row["bandwidth_gbs"]:.1f} GB/s', va='bottom', ha='center', color=style.get('color'), size='small')
        
        elif has_tflops and 'tflops' in data.columns:

            ax_perf.plot(
                data['cores'], data['tflops'],
                marker=style.get('marker'),
                color=style.get('color'),
                linestyle=LINESTYLE_PERF,
            )
            for _, row in data.iterrows():
                ax_perf.text(row['cores'], row['tflops'], f' {row["tflops"]:.1f} T', va='bottom', ha='center', color=style.get('color'), size='small')


        if has_time and 'avg_ms' in data.columns:
            ax_time.plot(
                data['cores'], data['avg_ms'],
                marker=style.get('marker'),
                color=style.get('color'),
                linestyle=LINESTYLE_LATENCY, 
            )
            for _, row in data.iterrows():
                ax_time.text(row['cores'], row['avg_ms'], f' {row["avg_ms"]:.2f} ms', va='top', ha='center', color=style.get('color'), size='small')


    legend_handles.append(plt.Line2D([0], [0], color='gray', linestyle=LINESTYLE_PERF, label='Performance (TFLOPS/GB/s)'))
    legend_handles.append(plt.Line2D([0], [0], color='gray', linestyle=LINESTYLE_LATENCY, label='Latency (ms)'))


    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=min(len(legend_handles), 4))
    

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"Plot saved to {output_filename}", file=sys.stdout)
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}", file=sys.stderr)


if __name__ == "__main__":

    print("Testing 5-value, 2-plot generation...")
    test_data = [
        # 2D MatMul
        {'test': '2D', 'cores': 1, 'tflops': 150.1, 'avg_ms': 50.2},
        {'test': '2D', 'cores': 4, 'tflops': 580.5, 'avg_ms': 12.1},
        {'test': '2D', 'cores': 8, 'tflops': 1100.9, 'avg_ms': 6.5},
        # 3D MatMul
        {'test': '3D', 'cores': 1, 'tflops': 200.3, 'avg_ms': 40.1},
        {'test': '3D', 'cores': 4, 'tflops': 790.0, 'avg_ms': 10.1},
        {'test': '3D', 'cores': 8, 'tflops': 1550.0, 'avg_ms': 5.2},
        # 2D FFT
        {'test': '2D_FFT', 'cores': 1, 'tflops': 50.0, 'avg_ms': 80.0},
        {'test': '2D_FFT', 'cores': 4, 'tflops': 190.0, 'avg_ms': 20.0},
        {'test': '2D_FFT', 'cores': 8, 'tflops': 350.0, 'avg_ms': 11.0},
        # 3D FFT
        {'test': '3D_FFT', 'cores': 1, 'tflops': 70.0, 'avg_ms': 70.0},
        {'test': '3D_FFT', 'cores': 4, 'tflops': 270.0, 'avg_ms': 18.0},
        {'test': '3D_FFT', 'cores': 8, 'tflops': 500.0, 'avg_ms': 10.0},
        # Bandwidth
        {'test': 'Bandwidth', 'cores': 1, 'bandwidth_gbs': 100.0, 'avg_ms': 15.0},
        {'test': 'Bandwidth', 'cores': 4, 'bandwidth_gbs': 380.0, 'avg_ms': 4.0},
        {'test': 'Bandwidth', 'cores': 8, 'bandwidth_gbs': 700.0, 'avg_ms': 2.5},
    ]
    plot_results(test_data, "test_plot_all_metrics.png")
    print("Test plot saved to test_plot_all_metrics.png")
