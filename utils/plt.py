#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot_results(results: list, output_filename: str = "benchmark_results.png"):
    """
    Plots the benchmark results (TFLOPS vs. Cores) for 2D and 3D tests.
    
    Args:
        results (list): A list of dictionaries, where each dict contains
                        'test' (str), 'cores' (int), and 'tflops' (float).
        output_filename (str): The name of the file to save the plot to.
    """
    
    if not results:
        print("No results to plot.", file=sys.stderr)
        return

    try:
        df = pd.DataFrame(results)
    except Exception as e:
        print(f"Error creating DataFrame for plotting: {e}", file=sys.stderr)
        return


    df_2d = df[df['test'] == '2D'].sort_values('cores')
    df_3d = df[df['test'] == '3D'].sort_values('cores')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    fig.suptitle('JAX TPU Benchmark Performance', fontsize=16)


    if not df_2d.empty:
        ax1.plot(df_2d['cores'], df_2d['tflops'], marker='o', linestyle='-', color='b', label='TFLOPS')
        ax1.set_title('2D Matrix Benchmark (TFLOPS vs. Cores)')
        ax1.set_xlabel('Number of Cores')
        ax1.set_ylabel('Total TFLOPS')
        ax1.set_xticks(df_2d['cores'].unique()) 
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        for i, row in df_2d.iterrows():
            ax1.text(row['cores'], row['tflops'], f' {row["tflops"]:.2f}', va='bottom', ha='center')


    if not df_3d.empty:
        ax2.plot(df_3d['cores'], df_3d['tflops'], marker='s', linestyle='-', color='r', label='TFLOPS')
        ax2.set_title('3D Tensor Benchmark (TFLOPS vs. Cores)')
        ax2.set_xlabel('Number of Cores')
        ax2.set_ylabel('Total TFLOPS')
        ax2.set_xticks(df_3d['cores'].unique())
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        for i, row in df_3d.iterrows():
            ax2.text(row['cores'], row['tflops'], f' {row["tflops"]:.2f}', va='bottom', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    try:
        plt.savefig(output_filename)
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}", file=sys.stderr)

if __name__ == "__main__":

    print("Testing plot generation...")
    test_data = [
        {'test': '2D', 'cores': 1, 'tflops': 150.1, 'avg_ms': 50.2},
        {'test': '2D', 'cores': 4, 'tflops': 580.5, 'avg_ms': 12.1},
        {'test': '2D', 'cores': 8, 'tflops': 1100.9, 'avg_ms': 6.5},
        {'test': '3D', 'cores': 1, 'tflops': 200.3, 'avg_ms': 40.1},
        {'test': '3D', 'cores': 4, 'tflops': 790.0, 'avg_ms': 10.1},
        {'test': '3D', 'cores': 8, 'tflops': 1550.0, 'avg_ms': 5.2},
    ]
    plot_results(test_data, "test_plot.png")
    print("Test plot saved to test_plot.png")
