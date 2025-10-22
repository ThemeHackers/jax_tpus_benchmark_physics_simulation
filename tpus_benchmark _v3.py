#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import platform
import psutil
import sys
import subprocess
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
WARMUP_STEPS = 10
NUM_STEPS = 1000
MATRIX_SIZE = 16384 


N = MATRIX_SIZE

GFLOPs_BASE_OPERATION = (2 * N**3 * 2) 


GFLOPs_MULTIPLIER = GFLOPs_BASE_OPERATION * 1.1


def install_dependencies():
    console.print(Panel.fit("[yellow]Checking Dependencies[/yellow]"))
    console.print("This script requires `jax`, `clu`, `tensorflow`, and `tensorflow_datasets`")
    
    answer = input("Do you want this script to install or upgrade these packages? (y/n): ").strip().lower()
    
    if answer != 'y':
        console.print("[cyan]Skipping dependency installation...[/cyan]\n")
        return

    commands = [
        ["install", "jax[tpu]", "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"],
        ["install", "--upgrade", "clu"],
        ["install", "tensorflow"],
        ["install", "tensorflow_datasets"]
    ]

    for cmd in commands:
        full_command = [sys.executable, "-m", "pip"] + cmd
        console.print(f"\n[cyan]Running:[/cyan] {' '.join(full_command)}")
        try:
            subprocess.check_call(full_command)
            console.print(f"[green]Successfully installed '{cmd[1]}'![/green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Installation failed for command: {' '.join(full_command)}[/red]")
            console.print(f"Error: {e}")
            console.print("[red]Please install manually and run the script again[/red]")
            sys.exit(1)

    console.print("\n[green]All dependencies are successfully installed![/green]\n")


def get_system_info():
    console.print(Panel.fit("[cyan]Collecting System Information...[/cyan]"))
    table = Table(title="System Information", show_header=False, expand=True)
    
    table.add_row("OS", f"{platform.system()} {platform.release()} ({platform.version()})")
    table.add_row("Machine", platform.machine())
    table.add_row("Processor", platform.processor() or "Unknown")
    table.add_row("Python Version", platform.python_version())
    table.add_row("CPU Cores", f"{psutil.cpu_count(logical=True)} logical, {psutil.cpu_count(logical=False)} physical")
    table.add_row("Total RAM", f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB")

    accelerator_info = "Unknown"
    try:
        import jax
        devices = jax.devices()
        accelerators = [d for d in devices if d.platform != 'cpu']
        if accelerators:
            accelerator_info = f"{accelerators[0].platform.upper()} ({accelerators[0].device_kind})"
        else:
            accelerator_info = "None (JAX found CPU only)"
    except ImportError:
        accelerator_info = "JAX not installed"
    except Exception as e:
        accelerator_info = f"JAX check failed: {e}"

    table.add_row("Accelerator (found by JAX)", accelerator_info)
    console.print(table)
    console.print()


def benchmark_jax():
    console.print(Panel.fit("[bold magenta]Starting JAX Complex Tensor Benchmark[/bold magenta]"))
    
    try:
        import jax
        import jax.numpy as jnp
        console.print("Successfully imported JAX and jax.numpy")
    except ImportError:
        console.print("[red]JAX not found! Please re-run and select 'y' to install[/red]")
        return

    try:
        devices = jax.devices()
        device_name = devices[0].platform.upper()
        console.print(f"JAX will run the benchmark on: [bold blue]{device_name}[/bold blue]")

        @jax.jit
        def complex_tensor_operation(a, b):
            """
            Performs a sequence of heavy and complex tensor operations:
            1. Matrix Multiplication (Heavy FLOPS)
            2. Non-linear Activation (Element-wise)
            3. Second Matrix Multiplication
            4. Element-wise combination (Log/Exp)
            """
 
            C = jnp.dot(a, b)
            

            D = jnp.tanh(C) + jnp.sin(C / (jnp.log(jnp.abs(a[0, 0]) + 1) * 2 + 1))
            

            E = jnp.dot(a, D)
            

            F = jnp.log1p(jnp.abs(E)) + jnp.exp(b * 0.001)
            
            return jnp.square(F) 


        console.print("Creating JAX PRNGKey (Random Number Generator)...")
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        console.print("Key created successfully")

        console.print(f"Creating 2 random matrices of size ({MATRIX_SIZE}x{MATRIX_SIZE}, float32)...")

        x = jax.random.normal(key1, (MATRIX_SIZE, MATRIX_SIZE), dtype=jnp.float32)
        y = jax.random.normal(key2, (MATRIX_SIZE, MATRIX_SIZE), dtype=jnp.float32)
        console.print("[green]Matrices created successfully[/green]")

        console.print(f"Warming up JIT compiler ({WARMUP_STEPS} steps)...")
        for _ in range(WARMUP_STEPS):

             _ = complex_tensor_operation(x, y).block_until_ready()
        console.print("[green]Warmup complete[/green]")

        console.print(f"Running benchmark ({NUM_STEPS} steps)...")
        start_time = time.perf_counter()
        
        for _ in range(NUM_STEPS):
            z = complex_tensor_operation(x, y)
        
        z.block_until_ready() 
        
        total_time = time.perf_counter() - start_time
        

        avg_time_per_op = total_time / NUM_STEPS
        GFLOPS = GFLOPs_MULTIPLIER / (avg_time_per_op * 10**9)
        TFLOPS = GFLOPS / 1000

        console.print(f"[green]Benchmark finished in {total_time:.2f} seconds[/green]")

        result_table = Table(title="[bold blue]JAX COMPLEX TENSOR BENCHMARK RESULTS[/bold blue]", 
                             show_lines=True)
        result_table.add_column("Metric", justify="right", style="bold yellow")
        result_table.add_column("Value", justify="left")
        
        result_table.add_row("Device", devices[0].platform.upper())
        result_table.add_row("Tensor Size", f"{MATRIX_SIZE}x{MATRIX_SIZE} (float32)")
        result_table.add_row("Steps (Complex Ops)", str(NUM_STEPS))
        result_table.add_row("---", "---")
        result_table.add_row("Total Time (s)", f"{total_time:.2f} s")
        result_table.add_row("[bold]Avg Time per Operation (ms)[/bold]", f"{avg_time_per_op * 1000:.3f} ms")
        result_table.add_row("---", "---")
        result_table.add_row("[bold green]Calculated GFLOPS[/bold green]", f"{GFLOPS:.2f} GFLOPS")
        result_table.add_row("[bold green]Calculated TFLOPS[/bold green]", f"{TFLOPS:.2f} TFLOPS")

        console.print(result_table)
        console.print()
        
    except Exception as e:
        console.print(f"[bold red]JAX benchmark failed: {e}[/bold red]")
        if "MEM_ALLOC_FAILURE" in str(e) or "OOM" in str(e) or "memory" in str(e).lower():
            console.print("[yellow]Hint: Matrix size is likely too large for your device's HBM/VRAM.[/yellow]")
            console.print(f"[yellow]Try reducing the MATRIX_SIZE from {MATRIX_SIZE} to 4096 or lower.[/yellow]")
        else:
            console.print(f"[red]Error: {e}[/red]")


def main():
    if not any('tpu' in arg.lower() for arg in sys.argv):
        console.print(
            Panel.fit(
                "[bold red]WARNING[/bold red]\nThis benchmark is optimized for TPU. Ensure you are running in a TPU environment!"
            )
        )
    install_dependencies()
    get_system_info()
    benchmark_jax()


if __name__ == "__main__":
    main()