#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import os
import time
import platform
import psutil
import sys
import subprocess
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

parser = argparse.ArgumentParser(description="JAX TPU Benchmark")
parser.add_argument("-w", "--warmup", type=int, default=10)
parser.add_argument("-m", "--steps", type=int, default=1000)
parser.add_argument("-mxs", "--matrix_size", type=int, default=16384)
parser.add_argument("-md", "--matrix_depth", type=int, default=128)
args = parser.parse_args()

WARMUP_STEPS = args.warmup
NUM_STEPS = args.steps
MATRIX_SIZE = args.matrix_size
MATRIX_DEPTH = args.matrix_depth

N = MATRIX_SIZE
GFLOPs_BASE_OPERATION = (2 * N**3 * 2)
GFLOPs_MULTIPLIER = GFLOPs_BASE_OPERATION * 1.1

console.print(f"[cyan]2D Matrix Size:[/cyan] {MATRIX_SIZE}x{MATRIX_SIZE}")
console.print(f"[cyan]3D Tensor Size:[/cyan] {MATRIX_DEPTH}x{MATRIX_SIZE}x{MATRIX_SIZE}")


def install_dependencies():
    console.print(Panel.fit("[yellow]Checking Dependencies[/yellow]"))
    console.print("This script requires `jax`, `clu`, `tensorflow`, and `tensorflow_datasets`")
    answer = input("Install/upgrade packages? (y/n): ").strip().lower()
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
        console.print(f"[cyan]Running:[/cyan] {' '.join(full_command)}")
        try:
            subprocess.check_call(full_command)
            console.print(f"[green]Successfully installed '{cmd[1]}'![/green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Installation failed: {' '.join(full_command)}[/red]")
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
    console.print("[green]All dependencies installed![/green]\n")


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
            accelerator_info = "None (JAX CPU only)"
    except ImportError:
        accelerator_info = "JAX not installed"
    except Exception as e:
        accelerator_info = f"JAX check failed: {e}"
    table.add_row("Accelerator (JAX)", accelerator_info)
    console.print(table)
    console.print()


def benchmark_jax_2d():
    console.print(Panel.fit("[magenta]JAX 2D Matrix Benchmark[/magenta]"))
    import jax
    import jax.numpy as jnp
    @jax.jit
    def op(a,b):
        C = jnp.dot(a,b)
        D = jnp.tanh(C)+jnp.sin(C/(jnp.log(jnp.abs(a[0,0])+1)*2+1))
        E = jnp.dot(a,D)
        F = jnp.log1p(jnp.abs(E))+jnp.exp(b*0.001)
        return jnp.square(F)
    key = jax.random.PRNGKey(0)
    key1,key2 = jax.random.split(key)
    x = jax.random.normal(key1,(MATRIX_SIZE,MATRIX_SIZE),dtype=jnp.float32)
    y = jax.random.normal(key2,(MATRIX_SIZE,MATRIX_SIZE),dtype=jnp.float32)
    for _ in range(WARMUP_STEPS):
        _ = op(x,y).block_until_ready()
    start = time.perf_counter()
    for _ in range(NUM_STEPS):
        z = op(x,y)
    z.block_until_ready()
    total = time.perf_counter()-start
    avg = total/NUM_STEPS
    GFLOPS = GFLOPs_MULTIPLIER/(avg*1e9)
    TFLOPS = GFLOPS/1000
    console.print(f"[green]2D Benchmark finished in {total:.2f}s[/green]")
    table = Table(title="2D Benchmark Results", show_lines=True)
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")
    table.add_row("Tensor Size", f"{MATRIX_SIZE}x{MATRIX_SIZE}")
    table.add_row("Steps", str(NUM_STEPS))
    table.add_row("Avg Time per Op (ms)", f"{avg*1000:.3f}")
    table.add_row("GFLOPS", f"{GFLOPS:.2f}")
    table.add_row("TFLOPS", f"{TFLOPS:.2f}")
    console.print(table)


def safe_create_3d_tensor(matrix_depth, matrix_size, dtype=jnp.float32):
    import jax
    element_size = jnp.dtype(dtype).itemsize
    total_bytes = matrix_depth*matrix_size*matrix_size*element_size
    try:
        from jax.lib import xla_bridge
        device_memory = xla_bridge.get_backend().device_memory_size(0)
        free_memory = device_memory*0.9
    except Exception:
        free_memory = psutil.virtual_memory().available
    if total_bytes>free_memory:
        scale = (free_memory/total_bytes)**(1/3)
        new_depth = max(1,int(matrix_depth*scale))
        new_size = max(1,int(matrix_size*scale))
        console.print(f"[yellow]Tensor too large, adjusting depth {matrix_depth}->{new_depth}, size {matrix_size}->{new_size}[/yellow]")
        matrix_depth,new_size= new_depth,new_size
        matrix_size=new_size
    key = jax.random.PRNGKey(42)
    key1,key2 = jax.random.split(key)
    x = jax.random.normal(key1,(matrix_depth,matrix_size,matrix_size),dtype=dtype)
    y = jax.random.normal(key2,(matrix_depth,matrix_size,matrix_size),dtype=dtype)
    return x,y,matrix_depth,matrix_size


def benchmark_jax_3d():
    console.print(Panel.fit("[magenta]JAX 3D Tensor Benchmark[/magenta]"))
    import jax
    import jax.numpy as jnp
    @jax.jit
    def op3d(a,b):
        C = jnp.matmul(a,b)
        D = jnp.tanh(C)+jnp.sin(C/(jnp.log(jnp.abs(a[0,0,0])+1)*2+1))
        E = jnp.matmul(a,D)
        F = jnp.log1p(jnp.abs(E))+jnp.exp(b*0.001)
        return jnp.square(F)
    x,y,depth,size = safe_create_3d_tensor(MATRIX_DEPTH,MATRIX_SIZE)
    for _ in range(WARMUP_STEPS):
        _ = op3d(x,y).block_until_ready()
    start = time.perf_counter()
    for _ in range(NUM_STEPS):
        z = op3d(x,y)
    z.block_until_ready()
    total = time.perf_counter()-start
    avg = total/NUM_STEPS
    GFLOPS = depth*GFLOPs_MULTIPLIER/(avg*1e9)
    TFLOPS = GFLOPS/1000
    console.print(f"[green]3D Benchmark finished in {total:.2f}s[/green]")
    table = Table(title="3D Benchmark Results", show_lines=True)
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")
    table.add_row("Tensor Size", f"{depth}x{size}x{size}")
    table.add_row("Steps", str(NUM_STEPS))
    table.add_row("Avg Time per Op (ms)", f"{avg*1000:.3f}")
    table.add_row("GFLOPS", f"{GFLOPS:.2f}")
    table.add_row("TFLOPS", f"{TFLOPS:.2f}")
    console.print(table)


def main():
    install_dependencies()
    get_system_info()
    benchmark_jax_2d()
    benchmark_jax_3d()


if __name__ == "__main__":
    main()
