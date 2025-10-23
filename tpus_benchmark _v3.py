#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import time
import platform
import psutil
import sys
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from utils.check_deps import check_dependencies
from utils.jax_devices import list_jax_devices 



try:
    NUM_AVAILABLE_CORES = jax.device_count()
except Exception as e:
    print(f"Error initializing JAX: {e}")
    print("Please ensure JAX and a TPU backend are correctly installed.")
    sys.exit(1)

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
console.print(Panel.fit(f"[bold cyan]Found {NUM_AVAILABLE_CORES} JAX devices (cores).[/bold cyan]"))


def get_system_info():
    console.print(Panel.fit("[cyan]Collecting System Information...[/cyan]"))
    table = Table(title="System Information", show_header=False, expand=True)
    table.add_row("OS", f"{platform.system()} {platform.release()} ({platform.version()})")
    table.add_row("Machine", platform.machine())
    table.add_row("Processor", platform.processor() or "Unknown")
    table.add_row("Python Version", platform.python_version())
    table.add_row("CPU Cores", f"{psutil.cpu_count(logical=True)} logical, {psutil.cpu_count(logical=False)} physical")
    table.add_row("Total RAM", f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    

    try:
        devices = jax.devices()
        if devices:

            dev0 = devices[0]
            accelerator_info = f"{dev0.platform.upper()} ({dev0.device_kind})"
            table.add_row("JAX Accelerator", accelerator_info)
            table.add_row("JAX Device Count", str(len(devices)))

            try:
                mem_bytes = dev0.memory_stats()["bytes_limit"]
                table.add_row(f"Memory (Device 0)", f"{round(mem_bytes / (1024**3), 2)} GB")
            except Exception:
                pass 
        else:
            table.add_row("JAX Accelerator", "None (CPU only)")
    except Exception as e:
        table.add_row("JAX Accelerator", f"JAX check failed: {e}")
        
    console.print(table)
    console.print()



@jax.jit
def op_2d(a,b):
    C = jnp.dot(a,b)
    D = jnp.tanh(C)+jnp.sin(C/(jnp.log(jnp.abs(a[0,0])+1)*2+1))
    E = jnp.dot(a,D)
    F = jnp.log1p(jnp.abs(E))+jnp.exp(b*0.001)
    return jnp.square(F)

@jax.jit
def op_3d(a,b):
    C = jnp.matmul(a,b) 
    D = jnp.tanh(C)+jnp.sin(C/(jnp.log(jnp.abs(a[0,0,0])+1)*2+1))
    E = jnp.matmul(a,D)
    F = jnp.log1p(jnp.abs(E))+jnp.exp(b*0.001)
    return jnp.square(F)



def benchmark_jax_2d(num_cores_to_use: int):
    mode = f"{num_cores_to_use}-Core ({'JIT' if num_cores_to_use == 1 else 'PMAP'})"
    console.print(Panel.fit(f"[magenta]JAX 2D Matrix Benchmark ({mode})[/magenta]"))

    key = jax.random.PRNGKey(0)

    try:
        if num_cores_to_use == 1:
            compiled_op = op_2d 
            shape = (MATRIX_SIZE, MATRIX_SIZE)
            keys = jax.random.split(key, 2)
            x = jax.random.normal(keys[0], shape, dtype=jnp.float32)
            y = jax.random.normal(keys[1], shape, dtype=jnp.float32)
        else:
            compiled_op = jax.pmap(op_2d)
            shape = (num_cores_to_use, MATRIX_SIZE, MATRIX_SIZE)
            shape_per_core = (MATRIX_SIZE, MATRIX_SIZE)

            key_x, key_y = jax.random.split(key, 2)
            keys_x = jax.random.split(key_x, num_cores_to_use)
            keys_y = jax.random.split(key_y, num_cores_to_use)

            x = jax.vmap(lambda k: jax.random.normal(k, shape_per_core, dtype=jnp.float32))(keys_x)
            y = jax.vmap(lambda k: jax.random.normal(k, shape_per_core, dtype=jnp.float32))(keys_y)
        
        console.print(f"Allocating 2D Tensors with shape: {shape}")
        x.block_until_ready()
        y.block_until_ready()

    except (jax.errors.JaxRuntimeError, RuntimeError) as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "OOM" in str(e):
            console.print(f"[red]OOM Error: Failed to allocate tensor of shape {shape}. Skipping 2D test.[/red]")
            console.print()
            return
        else:
            raise e

    for _ in range(WARMUP_STEPS):
        _ = compiled_op(x,y).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(NUM_STEPS):
        z = compiled_op(x,y)
    z.block_until_ready()
    
    total = time.perf_counter()-start
    avg = total/NUM_STEPS
    
    GFLOPS = (num_cores_to_use * GFLOPs_MULTIPLIER) / (avg * 1e9)
    TFLOPS = GFLOPS / 1000
    
    console.print(f"[green]2D Benchmark ({mode}) finished in {total:.2f}s[/green]")
    table = Table(title=f"2D Benchmark Results ({mode})", show_lines=True)
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")
    table.add_row("Mode", f"{num_cores_to_use} Core(s) - {'JIT' if num_cores_to_use == 1 else 'PMAP'}")
    table.add_row("Per-Core Tensor Size", f"{MATRIX_SIZE}x{MATRIX_SIZE}")
    table.add_row("Total Tensor Shape", str(shape))
    table.add_row("Steps", str(NUM_STEPS))
    table.add_row("Avg Time per Op (ms)", f"{avg*1000:.3f}")
    table.add_row("Total GFLOPS", f"{GFLOPS:.2f}")
    table.add_row("Total TFLOPS", f"{TFLOPS:.2f}")
    console.print(table)
    console.print()


def benchmark_jax_3d(num_cores_to_use: int):
    mode = f"{num_cores_to_use}-Core ({'JIT' if num_cores_to_use == 1 else 'PMAP'})"
    console.print(Panel.fit(f"[magenta]JAX 3D Tensor Benchmark ({mode})[/magenta]"))

    if MATRIX_DEPTH % num_cores_to_use != 0:
        console.print(f"[red]Error: MATRIX_DEPTH ({MATRIX_DEPTH}) must be divisible by num_cores_to_use ({num_cores_to_use}).[/red]")
        console.print(f"[yellow]Skipping 3D {mode} test.[/yellow]")
        console.print()
        return

    D_per_core = MATRIX_DEPTH // num_cores_to_use
    key = jax.random.PRNGKey(42)

    try:
        if num_cores_to_use == 1:
            compiled_op = op_3d 
            shape = (D_per_core, MATRIX_SIZE, MATRIX_SIZE)
            keys = jax.random.split(key, 2)
            x = jax.random.normal(keys[0], shape, dtype=jnp.float32)
            y = jax.random.normal(keys[1], shape, dtype=jnp.float32)
        else:
            compiled_op = jax.pmap(op_3d) 
            shape = (num_cores_to_use, D_per_core, MATRIX_SIZE, MATRIX_SIZE)
            shape_per_core = (D_per_core, MATRIX_SIZE, MATRIX_SIZE)
            
            key_x, key_y = jax.random.split(key, 2)
            keys_x = jax.random.split(key_x, num_cores_to_use)
            keys_y = jax.random.split(key_y, num_cores_to_use)

            x = jax.vmap(lambda k: jax.random.normal(k, shape_per_core, dtype=jnp.float32))(keys_x)
            y = jax.vmap(lambda k: jax.random.normal(k, shape_per_core, dtype=jnp.float32))(keys_y)

        console.print(f"Allocating 3D Tensors with shape: {shape}")
        x.block_until_ready()
        y.block_until_ready()

    except (jax.errors.JaxRuntimeError, RuntimeError) as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "OOM" in str(e):
            console.print(f"[red]OOM Error: Failed to allocate tensor of shape {shape}. Skipping 3D test.[/red]")
            console.print()
            return
        else:
            raise e

    for _ in range(WARMUP_STEPS):
        _ = compiled_op(x,y).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(NUM_STEPS):
        z = compiled_op(x,y)
    z.block_until_ready()
    
    total = time.perf_counter()-start
    avg = total/NUM_STEPS
    
    GFLOPS = (MATRIX_DEPTH * GFLOPs_MULTIPLIER) / (avg * 1e9)
    TFLOPS = GFLOPS / 1000
    
    console.print(f"[green]3D Benchmark ({mode}) finished in {total:.2f}s[/green]")
    table = Table(title=f"3D Benchmark Results ({mode})", show_lines=True)
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")
    table.add_row("Mode", f"{num_cores_to_use} Core(s) - {'JIT' if num_cores_to_use == 1 else 'PMAP'}")
    table.add_row("Per-Core Tensor Size", f"{D_per_core}x{MATRIX_SIZE}x{MATRIX_SIZE}")
    table.add_row("Total Tensor Shape", str(shape))
    table.add_row("Steps", str(NUM_STEPS))
    table.add_row("Avg Time per Op (ms)", f"{avg*1000:.3f}")
    table.add_row("Total GFLOPS", f"{GFLOPS:.2f}")
    table.add_row("Total TFLOPS", f"{TFLOPS:.2f}")
    console.print(table)
    console.print()
def warning(message: str):
    console.print(Panel.fit(f"[yellow]{message}[/yellow]"))
    console.print()

def main():
    warning("Make sure to run this script in an environment with JAX and TPU support installed.")
    warning("Adjust the matrix size and steps via command-line arguments if you encounter OOM errors.")
    warning("Example: python3 tpus-benchmark_v3.py -w 5 -m 500 -mxs 8192 -md 64")
    check_dependencies()
    list_jax_devices()
    get_system_info()


    console.print(Panel.fit("[green]Running Test 1: 1-Core (JIT) Benchmark[/green]"))
    benchmark_jax_2d(num_cores_to_use=1)
    benchmark_jax_3d(num_cores_to_use=1)

    if NUM_AVAILABLE_CORES >= 4:
        console.print(Panel.fit("[green]Running Test 2: 4-Core (PMAP) Benchmark[/green]"))
        benchmark_jax_2d(num_cores_to_use=4)
        benchmark_jax_3d(num_cores_to_use=4)
    else:
        console.print(Panel.fit(f"[yellow]Skipping 4-Core test (Only {NUM_AVAILABLE_CORES} cores available).[/yellow]"))
        console.print()

    if NUM_AVAILABLE_CORES > 1 and NUM_AVAILABLE_CORES != 4:
        console.print(Panel.fit(f"[green]Running Test 3: All-Core ({NUM_AVAILABLE_CORES}-Core PMAP) Benchmark[/green]"))
        benchmark_jax_2d(num_cores_to_use=NUM_AVAILABLE_CORES)
        benchmark_jax_3d(num_cores_to_use=NUM_AVAILABLE_CORES)
    elif NUM_AVAILABLE_CORES == 1:
        console.print(Panel.fit("[yellow]Skipping All-Core test (Only 1 core available).[/yellow]"))
        console.print()
    elif NUM_AVAILABLE_CORES == 4:
        console.print(Panel.fit("[yellow]Skipping All-Core test (Already ran as 4-Core test).[/yellow]"))
        console.print()


if __name__ == "__main__":
    main()
