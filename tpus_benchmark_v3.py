#!/usr/bin/env python
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import time
import platform
import psutil
import sys
import argparse
import math
import traceback
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from utils.check_deps import check_dependencies 
from utils.jax_devices import list_jax_devices 

try:
    from utils.plt import plot_results
    PLOTTING_ENABLED = True
except Exception:
    PLOTTING_ENABLED = False

console = Console()

parser = argparse.ArgumentParser(description="JAX TPU Benchmark (improved error handling & --max-cores)")
parser.add_argument("-w", "--warmup", type=int, default=10)
parser.add_argument("-m", "--steps", type=int, default=1000)
parser.add_argument("-mxs", "--matrix_size", type=int, default=16384)
parser.add_argument("-md", "--matrix_depth", type=int, default=128)
parser.add_argument("-c", "--conv_size", type=int, default=256)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("--precision", type=str, default="float32", choices=["float32", "bfloat16"])
parser.add_argument("--max_cores", type=int, default=0, help="Max cores to test (0=auto up to available)")

args = parser.parse_args()

WARMUP_STEPS = max(0, args.warmup)
NUM_STEPS = max(1, args.steps)
MATRIX_SIZE = max(1, args.matrix_size)
MATRIX_DEPTH = max(1, args.matrix_depth)
CONV_SIZE = max(1, args.conv_size)
BATCH_SIZE = max(1, args.batch_size)
PRECISION = jnp.float32 if args.precision == "float32" else jnp.bfloat16

N = MATRIX_SIZE

# Heuristic GFLOPs estimate for the base dense op (kept from your original)
GFLOPs_BASE_OPERATION = (2 * N**3 * 2)
GFLOPs_MULTIPLIER = GFLOPs_BASE_OPERATION * 1.1

FFT_FLOPS_2D_BASE = 10 * (N ** 2) * math.log2(N) if N > 1 else 0
FFT_FLOPS_3D_BASE_PER_DEPTH = 15 * (N ** 2) * math.log2(N) if N > 1 else 0
FFT_FLOPS_3D_BASE = FFT_FLOPS_3D_BASE_PER_DEPTH * MATRIX_DEPTH

def safe_device_count():
    try:
        cnt = jax.device_count()
        return int(cnt)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not get JAX device count: {e}[/yellow]")
        return 0

try:
    NUM_AVAILABLE_CORES = safe_device_count()
except Exception as e:
    console.print(f"[red]Error initializing JAX device count: {e}[/red]")
    NUM_AVAILABLE_CORES = 0

console.print(f"[cyan]2D Matrix Size:[/cyan] {MATRIX_SIZE}x{MATRIX_SIZE}")
console.print(f"[cyan]3D Tensor Size:[/cyan] {MATRIX_DEPTH}x{MATRIX_SIZE}x{MATRIX_SIZE}")
console.print(f"[cyan]Conv Input Size:[/cyan] {BATCH_SIZE}x{CONV_SIZE}x{CONV_SIZE}x3")
console.print(f"[cyan]2D FFT Matrix Size:[/cyan] {MATRIX_SIZE}x{MATRIX_SIZE}")
console.print(f"[cyan]3D FFT Tensor Size:[/cyan] {MATRIX_DEPTH}x{MATRIX_SIZE}x{MATRIX_SIZE}")
console.print(f"[cyan]Precision:[/cyan] {args.precision}")
console.print(Panel.fit(f"[bold cyan]Found {NUM_AVAILABLE_CORES} JAX devices (cores).[/bold cyan]"))

def get_system_info():
    console.print(Panel.fit("[cyan]Collecting System Information...[/cyan]"))
    try:
        devices = jax.devices()
    except Exception as e:
        console.print(f"[red]Error initializing JAX devices: {e}[/red]")
        console.print("[yellow]Please ensure JAX and a TPU/GPU backend are correctly installed.[/yellow]")
        devices = None

    table = Table(title="System Information", show_header=False, expand=True)
    table.add_row("OS", f"{platform.system()} {platform.release()} ({platform.version()})")
    table.add_row("Machine", platform.machine())
    table.add_row("Processor", platform.processor() or "Unknown")
    table.add_row("Python Version", platform.python_version())
    try:
        table.add_row("CPU Cores", f"{psutil.cpu_count(logical=True)} logical, {psutil.cpu_count(logical=False)} physical")
        table.add_row("Total System RAM (CPU)", f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    except Exception:
        table.add_row("CPU & RAM", "N/A")

    try:
        if devices:
            dev0 = devices[0]
            accelerator_info = f"{dev0.platform.upper()} ({getattr(dev0, 'device_kind', 'Unknown')})"
            table.add_row("JAX Accelerator", accelerator_info)
            table.add_row("JAX Device Count", str(len(devices)))
            for i, dev in enumerate(devices):
                try:
                    mem_bytes = dev.memory_stats().get("bytes_limit", None) if hasattr(dev, "memory_stats") else None
                    if mem_bytes:
                        table.add_row(f"Accelerator Memory (Device {i})", f"{round(mem_bytes / (1024**3), 2)} GB")
                    else:
                        table.add_row(f"Accelerator Memory (Device {i})", "N/A")
                except Exception:
                    table.add_row(f"Accelerator Memory (Device {i})", "N/A")
        else:
            table.add_row("JAX Accelerator", "None (CPU or JAX init failed)")
    except Exception as e:
        table.add_row("JAX Accelerator", f"JAX device check failed: {e}")

    console.print(table)
    console.print()

# --- JIT / PMAP ops (as before) ---
@jax.jit
def op_2d(a, b):
    C = jnp.dot(a, b)
    D = jnp.tanh(C) + jnp.sin(C / (jnp.log(jnp.abs(a[0, 0]) + 1) * 2 + 1))
    E = jnp.dot(a, D)
    F = jnp.log1p(jnp.abs(E)) + jnp.exp(b * 0.001)
    return jnp.square(F)

@jax.jit
def op_3d(a, b):
    C = jnp.matmul(a, b)
    D = jnp.tanh(C) + jnp.sin(C / (jnp.log(jnp.abs(a[0, 0, 0]) + 1) * 2 + 1))
    E = jnp.matmul(a, D)
    F = jnp.log1p(jnp.abs(E)) + jnp.exp(b * 0.001)
    return jnp.square(F)

@jax.jit
def op_conv(x, kernel):
    conv_out = lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=lax.ConvDimensionNumbers(
            lhs_spec=(0, 1, 2, 3),
            rhs_spec=(3, 0, 1, 2),
            out_spec=(0, 1, 2, 3)
        )
    )
    activated = jnp.tanh(conv_out)
    return jnp.sum(activated ** 2)

@jax.jit
def bandwidth_test(arr):
    copied = arr
    for _ in range(10):
        copied = jnp.copy(copied)
    reduced = jnp.sum(copied)
    return reduced

@jax.jit
def op_fft_2d(a):
    f = jnp.fft.fftn(a.astype(PRECISION))
    reconstructed = jnp.fft.ifftn(f).real
    return jnp.sum(jnp.abs(reconstructed - a) ** 2)

@jax.jit
def op_fft_3d(a):
    f = jnp.fft.fftn(a.astype(PRECISION))
    reconstructed = jnp.fft.ifftn(f).real
    return jnp.sum(jnp.abs(reconstructed - a) ** 2)

def benchmark_jax_2d(num_cores_to_use: int):
    mode = f"{num_cores_to_use}-Core ({'JIT' if num_cores_to_use == 1 else 'PMAP'})"
    console.print(Panel.fit(f"[magenta]JAX 2D Matrix Benchmark ({mode})[/magenta]"))

    key = jax.random.PRNGKey(0)

    try:
        if num_cores_to_use == 1:
            compiled_op = op_2d
            shape = (MATRIX_SIZE, MATRIX_SIZE)
            keys = jax.random.split(key, 2)
            x = jax.random.normal(keys[0], shape, dtype=PRECISION)
            y = jax.random.normal(keys[1], shape, dtype=PRECISION)
        else:
            # pmap over leading axis (cores)
            compiled_op = jax.pmap(op_2d)
            shape = (num_cores_to_use, MATRIX_SIZE, MATRIX_SIZE)
            shape_per_core = (MATRIX_SIZE, MATRIX_SIZE)

            key_x, key_y = jax.random.split(key, 2)
            keys_x = jax.random.split(key_x, num_cores_to_use)
            keys_y = jax.random.split(key_y, num_cores_to_use)

            x = jax.vmap(lambda k: jax.random.normal(k, shape_per_core, dtype=PRECISION))(keys_x)
            y = jax.vmap(lambda k: jax.random.normal(k, shape_per_core, dtype=PRECISION))(keys_y)

        console.print(f"Allocating 2D Tensors with shape: {shape}")
        x.block_until_ready()
        y.block_until_ready()

        for _ in range(WARMUP_STEPS):
            _ = compiled_op(x, y).block_until_ready()

        start = time.perf_counter()
        for _ in range(NUM_STEPS):
            z = compiled_op(x, y)
        z.block_until_ready()

        total = time.perf_counter() - start
        avg = total / NUM_STEPS

        GFLOPS = (num_cores_to_use * GFLOPs_MULTIPLIER) / (avg * 1e9)
        TFLOPS = GFLOPS / 1000

    except (RuntimeError) as e:
        error_msg = str(e).upper()
        if "RESOURCE_EXHAUSTED" in error_msg or "OOM" in error_msg:
            console.print(f"[red]OOM Error: Failed to allocate or execute 2D tensor operations.[/red]")
            console.print(f"[yellow]Try reducing --matrix_size (-mxs). Skipping 2D test.[/yellow]")
            console.print()
            return None
        else:
            console.print(f"[red]Unexpected runtime error in 2D benchmark: {e}[/red]")
            console.print(traceback.format_exc())
            return None
    except Exception as e:
        console.print(f"[red]Unhandled error in 2D benchmark: {e}[/red]")
        console.print(traceback.format_exc())
        return None

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

    return {
        'test': '2D',
        'cores': num_cores_to_use,
        'tflops': TFLOPS,
        'avg_ms': avg * 1000
    }

def benchmark_jax_3d(num_cores_to_use: int):
    mode = f"{num_cores_to_use}-Core ({'JIT' if num_cores_to_use == 1 else 'PMAP'})"
    console.print(Panel.fit(f"[magenta]JAX 3D Tensor Benchmark ({mode})[/magenta]"))

    if num_cores_to_use > 1 and MATRIX_DEPTH % num_cores_to_use != 0:
        console.print(f"[yellow]Skipping 3D test for {num_cores_to_use} cores because MATRIX_DEPTH ({MATRIX_DEPTH}) is not divisible by cores.[/yellow]")
        console.print()
        return None

    D_per_core = MATRIX_DEPTH // num_cores_to_use
    key = jax.random.PRNGKey(42)

    try:
        if num_cores_to_use == 1:
            compiled_op = op_3d
            shape = (D_per_core, MATRIX_SIZE, MATRIX_SIZE)
            keys = jax.random.split(key, 2)
            x = jax.random.normal(keys[0], shape, dtype=PRECISION)
            y = jax.random.normal(keys[1], shape, dtype=PRECISION)
        else:
            compiled_op = jax.pmap(op_3d)
            shape = (num_cores_to_use, D_per_core, MATRIX_SIZE, MATRIX_SIZE)
            shape_per_core = (D_per_core, MATRIX_SIZE, MATRIX_SIZE)

            key_x, key_y = jax.random.split(key, 2)
            keys_x = jax.random.split(key_x, num_cores_to_use)
            keys_y = jax.random.split(key_y, num_cores_to_use)

            x = jax.vmap(lambda k: jax.random.normal(k, shape_per_core, dtype=PRECISION))(keys_x)
            y = jax.vmap(lambda k: jax.random.normal(k, shape_per_core, dtype=PRECISION))(keys_y)

        console.print(f"Allocating 3D Tensors with shape: {shape}")
        x.block_until_ready()
        y.block_until_ready()

        for _ in range(WARMUP_STEPS):
            _ = compiled_op(x, y).block_until_ready()

        start = time.perf_counter()
        for _ in range(NUM_STEPS):
            z = compiled_op(x, y)
        z.block_until_ready()

        total = time.perf_counter() - start
        avg = total / NUM_STEPS

        GFLOPS = (MATRIX_DEPTH * GFLOPs_MULTIPLIER) / (avg * 1e9)
        TFLOPS = GFLOPS / 1000

    except (RuntimeError) as e:
        error_msg = str(e).upper()
        if "RESOURCE_EXHAUSTED" in error_msg or "OOM" in error_msg:
            console.print(f"[red]OOM Error: Failed to allocate or execute 3D tensor operations.[/red]")
            console.print(f"[yellow]The 3D MATRIX_DEPTH ({MATRIX_DEPTH}) is too large for the available accelerator memory.[/yellow]")
            # Suggest smaller depths (derived from divisors)
            original_depth = MATRIX_DEPTH
            suggestion_table = Table(title="Suggested '-md' values to try")
            suggestion_table.add_column("Command Line Flag", style="cyan")
            suggestion_table.add_column("Reason", style="dim")
            # pick divisors that produce reasonable per-core depth
            for d in [2, 4, 8, 16, 32]:
                if original_depth // d >= 1:
                    suggestion_table.add_row(f"-md {original_depth // d}", f"(Original {original_depth} // {d})")
            console.print(suggestion_table)
            console.print(f"[yellow]Skipping 3D test for {num_cores_to_use} cores.[/yellow]")
            console.print()
            return None
        else:
            console.print(f"[red]Unexpected runtime error in 3D benchmark: {e}[/red]")
            console.print(traceback.format_exc())
            return None
    except Exception as e:
        console.print(f"[red]Unhandled error in 3D benchmark: {e}[/red]")
        console.print(traceback.format_exc())
        return None

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

    return {
        'test': '3D',
        'cores': num_cores_to_use,
        'tflops': TFLOPS,
        'avg_ms': avg * 1000
    }

def benchmark_jax_conv(num_cores_to_use: int):
    mode = f"{num_cores_to_use}-Core ({'JIT' if num_cores_to_use == 1 else 'PMAP'})"
    console.print(Panel.fit(f"[magenta]JAX Convolution Benchmark ({mode})[/magenta]"))

    key = jax.random.PRNGKey(123)
    kernel_size = 3
    out_channels = 64
    in_channels = 3

    try:
        if num_cores_to_use == 1:
            compiled_op = op_conv
            input_shape = (BATCH_SIZE, CONV_SIZE, CONV_SIZE, in_channels)
            kernel_shape = (kernel_size, kernel_size, in_channels, out_channels)
            keys = jax.random.split(key, 2)
            x = jax.random.normal(keys[0], input_shape, dtype=PRECISION)
            kernel = jax.random.normal(keys[1], kernel_shape, dtype=PRECISION)
        else:
            batch_per_core = BATCH_SIZE // num_cores_to_use
            if batch_per_core == 0:
                console.print(f"[yellow]Skipping Conv for {num_cores_to_use} cores: batch size {BATCH_SIZE} too small to split.[/yellow]")
                return None
            input_shape_per_core = (batch_per_core, CONV_SIZE, CONV_SIZE, in_channels)
            kernel_shape = (kernel_size, kernel_size, in_channels, out_channels)
            key_x, key_k = jax.random.split(key, 2)
            keys_x = jax.random.split(key_x, num_cores_to_use)
            x_per_core = jax.vmap(lambda k: jax.random.normal(k, input_shape_per_core, dtype=PRECISION))(keys_x)
            # for conv we keep kernel broadcasted (None)
            x = jnp.concatenate(x_per_core, axis=0)
            kernel = jax.random.normal(key_k, kernel_shape, dtype=PRECISION)
            compiled_op = jax.pmap(op_conv, in_axes=(0, None))

        console.print(f"Allocating Conv Input: {x.shape}, Kernel: {kernel.shape}")
        x.block_until_ready()
        kernel.block_until_ready()

        for _ in range(WARMUP_STEPS):
            _ = compiled_op(x, kernel).block_until_ready()

        start = time.perf_counter()
        for _ in range(NUM_STEPS):
            z = compiled_op(x, kernel)
        z.block_until_ready()

        total = time.perf_counter() - start
        avg = total / NUM_STEPS

        out_h, out_w = CONV_SIZE, CONV_SIZE
        conv_gflops = 2 * BATCH_SIZE * out_h * out_w * out_channels * in_channels * kernel_size * kernel_size
        GFLOPS = conv_gflops / (avg * 1e9)
        TFLOPS = GFLOPS / 1000

    except (RuntimeError) as e:
        error_msg = str(e).upper()
        if "RESOURCE_EXHAUSTED" in error_msg or "OOM" in error_msg:
            console.print(f"[red]OOM Error: Failed to allocate or execute Conv operations.[/red]")
            console.print(f"[yellow]Try reducing --conv_size (-c) or --batch_size (-b). Skipping Conv test.[/yellow]")
            console.print()
            return None
        else:
            console.print(f"[red]Unexpected runtime error in Conv benchmark: {e}[/red]")
            console.print(traceback.format_exc())
            return None
    except Exception as e:
        console.print(f"[red]Unhandled error in Conv benchmark: {e}[/red]")
        console.print(traceback.format_exc())
        return None

    console.print(f"[green]Conv Benchmark ({mode}) finished in {total:.2f}s[/green]")
    table = Table(title=f"Conv Benchmark Results ({mode})", show_lines=True)
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")
    table.add_row("Mode", f"{num_cores_to_use} Core(s) - {'JIT' if num_cores_to_use == 1 else 'PMAP'}")
    table.add_row("Input Shape", str((BATCH_SIZE, CONV_SIZE, CONV_SIZE, in_channels)))
    table.add_row("Kernel Shape", str(kernel_shape))
    table.add_row("Steps", str(NUM_STEPS))
    table.add_row("Avg Time per Op (ms)", f"{avg*1000:.3f}")
    table.add_row("Total GFLOPS", f"{GFLOPS:.2f}")
    table.add_row("Total TFLOPS", f"{TFLOPS:.2f}")
    console.print(table)
    console.print()

    return {
        'test': 'Conv',
        'cores': num_cores_to_use,
        'tflops': TFLOPS,
        'avg_ms': avg * 1000
    }

def benchmark_bandwidth(num_cores_to_use: int):
    mode = f"{num_cores_to_use}-Core ({'JIT' if num_cores_to_use == 1 else 'PMAP'})"
    console.print(Panel.fit(f"[magenta]JAX Memory Bandwidth Benchmark ({mode})[/magenta]"))

    # Cap the bandwidth size to avoid accidental OOM on small devices.
    # Original size was huge: 1024*1024*256 (~268M elements) -> can be too big
    MAX_ELEM_PER_CORE = 64 * 1024 * 1024  # 64M elements per core (heuristic cap)
    requested_total = 1024 * 1024 * 256
    if num_cores_to_use > 1:
        per_core = min(MAX_ELEM_PER_CORE, requested_total // num_cores_to_use)
    else:
        per_core = min(MAX_ELEM_PER_CORE, requested_total)
    bandwidth_size = int(per_core)

    key = jax.random.PRNGKey(456)

    try:
        if num_cores_to_use == 1:
            compiled_op = bandwidth_test
            shape = (bandwidth_size,)
            x = jax.random.normal(key, shape, dtype=PRECISION)
        else:
            compiled_op = jax.pmap(bandwidth_test)
            shape_per_core = (bandwidth_size,)
            keys = jax.random.split(key, num_cores_to_use)
            x = jax.vmap(lambda k: jax.random.normal(k, shape_per_core, dtype=PRECISION))(keys)

        console.print(f"Allocating Bandwidth Array: {x.shape} elements (per-core: {bandwidth_size})")
        x.block_until_ready()

        for _ in range(WARMUP_STEPS):
            _ = compiled_op(x).block_until_ready()

        start = time.perf_counter()
        for _ in range(NUM_STEPS):
            z = compiled_op(x)
        z.block_until_ready()

        total = time.perf_counter() - start
        avg = total / NUM_STEPS

        bytes_per_element = 4 if PRECISION == jnp.float32 else 2
        total_bytes = x.size * bytes_per_element * 10 * NUM_STEPS
        bandwidth_gbs = (total_bytes / (1024**3)) / total

    except (RuntimeError) as e:
        error_msg = str(e).upper()
        if "RESOURCE_EXHAUSTED" in error_msg or "OOM" in error_msg:
            console.print(f"[red]OOM Error: Failed to allocate bandwidth array.[/red]")
            console.print(f"[yellow]Try smaller array size. Skipping bandwidth test.[/yellow]")
            console.print()
            return None
        else:
            console.print(f"[red]Unexpected runtime error in Bandwidth benchmark: {e}[/red]")
            console.print(traceback.format_exc())
            return None
    except Exception as e:
        console.print(f"[red]Unhandled error in Bandwidth benchmark: {e}[/red]")
        console.print(traceback.format_exc())
        return None

    console.print(f"[green]Bandwidth Benchmark ({mode}) finished in {total:.2f}s[/green]")
    table = Table(title=f"Bandwidth Benchmark Results ({mode})", show_lines=True)
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")
    table.add_row("Mode", f"{num_cores_to_use} Core(s) - {'JIT' if num_cores_to_use == 1 else 'PMAP'}")
    table.add_row("Array Elements per Core", str(bandwidth_size))
    table.add_row("Steps", str(NUM_STEPS))
    table.add_row("Avg Time per Op (ms)", f"{avg*1000:.3f}")
    table.add_row("Total Bandwidth", f"{bandwidth_gbs:.2f} GB/s")
    console.print(table)
    console.print()

    return {
        'test': 'Bandwidth',
        'cores': num_cores_to_use,
        'bandwidth_gbs': bandwidth_gbs,
        'avg_ms': avg * 1000
    }

def benchmark_jax_fft_2d(num_cores_to_use: int):
    mode = f"{num_cores_to_use}-Core ({'JIT' if num_cores_to_use == 1 else 'PMAP'})"
    console.print(Panel.fit(f"[magenta]JAX 2D FFT Benchmark ({mode})[/magenta]"))

    key = jax.random.PRNGKey(789)

    try:
        if num_cores_to_use == 1:
            compiled_op = op_fft_2d
            shape = (MATRIX_SIZE, MATRIX_SIZE)
            x = jax.random.normal(key, shape, dtype=PRECISION)
        else:
            compiled_op = jax.pmap(op_fft_2d)
            shape = (num_cores_to_use, MATRIX_SIZE, MATRIX_SIZE)
            shape_per_core = (MATRIX_SIZE, MATRIX_SIZE)

            keys = jax.random.split(key, num_cores_to_use)
            x = jax.vmap(lambda k: jax.random.normal(k, shape_per_core, dtype=PRECISION))(keys)

        console.print(f"Allocating 2D FFT Input with shape: {shape}")
        x.block_until_ready()

        for _ in range(WARMUP_STEPS):
            _ = compiled_op(x).block_until_ready()

        start = time.perf_counter()
        for _ in range(NUM_STEPS):
            z = compiled_op(x)
        z.block_until_ready()

        total = time.perf_counter() - start
        avg = total / NUM_STEPS

        flops_per_op = FFT_FLOPS_2D_BASE * num_cores_to_use
        GFLOPS = flops_per_op / (avg * 1e9)
        TFLOPS = GFLOPS / 1000

    except (RuntimeError) as e:
        error_msg = str(e).upper()
        if "RESOURCE_EXHAUSTED" in error_msg or "OOM" in error_msg:
            console.print(f"[red]OOM Error: Failed to allocate or execute 2D FFT operations.[/red]")
            console.print(f"[yellow]Try reducing --matrix_size (-mxs). Skipping 2D FFT test.[/yellow]")
            console.print()
            return None
        else:
            console.print(f"[red]Unexpected runtime error in 2D FFT benchmark: {e}[/red]")
            console.print(traceback.format_exc())
            return None
    except Exception as e:
        console.print(f"[red]Unhandled error in 2D FFT benchmark: {e}[/red]")
        console.print(traceback.format_exc())
        return None

    console.print(f"[green]2D FFT Benchmark ({mode}) finished in {total:.2f}s[/green]")
    table = Table(title=f"2D FFT Benchmark Results ({mode})", show_lines=True)
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")
    table.add_row("Mode", f"{num_cores_to_use} Core(s) - {'JIT' if num_cores_to_use == 1 else 'PMAP'}")
    table.add_row("Per-Core Matrix Size", f"{MATRIX_SIZE}x{MATRIX_SIZE}")
    table.add_row("Total Shape", str(shape))
    table.add_row("Steps", str(NUM_STEPS))
    table.add_row("Avg Time per Op (ms)", f"{avg*1000:.3f}")
    table.add_row("Approx FLOPS per Op", f"{flops_per_op:.2e}")
    table.add_row("Total GFLOPS", f"{GFLOPS:.2f}")
    table.add_row("Total TFLOPS", f"{TFLOPS:.2f}")
    console.print(table)
    console.print()

    return {
        'test': '2D_FFT',
        'cores': num_cores_to_use,
        'tflops': TFLOPS,
        'avg_ms': avg * 1000
    }

def benchmark_jax_fft_3d(num_cores_to_use: int):
    mode = f"{num_cores_to_use}-Core ({'JIT' if num_cores_to_use == 1 else 'PMAP'})"
    console.print(Panel.fit(f"[magenta]JAX 3D FFT Benchmark ({mode})[/magenta]"))

    if num_cores_to_use > 1 and MATRIX_DEPTH % num_cores_to_use != 0:
        console.print(f"[yellow]Skipping 3D FFT for {num_cores_to_use} cores because MATRIX_DEPTH ({MATRIX_DEPTH}) is not divisible by cores.[/yellow]")
        console.print()
        return None

    D_per_core = MATRIX_DEPTH // num_cores_to_use
    key = jax.random.PRNGKey(1011)

    try:
        if num_cores_to_use == 1:
            compiled_op = op_fft_3d
            shape = (D_per_core, MATRIX_SIZE, MATRIX_SIZE)
            x = jax.random.normal(key, shape, dtype=PRECISION)
        else:
            compiled_op = jax.pmap(op_fft_3d)
            shape = (num_cores_to_use, D_per_core, MATRIX_SIZE, MATRIX_SIZE)
            shape_per_core = (D_per_core, MATRIX_SIZE, MATRIX_SIZE)

            keys = jax.random.split(key, num_cores_to_use)
            x = jax.vmap(lambda k: jax.random.normal(k, shape_per_core, dtype=PRECISION))(keys)

        console.print(f"Allocating 3D FFT Input with shape: {shape}")
        x.block_until_ready()

        for _ in range(WARMUP_STEPS):
            _ = compiled_op(x).block_until_ready()

        start = time.perf_counter()
        for _ in range(NUM_STEPS):
            z = compiled_op(x)
        z.block_until_ready()

        total = time.perf_counter() - start
        avg = total / NUM_STEPS

        flops_per_op = FFT_FLOPS_3D_BASE
        GFLOPS = flops_per_op / (avg * 1e9)
        TFLOPS = GFLOPS / 1000

    except (RuntimeError) as e:
        error_msg = str(e).upper()
        if "RESOURCE_EXHAUSTED" in error_msg or "OOM" in error_msg:
            console.print(f"[red]OOM Error: Failed to allocate or execute 3D FFT operations.[/red]")
            console.print(f"[yellow]Try reducing --matrix_size (-mxs) or --matrix_depth (-md). Skipping 3D FFT test.[/yellow]")
            console.print()
            return None
        else:
            console.print(f"[red]Unexpected runtime error in 3D FFT benchmark: {e}[/red]")
            console.print(traceback.format_exc())
            return None
    except Exception as e:
        console.print(f"[red]Unhandled error in 3D FFT benchmark: {e}[/red]")
        console.print(traceback.format_exc())
        return None

    console.print(f"[green]3D FFT Benchmark ({mode}) finished in {total:.2f}s[/green]")
    table = Table(title=f"3D FFT Benchmark Results ({mode})", show_lines=True)
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")
    table.add_row("Mode", f"{num_cores_to_use} Core(s) - {'JIT' if num_cores_to_use == 1 else 'PMAP'}")
    table.add_row("Per-Core Tensor Size", f"{D_per_core}x{MATRIX_SIZE}x{MATRIX_SIZE}")
    table.add_row("Total Shape", str(shape))
    table.add_row("Steps", str(NUM_STEPS))
    table.add_row("Avg Time per Op (ms)", f"{avg*1000:.3f}")
    table.add_row("Approx FLOPS per Op", f"{flops_per_op:.2e}")
    table.add_row("Total GFLOPS", f"{GFLOPS:.2f}")
    table.add_row("Total TFLOPS", f"{TFLOPS:.2f}")
    console.print(table)
    console.print()

    return {
        'test': '3D_FFT',
        'cores': num_cores_to_use,
        'tflops': TFLOPS,
        'avg_ms': avg * 1000
    }

def warning(message: str):
    console.print(Panel.fit(f"[yellow]{message}[/yellow]"))
    console.print()

def compute_core_candidates(max_test_cores: int):
    """
    Build a filtered list of core counts to test:
    - Always include 1
    - include powers of two up to max_test_cores
    - also include max_test_cores itself if it's <= available and not already in list
    - filter out candidates that obviously won't work for 3D/conv (later checks still apply per-test)
    """
    candidates = set()
    candidates.add(1)
    i = 0
    while True:
        p = 2 ** i
        if p > max_test_cores:
            break
        candidates.add(p)
        i += 1
        if i > 20:
            break
    if max_test_cores not in candidates and max_test_cores >= 1:
        candidates.add(max_test_cores)
    # Sort ascending
    core_list = sorted(list(candidates))
    # Cap to available devices
    core_list = [c for c in core_list if c <= NUM_AVAILABLE_CORES] if NUM_AVAILABLE_CORES > 0 else core_list
    # Final safety: drop any core values <= 0
    core_list = [c for c in core_list if c > 0]
    return core_list

def benchmark_multiple_cores(max_cores: int):
    """
    Run benchmarks for multiple cores: 1, 2, 4, 8, ... up to max_cores
    """
    console.print(Panel.fit(f"[blue]Running multi-core benchmarks up to {max_cores} cores[/blue]"))

    core_counts = compute_core_candidates(max_cores)

    tested_cores = set()
    all_results = []

    for num_cores in core_counts:
        if num_cores in tested_cores:
            continue
        tested_cores.add(num_cores)
        console.print(Panel.fit(f"[green]Running benchmark for {num_cores} cores[/green]"))

        res_2d = benchmark_jax_2d(num_cores)
        res_3d = benchmark_jax_3d(num_cores)
        res_conv = benchmark_jax_conv(num_cores)
        res_fft_2d = benchmark_jax_fft_2d(num_cores)
        res_fft_3d = benchmark_jax_fft_3d(num_cores)
        res_bw = benchmark_bandwidth(num_cores)

        if res_2d: all_results.append(res_2d)
        if res_3d: all_results.append(res_3d)
        if res_conv: all_results.append(res_conv)
        if res_fft_2d: all_results.append(res_fft_2d)
        if res_fft_3d: all_results.append(res_fft_3d)
        if res_bw: all_results.append(res_bw)

    return all_results

def main():
    warning("Make sure to run this script in an environment with JAX and TPU/GPU support installed.")
    warning("Adjust the matrix size and steps via command-line arguments if you encounter OOM errors.")
    warning("New options: -c for conv_size, -b for batch_size, --precision for float32/bfloat16")
    warning("FFT uses --matrix_size and --matrix_depth for input size.")
    warning("Use --max-cores N to limit max cores tested (e.g., --max-cores 16).")
    warning("Example: python3 tpus_benchmark_v3.py -w 5 -m 500 -mxs 4096 -md 8 -c 128 -b 16 --precision bfloat16 --max-cores 16")

    try:
        check_dependencies()
    except Exception as e:
        console.print(f"[red]Dependency check failed: {e}[/red]")
        console.print(traceback.format_exc())

    try:
        list_jax_devices()
    except Exception as e:
        console.print(f"[yellow]Warning: list_jax_devices() failed: {e}[/yellow]")

    get_system_info()
    all_results = []

    console.print(Panel.fit("[green]Preparing benchmarks for available core counts.[/green]"))

    # Determine max_test_cores
    if args.max_cores > 0:
        max_test_cores = args.max_cores
        if NUM_AVAILABLE_CORES > 0 and args.max_cores > NUM_AVAILABLE_CORES:
            console.print(f"[yellow]Requested --max-cores {args.max_cores} is greater than discovered devices ({NUM_AVAILABLE_CORES}). Using {NUM_AVAILABLE_CORES} instead.[/yellow]")
            max_test_cores = NUM_AVAILABLE_CORES
    else:
        max_test_cores = NUM_AVAILABLE_CORES if NUM_AVAILABLE_CORES > 0 else 1

    if max_test_cores <= 0:
        console.print("[yellow]No JAX devices detected; falling back to single-core (JIT) tests on CPU.[/yellow]")
        max_test_cores = 1

    core_counts = compute_core_candidates(max_test_cores)

    if not core_counts:
        core_counts = [1]

    console.print(f"[cyan]Core candidate list to test:[/cyan] {core_counts}")

    try:
        all_results = benchmark_multiple_cores(max_test_cores)
    except KeyboardInterrupt:
        console.print("\n[red]Benchmark run interrupted by user (KeyboardInterrupt).[/red]")
        console.print("[yellow]Will attempt to save/plot whatever results were collected so far.[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error during benchmarking loop: {e}[/red]")
        console.print(traceback.format_exc())

    console.print()
    if all_results:
        console.print(Panel.fit("[blue]Generating benchmark plot...[/blue]"))
        if PLOTTING_ENABLED:
            try:
                plot_filename = "tpu_benchmark_results.png"
                plot_results(all_results, plot_filename)
                console.print(f"[green]Benchmark plot saved to [bold]{plot_filename}[/bold][/green]")
            except Exception as e:
                console.print(f"[red]An error occurred during plotting: {e}[/red]")
                console.print(traceback.format_exc())
        else:
            console.print("[yellow]Plotting skipped: 'matplotlib' or 'pandas' not found.[/yellow]")
            console.print("[yellow]Please run: [bold]pip install matplotlib pandas[/bold][/yellow]")
    else:
        console.print("[yellow]No benchmark results collected. Skipping plot generation.[/yellow]")

    console.print("[green]Benchmark script finished.[/green]")

if __name__ == "__main__":
    main()
