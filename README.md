
-----

# JAX TPU Benchmark 🚀

This project provides a comprehensive benchmark script designed to measure and analyze the performance of JAX on Google Cloud TPUs. It focuses on 2D matrix and 3D tensor computations, offering insights into performance scaling from single-core (JIT) to multi-core (PMAP) operations.
- I was invited to participate in the TPU research and development for the TRC project.
- Since there is a lot of competition for resources here, I got v4-8.
- Actually, there are other chips but I can't ask for them because they are full.
## ✨ Key Features

  * **Dual-Mode Benchmarking:** Tests both 2D matrix operations (`jnp.dot`) and 3D tensor operations (`jnp.matmul`) to simulate different types of workloads.
  * **Multi-Core Scaling Analysis:** Automatically runs benchmarks on a single core (using `jax.jit`) and scales up to 4 cores and all available cores (using `jax.pmap`) to evaluate parallel processing efficiency.
  * **System & Device Introspection:**
      * Gathers detailed system information, including OS, CPU, Python version, and total system RAM.
      * Lists all available JAX devices, their types (e.g., TPU), platform, and available accelerator memory.
  * **Configurable Workloads:** Allows users to specify workload parameters via command-line arguments:
      * `-w` / `--warmup`: Number of warmup steps.
      * `-m` / `--steps`: Number of benchmark steps to average.
      * `-mxs` / `--matrix_size`: The `N` dimension for `(N, N)` matrices.
      * `-md` / `--matrix_depth`: The `D` dimension for `(D, N, N)` tensors.
  * **Rich Reporting:**
      * Uses the `rich` library to print beautifully formatted tables and panels to the console for easy reading.
      * Reports key metrics including average operation time (ms) and total TFLOPS (TeraFLOPs per second).
  * **Automatic Plot Generation:**
      * At the end of the benchmark, it automatically generates a PNG plot (`tpu_benchmark_results.png`).
      * This plot visualizes TFLOPS and Avg. Time (ms) against the number of cores used, providing a clear comparison of 2D vs. 3D performance scaling.
  * **Robust Error Handling:** Includes graceful error handling for Out-of-Memory (OOM) exceptions, particularly for large 3D tensors. If an OOM error occurs, it suggests alternative `--matrix_depth` values to try.
  * **Dependency Checking:** A utility script (`utils/check_deps.py`) verifies that all required Python libraries (`jax`, `rich`, `psutil`) are installed before running.

-----

## 🛠️ Installation

The `install.sh` script provides the necessary commands to set up the environment for a Google Cloud TPU VM. It installs `jax` with TPU support, `torch_xla`, and the other required Python packages.

```bash
# Grant execution permissions
chmod +x install.sh

# Run the installer
./install.sh

source .venv/bin/activate
```

**Key libraries installed:**

  * `jax[tpu]`
  * `torch_xla`
  * `psutil` (for system info)
  * `rich` (for console UI)
  * `matplotlib` & `pandas` (for plotting results)

-----

## 🚀 How to Run

After installing the dependencies, you can run the main benchmark script.

**Default execution:**
This will run the benchmark with default settings (10 warmup, 1000 steps, 16384 matrix size, 128 matrix depth).

```bash
python3 tpus_benchmark_v3.py
```

**Custom execution:**
This example runs a lighter workload with 5 warmup steps, 500 test steps, an 8192x8192 matrix size, and a depth of 64.

```bash
python3 tpus_benchmark_v3.py -w 5 -m 500 -mxs 8192 -md 10
```

**Note:** The `--matrix_depth` (`-md`) value must be divisible by the number of cores being tested (e.g., 1, 4, and 8 if you have a TPU v4-8). The script will skip tests that do not meet this requirement.


## ⚠️ Parameter Warnings

- Setting command-line parameters is critical and can cause tests to fail
- Configuration requires careful reading of Google Cloud documentation to avoid errors
1.  **Out of Memory (OOM):**

      * Setting `-mxs` (matrix\_size) or `-md` (matrix\_depth) **too high** can cause your TPU/VM to run out of memory (OOM). The script will attempt to catch this error and skip the test, but it is best to start with lower values if you are unsure.
      * The script will suggest smaller `-md` values to try if an OOM occurs during the 3D test.

2.  **Matrix Depth (`-md`) Constraints:**

      * The `-md` (matrix\_depth) value **must be divisible by the number of cores being tested** (e.g., 1, 4, and 8 if you are using a TPU v4-8).
      * If the value is not divisible, the script will automatically skip the 3D (PMAP) test for that specific core count.
      * *Example:* If you use an 8-core TPU and set `-md 64`, all tests will run (64/1, 64/4, 64/8). But if you set `-md 100`, the script will only run the 1-core and 4-core tests, skipping the 8-core test.
