-----

# JAX TPU Benchmark & Physics Simulation 🚀

This project provides a comprehensive benchmark script (`tpus_benchmark_single-host_workload.py`) designed to measure and analyze the performance of JAX on Google Cloud TPUs. It has been expanded to test a wider variety of computations, including 2D/3D matrix operations, 2D/3D FFT, and memory bandwidth, offering insights into performance scaling from single-core (JIT) to multi-core (PMAP) operations.

This project also includes several physics simulation scripts accelerated with JAX, such as an N-Body black hole merger simulation (`nbody_bh_merger_sim_single-host_workload.py`), a Molecular Dynamics simulation (`molecular_dynamics_jax_single-host_workload.py`), and a three-particle simulation in a non-uniform EM field (`three_particles_em_nonuni_single-host_workload.py`).

  - I was invited to participate in TPU research and development for the TRC project.
  - Due to the high competition for resources here, I received version v4-8 (right away) (the chip is included in the project, so it's free).
  - I also tested version v3-8 and v2-8 (with a $300 credit).
  - There are actually other chips, but I couldn't request them because they were already full.
  - And most importantly, you might find that the installation completes and verifies, but a runtime error occurs. This means that the TPU for that chip was not found in the hardware. This is a bug in the library, which users haven't been able to fix. I've encountered this problem several times, but within this project, what I specified works, and what I don't specify means it doesn't work. I think
  - The specified chip is a single-host workload type, so tpus\_benchmark\_single-host\_workload.py can be used for testing.
## How to Create TPU
- Select according to the TRC project support.
- And most importantly, choose the TPU software version as tpu-ubuntu2204-base.
## ✨ Key Features

  * **Multi-Mode Benchmarking:** Tests a diverse set of operations:
      * 2D Matrix Operations (`jnp.dot`)
      * 3D Tensor Operations (`jnp.matmul`)
      * 2D & 3D FFT (`jnp.fft.fftn`)
      * Memory Bandwidth (`jnp.copy`, `jnp.sum`)
  * **Physics Simulation Examples:**
      * **N-Body Black Hole Merger**: Simulates N-body dynamics (e.g., 3-body), generates gravitational waveforms, and computes Lyapunov exponents for chaos analysis using JAX ODE integration (`nbody_bh_merger_sim_single-host_workload.py`).
      * **Molecular Dynamics**: A pure JAX implementation of a 2D Lennard-Jones fluid simulation using a JIT-compiled Verlet integrator, complete with equilibration and production runs (`molecular_dynamics_jax_single-host_workload.py`).
      * **Three-Particle EM Simulation**: Simulates three particles under gravity and a non-uniform electromagnetic field (`three_particles_em_nonuni_single-host_workload.py`).
  * **Multi-Core Scaling Analysis:** Automatically runs benchmarks on a single core (using `jax.jit`) and scales up to all available cores (using `jax.pmap`) to evaluate parallel processing efficiency.
  * **System & Device Introspection:**
      * Gathers detailed system information, including OS, CPU, Python version, and total system RAM.
      * Lists all available JAX devices, their types (e.g., TPU), platform, and available accelerator memory.
  * **Configurable Workloads:** Allows users to specify workload parameters via command-line arguments:
      * `-w` / `--warmup`: Number of warmup steps.
      * `-m` / `--steps`: Number of benchmark steps to average.
      * `-mxs` / `--matrix_size`: The `N` dimension for `(N, N)` matrices.
      * `-md` / `--matrix_depth`: The `D` dimension for `(D, N, N)` tensors.
      * `--precision`: Data type (`float32` or `bfloat16`).
      * `--csv`: Output results to a CSV file.
  * **Rich Reporting:**
      * Uses the `rich` library to print beautifully formatted tables and panels to the console for easy reading.
      * Reports key metrics including average operation time (ms) and total TFLOPS.
  * **Automatic Plot Generation:**
      * At the end of the benchmark, it automatically generates a PNG plot (`tpu_benchmark_results.png`).
      * This plot visualizes TFLOPS and Avg. Time (ms) against the number of cores used, providing a clear comparison of performance scaling.
  * **Robust Error Handling:** Includes graceful error handling for Out-of-Memory (OOM) exceptions, particularly for large 3D tensors. If an OOM error occurs, it suggests alternative `--matrix_depth` values to try.
  * **Dependency Checking:** A utility script (`utils/check_deps.py`) verifies that all required Python libraries (`jax`, `rich`, `psutil`) are installed before running.

-----

## 🛠️ Installation (Check before installing whether it is python3.10.x or not.)

The script provides the necessary commands to set up the environment for a Google Cloud TPU VM. It installs Python 3.10.x, `jax` with TPU support, `torch_xla`, and the other required Python packages.

```bash
-------------- Basic installation --------------
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
sudo apt install -y python3-dev
sudo apt install -y python3.10-venv
-------------- Basic installation --------------

-------------- Check before installing whether it is python3.10.x or not. --------------
export PYTHON_VERSION="3.10.x"
export PYTHON_PATH="/opt/python-$PYTHON_VERSION"
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
tar -xf Python-$PYTHON_VERSION.tgz
cd Python-$PYTHON_VERSION
./configure --enable-optimizations --prefix=$PYTHON_PATH
make -j$(nproc)
sudo make install
cd ..
sudo rm -rf Python-$PYTHON_VERSION Python-$PYTHON_VERSION.tgz
export PATH="$PYTHON_PATH/bin:$PATH"
sudo cp $PYTHON_PATH/bin/python3.10 /usr/bin/local
sudo cp $PYTHON_PATH/bin/pip3 /usr/bin/local
sudo cp $PYTHON_PATH/bin/pip3.10 /usr/bin/local
export VENV_NAME=".venv"
"$PYTHON_PATH/bin/python3" -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"
python3 --version
-------------- Check before installing whether it is python3.10.x or not. --------------

-------------- Installing project dependencies (PyTorch/XLA, JAX/TPU, etc.)... --------------
pip install --upgrade pip
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
pip install "transformers<5.8"
pip install jax>=0.4.0 flax orbax-checkpoint clu tensorflow-datasets tensorflow-metadata protobuf
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install psutil rich matplotlib pandas jax-md scipy imageio
pip check
-------------- Installing project dependencies (PyTorch/XLA, JAX/TPU, etc.)... --------------
```

**Key libraries installed:**

  * `jax[tpu]`
  * `torch_xla[tpu]`
  * `flax`, `orbax-checkpoint`, `clu`
  * `transformers<5.8`
  * `psutil` (for system info)
  * `rich` (for console UI)
  * `matplotlib` & `pandas` (for plotting results)
  * `jax-md`
  * `scipy` (for n-body sim)

-----

## 🚀 How to Run (Single-Host Workloads)

After installing the dependencies and activating the virtual environment, you can run the main benchmark script or the physics simulations.

### 1\. Main Benchmark (`tpus_benchmark_single-host_workload.py`)

This script is for benchmarking TPU performance. It tests various operations (2D/3D Matrix, 2D/3D FFT, Bandwidth) and scaling across multiple cores (JIT vs. PMAP).

**Default execution:**
(Uses default settings: 10 warmup, 1000 steps, 16384 matrix size, 128 matrix depth)

```bash
python3 tpus_benchmark_single-host_workload.py
```

**Custom execution (lighter workload):**
(Example: 5 warmup, 500 test steps, 8192x8192 matrix size, and 64 depth)

```bash
python3 tpus_benchmark_single-host_workload.py -w 5 -m 500 -mxs 8192 -md 64
```

**Run and Export to CSV:**
(Example: Test up to 8 cores and save results to `results.csv`)

```bash
python3 tpus_benchmark_single-host_workload.py --max_cores 8 --csv results.csv
```

**All Arguments:**

  * `-w` / `--warmup` (int, default: 10): The number of "warmup" runs before starting the actual benchmark.
  * `-m` / `--steps` (int, default: 1000): The number of test iterations to average for the results.
  * `-mxs` / `--matrix_size` (int, default: 16384): The size (N) for `(N, N)` matrices.
  * `-md` / `--matrix_depth` (int, default: 128): The depth (D) for 3D tensors `(D, N, N)`.
  * `-c` / `--conv_size` (int, default: 256): The size of the convolution input.
  * `-b` / `--batch_size` (int, default: 32): The batch size.
  * `--precision` (str, default: "float32"): The data precision to use (`float32` or `bfloat16`).
  * `--max_cores` (int, default: 0): The maximum number of cores to test (0 = auto-detect all available).
  * `--csv` (str, default: None): The filename to output results to a CSV file (e.g., `--csv results.csv`).

### 2\. Physics: N-Body Black Hole Merger (`nbody_bh_merger_sim_single-host_workload.py`)

This script simulates an N-body (e.g., 3-body) black hole merger and generates gravitational waveforms (GW). This script is **interactive** and will prompt you for parameters in the console instead of using arguments.

**How to Run:**

```bash
python3 nbody_bh_merger_sim_single-host_workload.py
```

**Parameters you will be prompted for:**

  * `Number of black holes` (int, default: 3): Number of black holes (2-5 recommended).
  * `Mass of BH{i+1} (M☉)` (float, default: 30.0): Mass for each black hole.
  * `Typical initial separation` (float, default: 100.0): Typical initial distance.
  * `Typical initial velocity (v/c)` (float, default: 0.1): Typical initial velocity (as a fraction of c).
  * `Simulation time` (float, default: 200.0): Total simulation time.
  * `GW observer distance (Mpc)` (float, default: 410.0): Distance to the GW observer (in Mpc).
  * `Compute Lyapunov exponent for chaos? (y/n)` (str, default: "y"): Whether to compute the Lyapunov exponent for chaos analysis.

### 3\. Physics: Molecular Dynamics (`molecular_dynamics_jax_single-host_workload.py`)

This script simulates the molecular dynamics (MD) of a 2D Lennard-Jones fluid using pure JAX.

**Default execution:**
(N=400, 10k eq\_steps, 10k prod\_steps)

```bash
python3 molecular_dynamics_jax_single-host_workload.py
```

**Custom execution (longer run):**

```bash
python3 molecular_dynamics_jax_single-host_workload.py --prod_steps 50000 --eq_steps 20000
```

**All Arguments:**

  * `--N` (int, default: 400): Number of particles.
  * `--rho` (float, default: 0.8): Density.
  * `--kT` (float, default: 1.0): Temperature (kT).
  * `--dt` (float, default: 1e-3): Time step size.
  * `--eq_steps` (int, default: 10000): Number of equilibration steps.
  * `--prod_steps` (int, default: 10000): Number of production (simulation) steps.
  * `--sample_every` (int, default: 100): Sample the state every N steps.
  * `--seed` (int, default: 42): PRNG seed.
  * `--output` (str, default: "g\_r\_plot.png"): Output filename for the g(r) plot.

### 4\. Physics: Three-Particle EM Simulation (`three_particles_em_nonuni_single-host_workload.py`)

This script simulates three particles under gravity and a non-uniform electromagnetic field.

**Default execution:**

```bash
python3 three_particles_em_nonuni_single-host_workload.py
```

**Custom execution (stronger B-field, more steps):**

```bash
python3 three_particles_em_nonuni_single-host_workload.py --Bz 5.0 --n_steps 2000
```

**All Arguments:**

  * `--dt` (float, default: 0.01): Time step size.
  * `--n_steps` (int, default: 1000): Total number of steps to simulate.
  * `--G` (float, default: 1.0): Gravitational constant.
  * `--Bz` (float, default: 1.0): Constant component of the magnetic field (Z-axis).
  * `--Bk` (float, default: 0.0): Gradient of the magnetic field along x (Bz = Bz + Bk\*x).
  * `--Ex` (float, default: 0.0): Electric field strength (X-axis).
  * `--Ey` (float, default: 0.0): Electric field strength (Y-axis).

### 5\. Physics: Variational Monte Carlo (VMC) and Diffusion Monte Carlo (DMC) Simulation using JAX for 1D Quantum Harmonic Oscillator (`vmc_qho_jax.py`)

wait

**Note:** For the main benchmark (`tpus_benchmark_single-host_workload.py`), the `--matrix_depth` (`-md`) value must be divisible by the number of cores being tested (e.g., 1, 4, and 8). The script will automatically skip tests that do not meet this requirement.

-----

## ⚠️ Parameter Warnings

  - Setting command-line parameters is critical and can cause tests to fail
  - Configuration requires careful reading of Google Cloud documentation to avoid errors

<!-- end list -->

1.  **Out of Memory (OOM):**

      * Setting `-mxs` (matrix\_size) or `-md` (matrix\_depth) **too high** can cause your TPU/VM to run out of memory (OOM). The script will attempt to catch this error and skip the test, but it is best to start with lower values if you are unsure.
      * The script will suggest smaller `-md` values to try if an OOM occurs during the 3D test.

2.  **Matrix Depth (`-md`) Constraints:**

      * The `-md` (matrix\_depth) value **must be divisible by the number of cores being tested** (e.g., 1, 4, and 8 if you are using a TPU v4-8).
      * If the value is not divisible, the script will automatically skip the 3D (PMAP) test for that specific core count.
      * *Example:* If you use an 8-core TPU and set `-md 64`, all tests will run (64/1, 64/4, 64/8). But if you set `-md 100`, the script will only run the 1-core and 4-core tests, skipping the 8-core test.
