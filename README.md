-----

# JAX TPU Benchmark & Physics Simulation 🚀

This project provides a comprehensive benchmark script (`tpus_benchmark_v3.py`) designed to measure and analyze the performance of JAX on Google Cloud TPUs. It has been expanded to test a wider variety of computations, including 2D/3D matrix operations, 2D/3D FFT, and memory bandwidth, offering insights into performance scaling from single-core (JIT) to multi-core (PMAP) operations.

This project also includes several physics simulation scripts accelerated with JAX, such as an N-Body black hole merger simulation (`nbody_bh_merger_sim.py`) and a Molecular Dynamics simulation (`molecular_dynamics_jax.py`).

  - I was invited to participate in the TPU research and development for the TRC project.
  - Since there is a lot of competition for resources here, I got v4-8 (Immediately) , v4-32 (Queued Resources). (The chip is included in the project so it's free.)
  - And I tested using 300$ credit with v3-8 , v2-8. (Chips from using 300$ credits)
  - So within this project code I have tested the following: v2-8, v3-8, v4-8 and I think other chips should work as well, but not sure for v6e, v5e chips because I have tested and there are bugs from jax lib and many others.
  - Actually, there are other chips but I can't ask for them because they are full.
  - And most importantly, you may find that the installation completes and checks, but a RuntimeError occurs, which means that the TPU was not found on the hardware for that chip. This is a bug in the library, which is not user-resolved. I've encountered this problem many times, but within this project, what I specified works, and what is not specified means it doesn't work, I think.

## ✨ Key Features

  * **Multi-Mode Benchmarking:** Tests a diverse set of operations:
      * 2D Matrix Operations (`jnp.dot`)
      * 3D Tensor Operations (`jnp.matmul`)
      * 2D & 3D FFT (`jnp.fft.fftn`)
      * Memory Bandwidth (`jnp.copy`, `jnp.sum`)
  * **Physics Simulation Examples:**
      * **N-Body Black Hole Merger**: Simulates N-body dynamics (e.g., 3-body), generates gravitational waveforms, and computes Lyapunov exponents for chaos analysis using JAX ODE integration.
      * **Molecular Dynamics**: A pure JAX implementation of a 2D Lennard-Jones fluid simulation using a JIT-compiled Verlet integrator, complete with equilibration and production runs.
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

## 🛠️ Installation

The `install.sh` script provides the necessary commands to set up the environment for a Google Cloud TPU VM. It installs Python 3.13.9, `jax` with TPU support, `torch_xla`, and the other required Python packages.

```bash
sudo apt update -y
sudo apt upgrade -y

sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget

sudo apt install -y python3-dev

export PYTHON_VERSION="3.13.9"
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

sudo cp $PYTHON_PATH/bin/python3.13 /usr/bin/local
sudo cp $PYTHON_PATH/bin/pip3 /usr/bin/local
sudo cp $PYTHON_PATH/bin/pip3.13 /usr/bin/local

export VENV_NAME=".venv"
echo "Creating and activating virtual environment: $VENV_NAME"

"$PYTHON_PATH/bin/python3" -m venv "$VENV_NAME"

source "$VENV_NAME/bin/activate"

python3 --version

echo "Installing project dependencies (PyTorch/XLA, JAX/TPU, etc.)..."

pip install --upgrade pip


pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

pip install "transformers<5.8"


pip install jax>=0.4.0 flax orbax-checkpoint clu tensorflow-datasets tensorflow-metadata protobuf

pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

pip install psutil rich matplotlib pandas jax-md

pip check
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
  * **Note:** `nbody_bh_merger_sim.py` also requires `scipy`. You may need to install it separately: `pip install scipy`.

-----

## 🚀 How to Run

After installing the dependencies and activating the virtual environment, you can run the main benchmark script or the physics simulations.

### 1\. Main Benchmark (`tpus_benchmark_v3.py`)

**Default execution:**
This will run the benchmark with default settings (10 warmup, 1000 steps, 16384 matrix size, 128 matrix depth).

```bash
python3 tpus_benchmark_v3.py
```

**Custom execution:**
This example runs a lighter workload with 5 warmup steps, 500 test steps, an 8192x8192 matrix size, and a depth of 64.

```bash
python3 tpus_benchmark_v3.py -w 5 -m 500 -mxs 8192 -md 64
```

**Export to CSV:**
This example runs a test up to 8 cores and saves the results to `results.csv`.

```bash
python3 tpus_benchmark_v3.py --max_cores 8 --csv results.csv
```

### 2\. Physics Simulations

**N-Body Black Hole Merger (Interactive):**
This script will prompt you for parameters (number of bodies, mass, etc.) in the console.

```bash
python3 nbody_bh_merger_sim.py
```

**Molecular Dynamics:**
You can run this with default parameters or specify your own.

```bash
# Run with defaults (N=400, 10k eq_steps, 10k prod_steps)
python3 molecular_dynamics_jax.py

# Run a longer production simulation
python3 molecular_dynamics_jax.py --prod_steps 50000 --eq_steps 20000
```

**Note:** The `--matrix_depth` (`-md`) value for the main benchmark must be divisible by the number of cores being tested (e.g., 1, 4, and 8 if you have a TPU v4-8). The script will skip tests that do not meet this requirement.

Here is the new section for your `README.md` file, written in English as requested.

-----

## 🔬 Physics Simulation Examples

This project includes several physics simulation scripts to demonstrate the power of JAX (e.g., `jit`, `vmap`, `grad`) in accelerating complex scientific computations.

### 1\. Molecular Dynamics (`molecular_dynamics_jax.py`)

This script implements a 2D Molecular Dynamics (MD) simulation of a Lennard-Jones fluid, written purely in JAX.

**Physics Concepts:**

  * **Lennard-Jones Potential:** Simulates the interaction force between two neutral particles ($U(r)$), featuring a strong short-range repulsion (Pauli exclusion) and a weaker long-range attraction (van der Waals force).
  * **Statistical Mechanics:** The system is initialized with random positions and velocities (based on temperature $kT$). It undergoes an "Equilibration" phase to reach a stable state before the "Production" (data collection) phase begins.
  * **Periodic Boundary Conditions:** Simulates an infinite system by using a finite number of particles in a "box" that wraps around on itself.
  * **Radial Distribution Function $g(r)$:** The final output, $g(r)$, describes the probability of finding another particle at a distance $r$ from a reference particle. Its shape reveals the state of matter (e.g., solid, liquid, gas).

**Mathematical & JAX Implementation:**

  * **Lennard-Jones Potential $U(r)$:** The total potential energy is the sum of all pairwise interactions:
    $$U(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$$
    This is implemented in the `total_energy_fn` function.
  * **Force Calculation (via `jax.grad`):** Instead of manually deriving the force equation ($F = -\nabla U$), the script uses JAX's automatic differentiation:
    ```python
    force_fn = jit(grad(lambda R: -total_energy_fn(R)))
    ```
    This is a key advantage of JAX for physics simulations.
  * **Integration (Velocity Verlet):** The `verlet_step` function uses the Velocity Verlet algorithm to integrate the equations of motion, which is popular in MD for its good energy conservation properties:
    1.  $V(t + \frac{1}{2}\Delta t) = V(t) + \frac{1}{2} A(t) \Delta t$
    2.  $R(t + \Delta t) = R(t) + V(t + \frac{1}{2}\Delta t) \Delta t$
    3.  $A(t + \Delta t) = F(R(t + \Delta t)) / m$
    4.  $V(t + \Delta t) = V(t + \frac{1}{2}\Delta t) + \frac{1}{2} A(t + \Delta t) \Delta t$
  * **Performance:** `jax.jit` is applied to `verlet_step`, `equilibrate_fn`, `production_fn`, and `calculate_g_r`. `jax.lax.fori_loop` is used to compile the entire simulation loop onto the TPU for maximum speed.

-----

### 2\. N-Body Black Hole Merger (`nbody_bh_merger_sim.py`)

This script simulates the dynamics of N bodies (e.g., 3 black holes) under their mutual gravity and computes the resulting gravitational waves (GW).

**Physics Concepts:**

  * **Newtonian Gravity (N-Body):** Solves the classic N-body problem by calculating the gravitational force between all pairs of objects.
  * **Gravitational Waves (Approximation):** The script computes the GW strain ($h_+$) using the Quadrupole approximation, summing the contributions from each orbiting pair.
  * **Chaos Theory:** For N \> 2, the system is often chaotic. The script can compute the **Lyapunov Exponent** ($\lambda$), which measures the exponential rate at which nearby trajectories diverge, quantifying the system's chaos.

**Mathematical & JAX Implementation:**

  * **ODE System:** The N-body problem is a system of Ordinary Differential Equations (ODEs). The state $Y = [\vec{r}_1, ..., \vec{r}_N, \vec{v}_1, ..., \vec{v}_N]$ evolves according to $dY/dt = [\vec{v}_1, ..., \vec{v}_N, \vec{a}_1, ..., \vec{a}_N]$.
  * **Gravitational Acceleration $a_i$:**
    $$\vec{a}_i = \sum_{j \neq i} \frac{G m_j (\vec{r}_j - \vec{r}_i)}{|\vec{r}_j - \vec{r}_i|^3}$$
    This is implemented in the `pairwise_forces` function.
  * **Numerical Integration (RK4):** The script uses a stable and accurate **4th-order Runge-Kutta (RK4)** integrator in `rk4_step` to solve the ODE system.
  * **GW Strain $h_+$:** Calculated from the orbital frequency $\omega = \sqrt{G(m_i+m_j)/r^3}$ and Chirp Mass $\mathcal{M}$. The amplitude $A \propto (\mathcal{M} \omega)^{2/3}$ and phase $\Phi = \int \omega dt$ give $h_+ \propto A \cos(2\Phi)$.
  * **Lyapunov Exponent $\lambda$:**
    $$\lambda \approx \frac{1}{t} \ln \left( \frac{|\delta(t)|}{|\delta(0)|} \right)$$
    This is computed by running two simulations: one with the initial state $Y_0$ and a "perturbed" one with $Y_0 + \delta_0$.
  * **Performance:** The entire simulation loop `simulate_nbody` is JIT-compiled using `jax.jit` and `jax.lax.scan`, allowing the whole trajectory to be computed efficiently on the TPU.

-----

### 3\. Three Particles in Non-Uniform E/M Field (`three_particles_em_nonuni.py`)

This is a simple simulation of 3 charged particles moving under their mutual gravity and an external, non-uniform electromagnetic field.

**Physics Concepts:**

  * **Superposition of Forces:** The net force on each particle is the vector sum of gravity from other particles and the Lorentz force from the E/M field:
    $$\vec{F}_i = \sum_{j \neq i} \vec{F}_{g,ij} + \vec{F}_{Lorentz,i}$$
  * **Lorentz Force:** The force from the E/M field is $\vec{F}_{Lorentz} = q(\vec{E} + \vec{v} \times \vec{B})$.
  * **Non-Uniform Field:** The magnetic field $B_z$ is position-dependent ($B_z = B_0 + B_k \cdot x$), making the dynamics more complex than a simple circular or helical motion.

**Mathematical & JAX Implementation:**

  * **Total Acceleration $\vec{a} = \vec{F}/m$:** The `acceleration` function computes the total acceleration by summing three components:
    1.  **Gravitational ($a_g$):** $\vec{a}_{g,i} = \sum_{j \neq i} G m_j (\vec{r}_j - \vec{r}_i) / |\vec{r}_j - \vec{r}_i|^3$ (in `pairwise_acc`)
    2.  **Electric ($a_E$):** $\vec{a}_E = (q/m) \vec{E}$ (in `elec_acc`)
    3.  **Magnetic ($a_B$):** $\vec{a}_B = (q/m) (\vec{v} \times \vec{B})$
  * **Cross Product in 2D:** For a 2D simulation (x,y) with $\vec{B} = [0, 0, B_z]$ and $\vec{v} = [v_x, v_y, 0]$, the cross product simplifies:
    $$\vec{v} \times \vec{B} = \det \begin{vmatrix} \hat{i} & \hat{j} & \hat{k} \\ v_x & v_y & 0 \\ 0 & 0 & B_z \end{vmatrix} = \hat{i}(v_y B_z) - \hat{j}(v_x B_z) = [v_y B_z, -v_x B_z, 0]$$
    This gives $\vec{a}_B = (q/m) [v_y B_z, -v_x B_z]$, matching the code in `mag_acc`: `qm * jnp.array([vy * bz, -vx * bz])`.
  * **Integration (Leapfrog/Velocity Verlet):** The `step` function uses the same Velocity Verlet integrator as the MD simulation, which is well-suited for velocity-dependent forces like the magnetic force.
  * **Performance:** `jax.vmap` is used to calculate the acceleration for all particles in parallel, and `jax.jit` compiles the entire `step` function.


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
