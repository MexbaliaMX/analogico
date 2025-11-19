# Simulation Ideas for HP-INV Research Concepts

## 1. Low-Precision Analogue Inversion Sandbox
- **Goal:** Illustrate how a 3-bit analogue inverter drifts from the ideal solution.
- **Tooling:** Python + NumPy + noise injection; optional PySpice for circuit-level modeling.
- **Key Steps:**
  - Generate random well-conditioned matrices (κ ≈ 5–10).
  - Quantize to 3-bit slices and simulate A₀⁻¹ with additive Gaussian noise.
  - Plot residual norm log₂‖r(k)‖ over iterations to show limited precision.

## 2. HP-INV Iterative Refinement Demo
- **Goal:** Reproduce 24-bit convergence by combining LP-INV and HP-MVM slices.
- **Tooling:** Jupyter notebook with `jax` or `numpy`; use `jax.vmap` for fast MVM.
- **Key Steps:**
  - Implement bit-sliced multiplication `A = Σ 2^{im} A_i` with m=3.
  - Run refinement loop `x_{k+1} = x_k + Δx_k`, `Δx_k = A₀⁻¹ r_k`.
  - Track relative error vs. iteration for different condition numbers.

## 3. BlockAMC Scaling Visualization
- **Goal:** Show how block partitioning enables large matrix inversion with small arrays.
- **Tooling:** Python + NetworkX for block dependency graphs.
- **Key Steps:**
  - Create 16×16 matrix, split into 4×4 blocks.
  - Simulate two-stage BlockAMC flow with intermediate result caching.
  - Display Gantt chart of analogue operations to highlight parallelism.

## 4. Massive MIMO Zero-Forcing Pipeline
- **Goal:** Compare BER performance of HP-INV vs FP32 digital detection.
- **Tooling:** `sionna` or custom Python channel simulator.
- **Key Steps:**
  - Generate Rayleigh fading channels for 16×4 and 128×8 systems.
  - Encode 256-QAM with Gray mapping; add AWGN at varying SNR.
  - Run detection using analogue-refined inverse and plot BER vs SNR.

## 5. Analogue vs Digital Throughput Model
- **Goal:** Highlight projected ×10 throughput advantage.
- **Tooling:** Python script using measured latency/energy parameters from paper.
- **Key Steps:**
  - Parameterize response times (INV=120 ns, MVM=60 ns) and iteration counts.
  - Compute equivalent GFLOPS and energy per operation across matrix sizes.
  - Compare against GPU/ASIC data via bar charts.

## 6. Device Variability Stress Test
- **Goal:** Explore resilience to RRAM conductance drift and stuck-at faults.
- **Tooling:** Monte Carlo simulation in Python; optionally integrate with `PySpice` and `LTspice` for hardware correlation.
- **Key Steps:**
  - Sample conductance deviations (±5%) and line resistances (≈1.7 Ω).
  - Run HP-INV iterations and measure convergence rate distribution.
  - Inject stuck-at faults, test redundancy schemes (row/column swapping).

## 7. Hardware-in-the-Loop Emulator
- **Goal:** Provide developers with a mock interface mimicking 4-bit DAC/ADC quantization effects.
- **Tooling:** Python or Rust service exposing gRPC/REST endpoints; integrate with SDR scripts.
- **Key Steps:**
  - Quantize inputs/outputs to 4 bits and model conversion latency.
  - Allow toggling between ideal and noisy analogue responses.
  - Log waveforms for training operators on tuning thresholds.
