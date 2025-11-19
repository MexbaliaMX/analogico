# HP-INV Research Context

## Core Problem Space
- Linear system solving (Ax = b) underpins scientific computing, RF signal processing, and second-order ML training.\n- Digital inversion carries O(N³) complexity and is bottlenecked by memory bandwidth (von Neumann separation).\n- Analogue matrix computing (AMC) uses resistive RAM (RRAM) crossbars as physical matrices, performing MVM in O(1) time by exploiting Kirchoff’s laws.

## Precision Bottleneck & Motivation
- Classic analogue inversion circuits suffer from low precision (≈2–3 bits) due to device variability, noise, and coarse conductance levels.\n- Iterative analogue solvers need reliable distributive properties, but inversion lacks a straightforward block law, hindering scaling.\n- High precision is mandatory for inverse problems, massive MIMO detection, and scientific simulations where errors compound.

## High-Precision Inversion (HP-INV)
- Iterative refinement loop entirely in analogue domain: low-precision inversion (LP-INV) + high-precision matrix–vector multiplication (HP-MVM).\n- Matrix A is decomposed into 3-bit slices (A = Σ 2^{im} A_i); the most significant slice (A₀) is mapped to LP-INV, remaining slices drive HP-MVM via bit-slicing.\n- DAC/ADC precision kept to 4 bits because refinement cancels coarse analogue errors.\n- Convergence condition requires spectral radius ρ(I − A A₀⁻¹) < 1; programming errors and matrix conditioning govern rate.

## Hardware Platform
- Various RRAM materials supported including HfO₂, TaOₓ, TiO₂, NiOₓ, and CoOₓ with material-specific characteristics.\n- Material-specific switching mechanisms: valence change mechanism (VCM) for HfO₂/TaOₓ/TiO₂, electrochemical metallization (ECM) for NiOₓ/CoOₓ, and mixed mechanisms.\n- Eight stable conductance levels (0.5–35 μS) achieved with ASAP write–verify algorithm; 100% success over 400 cells.\n- Operational amplifiers (AD823, OPA690) close the inversion loop; transient response varies by material (~120 ns typical for LP-INV).\n- Wire resistance (~1.73 Ω per adjacent devices) causes minimal convergence drift up to 128×128 matrices.\n- Temperature-dependent behavior and material-specific ageing effects modeled for realistic simulation.

## BlockAMC Scalability
- Large matrices partitioned into block submatrices; Stage 1 solves Re(A) via LP-INV and Im(A) via MVM; Stage 2 recursively inverts block diagonals.\n- Enables 16×16 real-valued inversions using 8×8 arrays without reprogramming.\n- Demonstrated 24-bit fixed-point accuracy (≈FP32) after ≤10 iterations.

## Massive MIMO Application
- Tested 16×4 and 128×8 uplink zero-forcing detection with 256-QAM modulation.\n- HP-INV achieves BER on par with FP32 digital processors within ≤3 iterations; vanilla analogue inversion fails to converge to acceptable BER.\n- Iterations reconstruct full constellation without symbol errors, validating precision for RF payloads.

## Throughput & Energy Benchmarking
- Analog operations have constant latency, yielding ~×10 throughput vs GPU/ASIC at N = 128; energy efficiency ×3–5 better.\n- Projected improvements (faster op-amps) could unlock ×1000 throughput, ×100 energy gains over digital for same precision.

## Reliability & Future Directions
- Device drift addressed with confirm cycles, redundant arrays (row/column substitution).\n- ADC area dominates chip footprint; integrating LP-INV and HP-MVM on a single die is a key next step.\n- Research opportunities: larger array tapes-outs, temperature drift compensation, hybrid SDR integration, open-source firmware for HP-INV pipelines.
