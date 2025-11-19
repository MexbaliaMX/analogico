# HackGDL2026 RF Village Presentation

## Session Overview
- **Title:** Precise Analogue Matrix Solving with Foundry-Grade RRAM
- **Format:** 30-minute talk + 10-minute live demo + 10-minute Q&A
- **Audience:** RF engineers, SDR hackers, and wireless researchers exploring post-digital accelerators.

## Why RF Hackers Should Care
- Massive MIMO base stations and dense SDR deployments are bounded by Gram-matrix inversion throughput.
- Energy efficiency at the network edge dictates autonomy for field kits, drones, and pop-up cells.
- Analogue resistive arrays collapse data movement by co-locating storage and compute, shrinking latency below the OFDM symbol budget.

## Core Contributions from the Paper
1. **HP-INV Algorithm:** Iterative refinement that pairs 3-bit analogue inversion with high-precision analogue MVM, reaching 24-bit fixed-point accuracy in ≤10 cycles.
2. **BlockAMC Scaling:** Hierarchical block partitioning enables 16×16 inversions using only 8×8 RRAM arrays without reprogramming overhead.
3. **RRAM Hardware:** 40-nm CMOS-compatible 1T1R cells with eight verified conductance states (0.5–35 μS) deliver 100% 3-bit programming yield across 400 devices.
4. **MIMO Benchmark:** 16×4 and 128×8 uplink scenarios with 256-QAM achieve FP32-equivalent BER within three analogue iterations.
5. **Performance Outlook:** Projected ×10 throughput and ×3–5 energy savings vs. current GPU/ASIC baselines for N=128 matrices.

## Talk Flow
1. **Problem Setup (5 min):** Contrast digital O(N³) inversion costs with analogue O(1) time complexity; frame Moore’s law slowdown and von Neumann bottlenecks.
2. **RRAM Primer (5 min):** Explain 1T1R stack, conductance levels, and write–verify ASAP algorithm; show TEM imagery and CDF of states.
3. **Algorithm Deep Dive (10 min):** Walk through residual update equation, bit-slicing process, and convergence guarantees (require ρ(I−AA₀⁻¹)<1).
4. **RF Use-Case (10 min):** Present massive MIMO zero-forcing pipeline; display recovered constellation plots for cycle 1–3; highlight hardware-in-the-loop fixture.
5. **Demo (10 min):** Run Python notebook that feeds recorded channel matrices into HP-INV emulator, outputting BER vs. SNR curves live.
6. **Q&A (10 min):** Field questions on noise tolerance, temperature drift, and integration with SDR front-ends.

## Live Demo Checklist
- Pre-load `assets/mimo_channels.h5` and `scripts/run_hp_inv.py` on presenter laptop.
- Connect logic analyzer to showcase 60 ns MVM and 120 ns INV transient captures.
- Keep fallback GIFs of oscilloscope traces in case hardware jitters.

## Talking Points & Analogies
- “RRAM arrays act as physical matrices; currents encode multiplications while op-amps close the loop for inversion.”
- “Think of HP-INV as an analogue preconditioner constantly cleaned up by high-precision bit slices.”
- “Energy per operation resembles turning a potentiometer once instead of clocking a GPU billions of times.”

## Anticipated Questions & Responses
- **Q:** How robust is the system to wire resistance and parasitics?
  **A:** Simulations with 1.73 Ω line resistance show negligible convergence drift up to 128×128 matrices.
- **Q:** What happens under device drift or stuck-at faults?
  **A:** Integrate confirm cycles and redundant array substitution (per Xia et al.) before mission-critical deployments.
- **Q:** Can we co-package with RF front ends?
  **A:** Analogue compute tiles sit alongside SDR FPGAs; DAC/ADC interface is 4-bit, easing mixed-signal integration.

## Call to Action for RF Village
- Share channel traces for community benchmarking; target exotic fading profiles (vehicular, mmWave).
- Collaborate on open-source HP-INV firmware tuned for portable base stations.
- Explore hybrid analogue/digital workflows where RRAM handles Gram inverses while FPGAs manage MAC scheduling.

## Closing Slide Summary
- **Speed:** 60–120 ns analogue kernels crush digital latency.
- **Precision:** 24-bit fixed-point parity with FP32 after ≤3 cycles for RF workloads.
- **Scalability:** BlockAMC roadmap to 128×8+ MIMO without prohibitive area or power.
- **Community Fit:** Aligns with RF Village’s quest for resilient, deployable wireless hacks beyond conventional silicon.
