# HP-INV Analogue Matrix Computing

## 1. Title Slide
- High-Precision Analogue Matrix Inversion (HP-INV)
- Bridging resistive crossbars with digital-grade accuracy
- Presenters: RRAM Systems Lab · Analogico Initiative

## 2. Agenda
- Motivation: matrix inversion bottlenecks
- HP-INV architecture & workflow
- Hardware platform highlights
- Benchmark results vs digital baselines
- Roadmap and partnership opportunities

## 3. Motivation
- Linear solves (Ax = b) drive RF detection, scientific computing, and ML training.
- Digital inversion scales with O(N^3) complexity and memory traffic limits throughput.
- Conventional analogue in-memory computing offers massive parallelism but stalls at 2–3 bit precision.
- Goal: marry analogue speed with FP32-class accuracy for production-grade workloads.

## 4. HP-INV Concept
- Two-stage analogue refinement loop:
  - LP-INV: 3-bit coarse inversion on 8×8 RRAM arrays.
  - HP-MVM: high-precision matrix–vector corrections via bit-sliced accumulation.
- Iterative refinement cancels device noise using 4-bit DAC/ADC interfaces.
- Convergence guaranteed when ρ(I − A A₀⁻¹) < 1; verified on RF and scientific matrices.

## 5. System Architecture
- Matrix decomposition: A = Σ 2^{im} A_i with A₀ mapped to LP-INV.
- ASAP write–verify algorithm programs stable conductance in TaOₓ 1T1R cells.
- Analogue loop closed with AD823 and OPA690 op-amps; sub-120 ns response.
- BlockAMC extends inversion beyond native array size via block-partitioning.

## 6. Hardware Snapshot
- 40 nm foundry-fabricated RRAM; eight discrete conductance levels (0.5–35 μS).
- LP-INV array (8×8) paired with 1 Mb HP-MVM fabric; redundant rows mitigate drift.
- Confirm cycles and temperature monitoring ensure long-term stability.
- ADC footprint currently dominates area; monolithic integration underway.

## 7. Benchmark Highlights
- Massive MIMO zero-forcing (16×4, 128×8) achieves BER parity with FP32 GPUs in ≤3 iterations.
- Throughput: ~×10 speedup vs latest GPU at N = 128, with ×3–5 energy savings.
- Projected upgrades (faster op-amps, denser arrays) unlock ×1000 throughput, ×100 energy gains.
- Scientific workloads (Poisson solvers) converge to 24-bit fixed-point accuracy.

## 8. Software & Tooling
- Python/Poetry stack with pytest + hypothesis for algorithm validation.
- LTspice and PySpice models replicate HP-INV loop for rapid what-if analysis.
- Invoke-based tooling automates linting, benchmarking, and coverage tracking.
- Notebooks demonstrate hardware-in-the-loop simulation paths.

## 9. Roadmap
- Short term: integrate HP-INV firmware, expand coverage of BlockAMC regression tests.
- Medium term: tape-out larger arrays, co-design temperature compensation circuits.
- Long term: deploy hybrid SDR demonstrator and open-source HP-INV SDK.

## 10. Collaboration Opportunities
- RF partners for massive MIMO pilots and OTA testing.
- Academic collaborations on algorithmic convergence proofs and variability modeling.
- Industry co-design for ADC integration and low-power op-amp stacks.

## 11. Call to Action
- Download the HP-INV toolkit and run provided benchmarks.
- Join the Analogico working group for monthly design reviews.
- Provide candidate workloads to stress-test BlockAMC extensions.
- Reach out: hp-inv@analogico.dev
