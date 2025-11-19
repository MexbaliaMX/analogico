## 🧩 Context and Problem

Solving matrix equations of the type **Ax = b** is fundamental in many domains:

* signal processing,
* scientific simulations,
* and neural network training (e.g., second-order optimization).

Traditional digital computing faces scalability limits (O(N³)) and energy consumption, in addition to the "**von Neumann bottleneck**", which separates memory and processing.
Therefore, researchers are exploring **analog computing with resistive random-access memories (RRAM)**, where memory cells act directly as elements of a physical matrix.

---

## ⚙️ Main Innovation: HP-INV and BlockAMC

The study introduces a **high-precision analog solver (HP-INV)** that combines:

1. **Low-precision analog matrix inversion (LP-INV)**
2. **High-precision matrix-vector multiplications (HP-MVM)**

Both operations are implemented in hardware with **3-bit RRAM chips**, fabricated in **40 nm CMOS technology**, using 1T1R cells (one transistor, one resistor).

The method is based on **fully analog iterative refinement**, meaning each iteration refines the result's precision without the need for intermediate digital calculations.
Furthermore, the **BlockAMC** algorithm is integrated, which allows **dividing large matrices into blocks** and solving them in parallel, ensuring **scalability up to 16×16 matrices** with **24-bit fixed precision**, equivalent to **digital FP32**.

---

## 🧠 Key Results

### 1. Precision and Convergence

* Each iteration improves precision by about **3 bits**, reaching 24 bits in 9-10 iterations.
* It was validated on both **real** and **complex** matrices, using techniques such as **bias-column shifting** and **BlockAMC partitioning**.

### 2. Application in Massive MIMO (6G)

* The method was applied to **zero-forcing (ZF)** detection in **16×4 and 128×8 MIMO** systems.
* With only **2-3 HP-INV cycles**, the performance of digital FP32 processors for **256-QAM modulation** was matched, with no observable bit errors.
* The transmitted image (Peking University emblem) was reconstructed with full fidelity in the second iteration.

### 3. Performance and Efficiency

* **INV circuit response time:** ~120 ns
* **Analog MVM:** ~60 ns
* **Throughput:** up to **1000× faster** than equivalent GPUs or ASICs.
* **Energy efficiency:** **100× better** than digital processors at the same precision level (FP32).
* Demonstrated scalability up to **128×128 matrices**, robust against wiring resistance.

---

## 🧪 RRAM Technology Used

* **Material:** TaOx
* **Estructura:** 1T1R (transistor + resistor)
* **Conductance levels:** 8 states (3 bits)
* **Programming method:** *Write-verify ASAP* (Adaptive Step Adjustment Programming), which combines coarse and fine adjustment to ensure uniformity and speed.
* **Compatibility:** fully integrated with **standard CMOS processes**, with no exotic materials.

---

## 🧮 Conclusion

The work demonstrates for the first time an **analog matrix equation solver** with:

* Precision equivalent to FP32,
* Proven scalability through **BlockAMC**,
* Manufacturing compatible with industrial processes (40 nm CMOS),
* And **theoretical performance up to 1000× higher** in throughput and 100× in energy efficiency compared to traditional digital architectures.

---

## 📘 Reference

**Zuo, P., Wang, Q., Luo, Y., et al. (2025).**
*Precise and scalable analogue matrix equation solving using resistive random-access memory chips.*
**Nature Electronics.** DOI: [10.1038/s41928-025-01477-0](https://doi.org/10.1038/s41928-025-01477-0) .