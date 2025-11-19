# GEMINI.md: Project Context for "analogico"

This document provides a comprehensive overview of the "analogico" project for new contributors and AI assistants. Its purpose is to bootstrap the context required to understand and contribute to the project.

## 1. Project Overview

The "analogico" project is focused on **generating a comprehensive understanding of the research paper 'Precise and scalable analogue matrix equation solving using resistive random-access memory chips'** in the context of the **RF Village MX** community.

The primary goal is to **distill the paper's core concepts**, including the **High-Precision Inversion (HP-INV)** algorithm, and make them accessible to a broader audience of RF engineers, hackers, and researchers. This involves creating summaries, documentation, and simulation tools to explore the paper's findings and their practical implications for the RF Village MX community.

The project aims to serve as an educational resource and a starting point for discussions and experimentation related to the paper's innovative analog computing techniques.

## 2. Core Technology

The cornerstone of the project is the **HP-INV algorithm**, an iterative refinement method that operates entirely in the analog domain.

- **LP-INV (Low-Precision Inversion):** A coarse, initial solution is computed using a 3-bit RRAM array that acts as a physical, low-precision matrix inverter.
- **HP-MVM (High-Precision Matrix-Vector Multiplication):** The solution is iteratively refined using highly accurate matrix-vector multiplications, also implemented with RRAM in a bit-sliced manner.
- **BlockAMC:** To handle large matrices, the project explores the **BlockAMC** algorithm, which partitions a large matrix into smaller blocks that can be processed by the hardware.

The underlying technology, as described in the source paper, is a **40nm CMOS process with embedded Tantalum Oxide (TaOx) RRAM cells**.

## 3. Repository Structure

The repository is organized as follows:

- `src/`: Contains the core Python source code for simulating the HP-INV algorithm, RRAM device models, and related functions.
- `tests/`: Houses unit and integration tests for the code in `src/`.
- `notebooks/`: Jupyter notebooks for exploratory analysis, data visualization, and interactive simulations.
- `*.md`: A collection of Markdown files providing detailed documentation, including summaries of the source research paper, a project whitepaper, presentation materials, and contribution guidelines.
- `pyproject.toml`: The project's configuration file, managing dependencies and tool settings for `poetry`, `ruff`, and `mypy`.

## 4. Key Files

To quickly get up to speed, review the following files:

- **`Nature_RRAM.md`**: A detailed summary of the foundational research paper that introduces the HP-INV method.
- **`Whitepaper.md`**: A document that contextualizes the HP-INV methodology for a practical application at the "HackGDL 2026 RF Village".
- **`context.md` & `executive.md`**: Concise summaries of the research context and an executive overview of the project's goals and findings.
- **`src/hp_inv.py`**: The core Python implementation of the HP-INV algorithm simulation.
- **`src/rram_model.py`**: The simulation model for the RRAM device, including variability and fault modeling.
- **`pyproject.toml`**: Defines all project dependencies and development tool configurations.

## 5. Getting Started

To set up the development environment and run the simulations, you will need Python and Poetry.

1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    poetry install
    ```
3.  **Run the test suite:**
    ```bash
    poetry run pytest
    ```
4.  **Run a stress test simulation:**
    ```bash
    python src/stress_test.py
    ```
5.  **Explore the notebooks** in the `notebooks/` directory for interactive examples.

## 6. How to Contribute

This project adheres to standard open-source contribution practices.

-   **Coding Style:** Follow PEP 8 guidelines. The project uses `ruff` for formatting and `mypy` for static type checking. Run `poetry run invoke lint` before committing.
-   **Testing:** New features must be accompanied by tests. The project aims for high test coverage.
-   **Pull Requests:** Submit changes via pull requests with clear, descriptive titles and summaries.
-   **Guidelines:** For more detailed instructions, refer to **`AGENTS.md`**.
