# Next Steps

1. Wire up CI (e.g., GitHub Actions) that installs via Poetry and runs `pytest`, `pytest --cov=src`, and `invoke lint` to guarantee the verification workflow executes on every push/PR.
2. Integrate redundancy modeling (from `src/redundancy.py`) into the RRAM generator and stress tests, adding deterministic seeding and configurable knobs for variability, stuck faults, and line resistance.
3. Harden `hp_inv` by clarifying numerical assumptions, handling singular/ill-conditioned LP inverses gracefully, and exposing convergence diagnostics (residual norms, failure flags).
4. Replace the placeholder `invoke bench` task with a real benchmarking pipeline that runs stress tests, aggregates statistics, and emits plots/CSV artifacts into `assets/`.
5. Strengthen automated tests: add Hypothesis-based property checks, integration suites that exercise the full pipeline, and enforce coverage reporting with `pytest --cov=src`.
6. Keep contextual docs (`context.md`, `executive.md`, learning backlogs) in sync with new modeling choices and experimental findings to support knowledge sharing.
