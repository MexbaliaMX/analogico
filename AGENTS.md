# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core analogue computing algorithms and supporting utilities.
- `tests/`: automated test suites mirroring the `src/` hierarchy; use `*_spec.py` files.
- `assets/`: reference datasets, figures, and hardware configuration stubs.
- `docs/`: design notes, experiment logs, and supplementary materials.
- `scripts/`: helper tooling for synthesis, data prep, and benchmarking.
- `notebooks/`: starter analysis notebooks (for example, `low_precision_analog_inversion.ipynb` simulates 3-bit LP-INV noise).
- `context.md`, `executive.md`, `Content.md`, `YouTube.md`, `List.md`: quick-reference summaries, study guides, and simulation ideas derived from the HP-INV research paper.

## Build, Test, and Development Commands
- `poetry install` — install Python dependencies and set up the virtual environment.
- `poetry run pytest` — execute the full automated test suite.
- `poetry run pytest tests/integration` — run only integration scenarios against RRAM mocks.
- `poetry run invoke lint` — apply formatting and static analysis (ruff, mypy).
- `poetry run invoke bench` — reproduce analogue-versus-digital benchmarking plots.

## Coding Style & Naming Conventions
- Python code follows PEP 8 with 4-space indentation; prefer type hints throughout.
- Modules use lowercase_with_underscores; classes use CapWords; constants are UPPER_CASE.
- Keep public APIs documented with docstrings; include units for analogue parameters.
- Run `poetry run invoke lint` before submitting changes to ensure ruff/mypy compliance.

## Testing Guidelines
- Primary framework: pytest with hypothesis for property-based checks.
- Name unit tests `test_<feature>()`; integration suites live under `tests/integration/`.
- Maintain ≥90% coverage on critical solvers; report coverage via `pytest --cov=src`.
- Provide hardware-in-the-loop fixtures under `tests/hardware/`, guarded by `@pytest.mark.hw`.

## Commit & Pull Request Guidelines
- Use imperative commit messages (e.g., `Add BlockAMC convergence guard`).
- Reference tickets with `Refs #<id>` when applicable; squash trivial fixups.
- PRs must include: summary, verification steps, and screenshots/logs for analogue runs.
- Ensure CI passes (lint, unit, integration, coverage) before requesting review.

## Context & Knowledge Sharing
- Review `context.md` and `executive.md` for the latest interpretation of the HP-INV paper and its engineering implications.
- `Content.md`, `YouTube.md`, and `List.md` list vetted educational resources and simulation backlogs; keep these updated as new findings emerge.
- Capture RF Village presentation assets under `HackGDL2026.md` and social outreach drafts (for example, `LinkedIn.md`) to maintain consistent messaging.
