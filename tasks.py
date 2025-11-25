from invoke import task

@task
def lint(c):
    """Apply formatting and static analysis (ruff, mypy)."""
    c.run("ruff check src tests")
    c.run("mypy src tests")

@task(help={'type': "Type of benchmark to run (comprehensive, scalability, rram_effects, hp_inv_vs_numpy)"})
def bench(c, type="comprehensive"):
    """Reproduce analogue-versus-digital benchmarking plots."""
    c.run(f"python benchmark_runner.py --type {type}")