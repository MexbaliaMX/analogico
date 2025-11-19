from invoke import task

@task
def lint(c):
    """Apply formatting and static analysis (ruff, mypy)."""
    c.run("ruff check src tests")
    c.run("mypy src tests")

@task
def bench(c):
    """Reproduce analogue-versus-digital benchmarking plots."""
    # Placeholder: implement benchmarking logic here
    c.run("python -c 'print(\"Benchmarking not yet implemented\")'")