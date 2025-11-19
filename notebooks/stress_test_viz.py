import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from src.stress_test import run_stress_test

# Run stress test
print("Running stress test...")
iters, errors = run_stress_test(num_samples=50, n=8)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.hist(iters, bins=range(min(iters), max(iters)+2), alpha=0.7, edgecolor='black')
ax1.set_title('Distribution of Convergence Iterations')
ax1.set_xlabel('Iterations to Converge')
ax1.set_ylabel('Frequency')

ax2.hist(np.log10(errors), bins=10, alpha=0.7, edgecolor='black')
ax2.set_title('Distribution of Log Relative Errors')
ax2.set_xlabel('Log10(Relative Error)')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('notebooks/stress_test_results.png', dpi=150)
# plt.show()  # Commented out for headless execution

print(f"Average iterations: {np.mean(iters):.2f}")
print(f"Average relative error: {np.mean(errors):.2e}")
print("Plot saved as stress_test_results.png")