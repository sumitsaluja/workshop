from time import perf_counter

N = 2000
cpu_runs = 5

times = []
import numpy as np
X = np.random.randn(N, N).astype(np.float64)
for _ in range(cpu_runs):
  t0 = perf_counter()
  u, s, v = np.linalg.svd(X)
  times.append(perf_counter() - t0)

print("CPU Execution time: ", min(times))
print("sum(s) = ", s.sum())
print("NumPy version: ", np.__version__)
