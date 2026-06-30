from numba import njit, prange
import numpy as np

import time


@njit(cache=False, parallel=True)
def test_prange(values):
    output = np.empty_like(values)

    for i in prange(values.shape[0]):
        value = values[i]

        # Enough work that parallel execution becomes visible.
        result = 0.0
        for j in range(int(1e8)):
            result += np.sin(value + j * 1e-6)

        output[i] = result

    return output



def main():
    values = np.arange(100, dtype=np.float64)

    t_start = time.perf_counter()

    test_prange(values)
    # test_prange.parallel_diagnostics(level=4)
    
    t_end = time.perf_counter() - t_start
    print(f"Execution time: {t_end:.4f} seconds")


if __name__ == "__main__":
    main()