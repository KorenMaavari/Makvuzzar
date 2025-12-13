import timeit

import numpy as np
from numba import cuda, njit, prange


def matmul_transpose_trivial(X):
    n, m = X.shape[0], X.shape[1]
    ret = np.empty(shape=(n, n), dtype=X.dtype)
    for x in range(n):
        for y in range(n):
            tmp = 0
            for k in range(m):
                tmp += X[x, k] * X[y, k]
            ret[x, y] = tmp

    return ret


@njit(parallel=True)
def matmul_transpose_numba(X):
    n, m = X.shape[0], X.shape[1]
    ret = np.empty(shape=(n, n), dtype=X.dtype)
    for x in prange(n):
        for y in range(n):
            tmp = 0
            for k in range(m):
                tmp += X[x, k] * X[y, k]
            ret[x, y] = tmp

    return ret


def matmul_transpose_gpu(X):
    n = X.shape[0]
    d_X = cuda.to_device(X)
    d_ret = cuda.device_array(shape=(n, n), dtype=X.dtype)
    matmul_kernel[1, 1024](d_X, d_ret)
    return d_ret

@cuda.jit
def matmul_kernel(A, C):
    n, m = A.shape[0], A.shape[1]
    idx = cuda.threadIdx.x
    while idx < n * n:
        x, y = idx // n, idx % n
        tmp = 0.0
        for k in range(m):
            tmp += A[x, k] * A[y, k]
        C[x, y] = tmp
        idx += 1024

def verify_solution():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    if not np.allclose(matmul_transpose_trivial(X), np.matmul(X, Xt)):
        print("[-] matmul_transpose_trivial failed")
        exit(0)
    else:
        print("[+] matmul_transpose_trivial passed")

    if not np.allclose(matmul_transpose_numba(X), np.matmul(X, Xt)):
        print("[-] matmul_transpose_numba failed")
        exit(0)
    else:
        print("[+] matmul_transpose_numba passed")

    if not np.allclose(matmul_transpose_gpu(X), np.matmul(X, Xt)):
        print("[-] matmul_transpose_gpu failed")
        exit(0)
    else:
        print("[+] matmul_transpose_gpu passed")

    print("[+] All tests passed\n")


# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(
            timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(
                3, 100
            )
        )

    # print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    numpy_time = timer(np.matmul, 2)
    numba_time = timer(matmul_transpose_numba, 1)
    gpu_time = timer(matmul_transpose_gpu, 1)

    print("Numpy:", numpy_time)
    print("Numba:", numba_time)
    print("CUDA:", gpu_time)

    print(f"[*] Speedup GPU/Numba: {numba_time / gpu_time:.2f}x")
    print(f"[*] Speedup GPU/Numpy: {numpy_time / gpu_time:.2f}x")


if __name__ == "__main__":
    verify_solution()
    matmul_comparison()
