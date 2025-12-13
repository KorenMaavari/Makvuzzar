import timeit

import numpy as np
from numba import cuda, njit, prange


def max_cpu(A, B):
    """
    Returns
    -------
    np.array
        element-wise maximum between A and B
    """
    C = np.zeros_like(A)
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            C[x, y] = max(A[x, y], B[x, y])
    return C


@njit(parallel=True)
def max_numba(A, B):
    """
    Returns
    -------
    np.array
        element-wise maximum between A and B
    """
    C = np.zeros_like(A)
    for x in prange(A.shape[0]):
        for y in prange(A.shape[1]):
            C[x, y] = max(A[x, y], B[x, y])
    return C


def max_gpu(A, B):
    """
    Returns
    -------
    np.array
        element-wise maximum between A and B
    """
    d_A, d_B = cuda.to_device(A), cuda.to_device(B)
    d_C = cuda.to_device(np.zeros_like(A))

    max_kernel[1000, 1000](d_A, d_B, d_C)
    return d_C.copy_to_host()

@cuda.jit
def max_kernel(A, B, C):
    x, y = cuda.blockIdx.x, cuda.threadIdx.x
    C[x, y] = max(A[x, y], B[x, y])


def verify_solution():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    if not np.all(max_cpu(A, B) == np.maximum(A, B)):
        print("[-] max_cpu failed")
        exit(0)
    else:
        print("[+] max_cpu passed")
    if not np.all(max_numba(A, B) == np.maximum(A, B)):
        print("[-] max_numba failed")
        exit(0)
    else:
        print("[+] max_numba passed")

    if not np.all(max_gpu(A, B) == np.maximum(A, B)):
        print("[-] max_gpu failed")
        exit(0)
    else:
        print("[+] max_gpu passed")

    print("[+] All tests passed\n")


# this is the comparison function - keep it as it is.
def max_comparison():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    def timer(f):
        return min(timeit.Timer(lambda: f(A, B)).repeat(3, 20))

    cpu_time = timer(max_cpu)
    numba_time = timer(max_numba)
    gpu_time = timer(max_gpu)

    print("[*] CPU:", cpu_time)
    print("[*] Numba:", numba_time)
    print("[*] CUDA:", gpu_time)

    print(f"[*] Speedup GPU/Numba: {numba_time / gpu_time:.2f}x")
    print(f"[*] Speedup GPU/CPU: {cpu_time / gpu_time:.2f}x")


if __name__ == "__main__":
    verify_solution()
    max_comparison()
