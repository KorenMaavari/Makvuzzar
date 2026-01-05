#
#   @date:
#       28/12/25
#   @author:
#       Tal Ben Ami, 212525257
#       Koren Maavari, 207987314
# 
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
import imageio
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, njit, prange


def correlation_gpu(kernel, image):
    """Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels

    Return
    ------
    An numpy array of same shape as image
    """
    # Ensure kernel and image are numpy arrays of dtype float32 for CUDA
    kernel = np.asarray(kernel, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)

    k_h, k_w = kernel.shape
    i_h, i_w = image.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Pad image with zeros
    padded_image = np.zeros((i_h + 2 * pad_h, i_w + 2 * pad_w), dtype=np.float32)
    padded_image[pad_h : pad_h + i_h, pad_w : pad_w + i_w] = image

    result = np.zeros((i_h, i_w), dtype=np.float32)

    # Device allocations
    d_padded_image = cuda.to_device(padded_image)
    d_kernel = cuda.to_device(kernel)
    d_result = cuda.device_array((i_h, i_w), dtype=np.float32)

    # Launch kernel (1 thread per pixel)
    threadsperblock = (16, 16)
    blockspergrid_x = (i_h + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (i_w + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    correlation_kernel[blockspergrid, threadsperblock](
        d_padded_image, d_kernel, d_result, k_h, k_w, pad_h, pad_w
    )
    result = d_result.copy_to_host()
    return result


@cuda.jit
def correlation_kernel(padded_image, kernel, result, k_h, k_w, pad_h, pad_w):
    i, j = cuda.grid(2)

    i_h = result.shape[0]
    i_w = result.shape[1]

    if i < i_h and j < i_w:
        acc = 0.0
        for ki in range(k_h):
            for kj in range(k_w):
                pi = i + ki
                pj = j + kj
                acc += padded_image[pi, pj] * kernel[ki, kj]
        result[i, j] = acc


@njit(parallel=True)
def correlation_numba(kernel, image):
    """Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels

    Return
    ------
    An numpy array of same shape as image
    """
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Pad image with zeros
    padded_image = np.zeros((i_h + 2 * pad_h, i_w + 2 * pad_w))
    padded_image[pad_h : pad_h + i_h, pad_w : pad_w + i_w] = image

    result = np.zeros(image.shape, dtype=np.float64)

    for i in prange(i_h):
        for j in range(i_w):
            region = padded_image[i : i + k_h, j : j + k_w]
            result[i, j] = np.sum(region * kernel)
    return result


def sobel_operator():
    """Load the image and perform the operator
    ----------
    Return
    ------
    An numpy array of the image
    """
    pic = load_image()

    # Define the Sobel filter
    sobel_filter = np.array([[+1, 0, -1], [+2, 0, -2], [+1, 0, -1]], dtype=np.float64)
    #kernel1
    # sobel_filter = np.array([[+3, 0, -3], [+10, 0, -10], [+3, 0, -3]], dtype=np.float64)
    #kernel2
    # sobel_filter = np.array([[+1, 0, -1], [+2, 0, -2], [+1, 0, -1],[+2, 0, -2], [+1, 0, -1]], dtype=np.float64)
    #kernel3
    # sobel_filter = np.array([[+1, +1, +1], [+1, 0, +1], [+1, +1, +1]], dtype=np.float64)

    # Compute Gx = correlation(sobel_filter, image)
    Gx = correlation_numba(sobel_filter, pic)

    # Compute Gy = correlation(sobel_filter.T, image)
    Gy = correlation_numba(np.transpose(sobel_filter), pic)

    # result[i,j] = sqrt(Gx[i,j]² + Gy[i,j]²)
    result = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))

    return result



def load_image():
    fname = "data/image.jpg"
    pic = imageio.imread(fname)
    to_gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap="gray")
    plt.show()
