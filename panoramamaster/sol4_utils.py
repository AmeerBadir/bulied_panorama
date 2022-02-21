from scipy.signal import convolve2d
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy import ndimage


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img

def read_image(filename, representation):
    """
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
                           image (1) or an RGB image (2).
    :return:    function which reads an image file and converts it into a given representation  and returns it.
    """
    im = imread(filename)  # imread return a numpy array

    # convert rgb to grayscale
    if representation == 1 and len(im.shape) == 3:
        return convert_rgb_to_grayscale(im)
    # normalized grayscale
    if representation == 1:
        im = im / 255
        return im.astype(np.float64)
    else:
        # convert to rgb
        return (im / 255).astype(np.float64)



def convert_rgb_to_grayscale(image):
    """
    :param image: image
    :return: a grayscale image
    """
    im = rgb2gray(image).astype(np.float64)
    return im



def get_filtered_vec(filtered_size, binom_gaus):
    """
    :param filtered_size: the size of the Gaussian filter
    :return: the resulting pyramid pyr and filter_vec which is row vector of shape (1, filter_size) used for the pyramid construction
    and the factor to normalization
    """
    norm = 2 ** (filtered_size - 1)
    if filtered_size - 1 <= 0:
        return np.array([1]), 1
    first_cov = np.array([1, 1])
    i = 2
    while i != filtered_size:
        first_cov = np.convolve(first_cov, binom_gaus)
        i += 1
    return first_cov, norm


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im:  a grayscale image with double values in [0, 1]
    :param max_levels:  the maximal number of levels1 in the resulting pyramid.
    :param filter_size:  the size of the Gaussian filter
    :return:  a gaussian pyramid

    """
    binom_gaus = np.array([1, 1])
    filtered_vec, nor_factor = get_filtered_vec(filter_size, binom_gaus)
    if nor_factor != 1:
        filtered_vec = np.array([filtered_vec]) * (1 / nor_factor)
    else:
        filtered_vec = np.array([1])
    pyr = list()
    if max_levels == 1:
        return [im], filtered_vec
    pyr.append(im)
    i = 1
    new_img = np.copy(im)
    transpose_filter = filtered_vec.T
    while i < max_levels and (new_img.shape[0] / 2 >= 16) and new_img.shape[1] / 2 >= 16:
        reduce = ndimage.convolve(ndimage.convolve(new_img, filtered_vec), transpose_filter)
        reduce = reduce[::2, ::2]
        pyr.append(reduce)
        new_img = reduce.copy()
        i = i + 1
    return pyr, filtered_vec


