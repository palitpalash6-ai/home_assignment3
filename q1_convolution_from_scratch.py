"""
Question 1: Implement Convolution from Scratch
CS5720 - Home Assignment 3
Student: Palash Palit

Performs 2-D convolution (filtering) using only NumPy loops -- no built-in
convolution function. Stride = 1, padding = 0.
"""

import numpy as np

# a. Store the input and filter as NumPy arrays
input_matrix = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0],
])

kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
])


def convolve2d(image, kernel, stride=1, padding=0):
    """Slide `kernel` over `image` and return the feature map.

    At each valid location, take the patch under the kernel, multiply it
    element by element with the kernel, and sum the products (a dot product).
    """
    # Add a border of zeros if padding > 0
    if padding > 0:
        image = np.pad(image, padding, mode="constant", constant_values=0)

    n = image.shape[0]      # input size N (after padding)
    f = kernel.shape[0]     # filter size F

    # Output size: floor((N - F) / S) + 1  (padding is already applied to N)
    out_size = (n - f) // stride + 1
    output = np.zeros((out_size, out_size), dtype=int)

    # b. Slide the filter over the input matrix
    for i in range(out_size):
        for j in range(out_size):
            row, col = i * stride, j * stride
            patch = image[row:row + f, col:col + f]
            # c. Dot product of the patch and the kernel at this location
            output[i, j] = np.sum(patch * kernel)
    return output


feature_map = convolve2d(input_matrix, kernel, stride=1, padding=0)

print("Input (5x5):")
print(input_matrix)
print("\nFilter (3x3):")
print(kernel)

# d. Print the complete output feature map
print("\nOutput feature map (stride = 1, padding = 0):")
print(feature_map)

# e. Print the output shape
print("\nOutput shape:", feature_map.shape)

# Show the work for the top-left position
patch = input_matrix[0:3, 0:3]
print("\nExample - top-left position (0,0):")
print("patch * kernel =")
print(patch * kernel)
print("sum =", np.sum(patch * kernel))

# f. Effect of changing the stride from 1 to 2
feature_map_s2 = convolve2d(input_matrix, kernel, stride=2, padding=0)
print("\nOutput feature map with stride = 2 (for comparison):")
print(feature_map_s2)
print("Output shape:", feature_map_s2.shape)

print("""
Effect of stride 2:
With stride 1 the filter moves one cell at a time and visits every valid
position, giving a (5 - 3)/1 + 1 = 3 x 3 output. With stride 2 the filter
jumps two cells each step, so it only visits positions (0,0), (0,2), (2,0)
and (2,2). The output shrinks to (5 - 3)/2 + 1 = 2 x 2, and its values are
the four corner values of the stride-1 output. A larger stride reduces the
output size and computation but skips positions, so some detail is lost.
""")
