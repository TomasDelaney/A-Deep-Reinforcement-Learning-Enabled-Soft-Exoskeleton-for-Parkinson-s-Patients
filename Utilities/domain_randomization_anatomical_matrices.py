import numpy as np


def add_symmetric_noise(matrix: np.ndarray, noise_fraction: float = 0.10):
    """
    Adds random symmetric noise to the matrices in the joint angle calculation.

    Parameters:
    - matrix (np.ndarray): A symmetric matrix to which noise will be added.
    - noise_fraction (float): The fraction of noise to add around the original values.

    Returns:
    - np.ndarray: The matrix with added symmetric noise.
    """
    if not np.allclose(matrix, matrix.T):
        raise ValueError("The input matrix must be symmetric.")

    # Create noise matrix
    noise = np.random.uniform(-noise_fraction, noise_fraction, size=matrix.shape)

    # Make noise matrix symmetric
    noise = (noise + noise.T) / 2

    # Add noise to the original matrix
    noisy_matrix = matrix + noise * matrix

    return noisy_matrix
