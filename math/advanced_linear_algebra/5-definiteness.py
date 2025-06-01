import numpy as np

def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.allclose(matrix, matrix.T):
        return None

    eigvals = np.linalg.eigvals(matrix)

    if np.all(eigvals > 0):
        return "Positive definite"
    if np.all(eigvals >= 0):
        return "Positive semi-definite"
    if np.all(eigvals < 0):
        return "Negative definite"
    if np.all(eigvals <= 0):
        return "Negative semi-definite"
    if np.any(eigvals > 0) and np.any(eigvals < 0):
        return "Indefinite"

    return None
