
import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    m, n = A.shape
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not

    if m != n:
        raise ValueError("Matrix is not square")
    if n != b.shape[0]:
        raise ValueError("Matrix and vector sizes are incompatible")
    if not use_pivoting and any(A.diagonal() == 0):
        raise ValueError("Pivoting is disabled but necessary")
    
    # TODO: Perform gaussian elimination

    for k in range(n):
        if use_pivoting:
            max_index = np.argmax(np.abs(A[k:, k])) + k
            A[[k, max_index]] = A[[max_index, k]]
            b[[k, max_index]] = b[[max_index, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]


    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix is not square")
    if n != b.shape[0]:
        raise ValueError("Matrix and vector sizes are incompatible")

    # TODO: Initialize solution vector with proper size
    x = np.zeros(n)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    if any(A.diagonal() == 0):
        raise ValueError("No/infinite solutions exist")
    
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L : Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape

    if n != m:
        raise ValueError("Matrix is not square")
    if not np.allclose(M, M.T):
        raise ValueError("Matrix is not symmetric and positive (semi-)definite")

    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                temp = M[i, i] - np.sum(L[i, :i] ** 2)
                if temp <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i, i] = np.sqrt(temp)
            else:
                if L[j, j] <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i, j] = (M[i, j] - np.dot(L[i, :j], L[j, :j])) / L[j, j]

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape

    if m != n:
        raise ValueError("Matrix is not square")
    if n != b.shape[0]:
        raise ValueError("Matrix and vector sizes are incompatible")

    # TODO Solve the system by forward- and backsubstitution
    x = np.zeros(m)

    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]

    for i in range(n - 1, -1, -1):
        x[i] = (x[i] - np.dot(L[i + 1:, i], x[i + 1:])) / L[i, i]

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    L = np.zeros((n_rays * n_shots, n_grid * n_grid))
    # TODO: Initialize intensity vector
    g = np.zeros(n_rays * n_shots)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    for i in range(n_shots):
        theta = i * np.pi / n_shots
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)
        g[i * n_rays:(i + 1) * n_rays] = intensities
        for j in range(len(ray_indices)):
            L[i * n_rays + ray_indices[j], isect_indices[j]] += lengths[j]
    # Take a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.

    return [L, g]


def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # TODO: Solve for tomographic image using your Cholesky solver
    regularization = (np.finfo(np.float64).eps) * 10
    regularized_matrix = np.dot(L.T, L) + regularization * np.eye(L.shape[1])
    cholesky_factor = compute_cholesky(regularized_matrix)
    tim = solve_cholesky(cholesky_factor, np.dot(L.T, g))
    # (alternatively use Numpy's Cholesky implementation)

    # TODO: Convert solution of linear system to 2D image
    tim = tim.reshape((n_grid, n_grid))
    
    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
