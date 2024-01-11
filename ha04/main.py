import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    n = x.size
    polynomial = np.poly1d(0)
    base_functions = []

    # TODO: Generate Lagrange base polynomials and interpolation polynomial

    for i in range(n):
        base = np.poly1d([1])
        for j in range(n):
            if j != i:
                base *= np.poly1d([1, -x[j]]) / (x[i] - x[j])
        base_functions.append(base)
        polynomial += y[i] * base

    return polynomial, base_functions




def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    # TODO compute piecewise interpolating cubic polynomials

    A = np.zeros((4, 4))

    for i in range(x.size - 1):
        A[0, :] = [x[i] ** 3, x[i] ** 2, x[i], 1]
        A[1, :] = [x[i + 1] ** 3, x[i + 1] ** 2, x[i + 1], 1]
        A[2, :] = [3 * x[i] ** 2, 2 * x[i], 1, 0]
        A[3, :] = [3 * x[i + 1] ** 2, 2 * x[i + 1], 1, 0]

        b = np.array([y[i], y[i + 1], yp[i], yp[i + 1]])
        c = np.linalg.solve(A, b)
        spline.append(np.poly1d(c))

    return spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """
    
    assert (x.size == y.size)
    # TODO construct linear system with natural boundary conditions
    n = x.size
    system_matrix = np.zeros((4*n-4, 4*n-4))
    right_hand_side = np.zeros(4*n-4)

    for i in range(n-1):
        system_matrix[i, 4*i:4*i+4] = [x[i] ** 3, x[i] ** 2, x[i], 1]
        right_hand_side[i] = y[i]
        system_matrix[i+n-1, 4*i:4*i+4] = [x[i+1] ** 3, x[i+1] ** 2, x[i+1], 1]
        right_hand_side[i+n-1] = y[i+1]
    
    for i in range(n-2):
        system_matrix[2*n-2+i, 4*i:4*i+8] = [3 * x[i+1] ** 2, 2 * x[i+1], 1, 0, -3 * x[i+1] ** 2, -2 * x[i+1], -1, 0]
        system_matrix[3*n-4+i, 4*i:4*i+8] = [6 * x[i+1], 2, 0, 0, -6 * x[i+1], -2, 0, 0]
    
    system_matrix[4*n-6, 0:4] = [6 * x[0], 2, 0, 0]
    system_matrix[4*n-5, -4:] = [6 * x[-1], 2, 0, 0]
    right_hand_side[4*n-6] = 0
    right_hand_side[4*n-5] = 0

    # TODO solve linear system for the coefficients of the spline

    solution = np.linalg.solve(system_matrix, right_hand_side)
    print("solution=", solution)

    spline = []
    # TODO extract local interpolation coefficients from solution

    for i in range(n-1):
        spline.append(np.poly1d(solution[4*i:4*i+4]))
    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO: construct linear system with periodic boundary conditions
    n = x.size
    system_matrix = np.zeros((4*n-4, 4*n-4))
    right_hand_side = np.zeros(4*n-4)

    for i in range(n-1):
        system_matrix[i, 4*i:4*i+4] = [x[i] ** 3, x[i] ** 2, x[i], 1]
        right_hand_side[i] = y[i]
        system_matrix[i+n-1, 4*i:4*i+4] = [x[i+1] ** 3, x[i+1] ** 2, x[i+1], 1]
        right_hand_side[i+n-1] = y[i+1]
    
    for i in range(n-2):
        system_matrix[2*n-2+i, 4*i:4*i+8] = [3 * x[i+1] ** 2, 2 * x[i+1], 1, 0, -3 * x[i+1] ** 2, -2 * x[i+1], -1, 0]
        system_matrix[3*n-4+i, 4*i:4*i+8] = [6 * x[i+1], 2, 0, 0, -6 * x[i+1], -2, 0, 0]
    
    system_matrix[4*n-6, 0:4] = [3*x[0]**2, 2*x[0], 1, 0]
    system_matrix[4*n-5, 0:4] = [6*x[0], 2, 0, 0]
    system_matrix[4*n-6, -4:] = [-3*x[-1]**2, -2*x[-1], -1, 0]
    system_matrix[4*n-5, -4:] = [-6*x[-1], -2, 0, 0]
    right_hand_side[4*n-6] = 0
    right_hand_side[4*n-5] = 0


    # TODO solve linear system for the coefficients of the spline

    solution = np.linalg.solve(system_matrix, right_hand_side)
    print("solution=", solution)

    spline = []

    # TODO extract local interpolation coefficients from solution

    for i in range(n-1):
        spline.append(np.poly1d(solution[4*i:4*i+4]))

    return spline


if __name__ == '__main__':

    x = np.array( [1.0, 2.0, 3.0, 4.0])
    y = np.array( [3.0, 2.0, 4.0, 1.0])

    splines = natural_cubic_interpolation( x, y)

    # # x-values to be interpolated
    # keytimes = np.linspace(0, 200, 11)
    # # y-values to be interpolated
    # keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
    #              np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5
    # keyframes.append(keyframes[0])
    # splines = []
    # for i in range(11):  # Iterate over all animated parts
    #     x = keytimes
    #     y = np.array([keyframes[k][i] for k in range(11)])
    #     spline = natural_cubic_interpolation(x, y)
    #     if len(spline) == 0:
    #         animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
    #         self.fail("Natural cubic interpolation not implemented.")
    #     splines.append(spline)

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
