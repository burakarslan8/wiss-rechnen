import numpy as np


####################################################################################################
# Exercise 1: Function Roots

def find_root_bisection(f: object, lival: np.floating, rival: np.floating, ival_size: np.floating = -1.0, n_iters_max: int = 256) -> np.floating:
    """
    Find a root of function f(x) in (lival, rival) with bisection method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    lival: initial left boundary of interval containing root
    rival: initial right boundary of interval containing root
    ival_size: minimal size of interval / convergence criterion (optional)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root of the function
    """

    assert (n_iters_max > 0)
    assert (rival > lival)

    # TODO: set meaningful minimal interval size if not given as parameter, e.g. 10 * eps
    ival_size = 10 * np.finfo(np.float64).eps

    # intialize iteration
    fl = f(lival)
    fr = f(rival)

    # make sure the given interval contains a root
    assert (not ((fl > 0.0 and fr > 0.0) or (fl < 0.0 and fr < 0.0)))

    n_iterations = 0
    # TODO: loop until final interval is found, stop if max iterations are reached
    for i in range(n_iters_max):
        fmid= f((lival + rival) / 2)
        if fl * fmid < 0:
            rival = (lival + rival) / 2
            fr = fmid
        else:
            lival = (lival + rival) / 2
            fl = fmid

        if abs(rival - lival) < ival_size:
            break


    # TODO: calculate final approximation to root
    root = np.float64(0.0)
    root = (lival + rival) / 2

    print(root)

    return root


def func_f(x):
    return x**3 - 2*x + 2 # -1.76929235423863

def deri_f(x):
    return 3 * x**2 - 2

def func_g(x):
    return 6*x/(x**2 + 1)

def deri_g(x):
    return 6 * (1 - x**2) / (x**2 + 1)**2

def find_root_newton(f: object, df: object, start: np.inexact, n_iters_max: int = 256) -> (np.inexact, int):
    """
    Find a root of f(x)/f(z) starting from start using Newton's method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    df: derivative of function f, also callable
    start: start position, can be either float (for real valued functions) or complex (for complex valued functions)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root, should have the same format as the start value start
    n_iterations: number of iterations
    """

    assert(n_iters_max > 0)

    # Initialize root with start value
    root = start

    # TODO: chose meaningful convergence criterion eps, e.g 10 * eps
    convergence_criterion = 10 * np.finfo(np.float64).eps

    # Initialize iteration
    fc = f(root)
    dfc = df(root)
    n_iterations = 0

    # TODO: loop until convergence criterion eps is met
    for i in range(n_iters_max):
        # TODO: return root and n_iters_max+1 if abs(derivative) is below f_eps or abs(root) is above 1e5 (to avoid divergence)
        n_iterations += 1
        if abs(dfc) < convergence_criterion or abs(root) > 1e5:
            break

        # TODO: update root value and function/dfunction values
        root = root - fc / dfc
        fc = f(root)
        dfc = df(root)

        # TODO: avoid infinite loops and return (root, n_iters_max+1)
        if abs(fc) < convergence_criterion:
            break

    print(root)

    return root, n_iterations

####################################################################################################
# Exercise 2: Newton Fractal


def generate_newton_fractal(f: object, df: object, roots: np.ndarray, sampling: np.ndarray, n_iters_max: int=20) -> np.ndarray:
    """
    Generates a Newton fractal for a given function and sampling data.

    Arguments:
    f: function (handle)
    df: derivative of function (handle)
    roots: array of the roots of the function f
    sampling: sampling of complex plane as 2d array
    n_iters_max: maxium number of iterations the newton method can calculate to find a root

    Return:
    result: 3d array that contains for each sample in sampling the index of the associated root and the number of iterations performed to reach it 
    """

    result = np.zeros((sampling.shape[0], sampling.shape[1], 2), dtype=int)

    # TODO: iterate over sampling grid
    for i in range(sampling.shape[0]):
        for j in range(sampling.shape[1]):
            # TODO: run Newton iteration to find a root and the iterations for the sample (in maximum n_iters_max iterations)
            root, n_iters = find_root_newton(f, df, sampling[i, j], n_iters_max)
            
            # TODO: determine the index of the closest root from the roots array. The functions np.argmin and np.tile could be helpful.
            index = np.argmin(np.abs(roots - root))
            
            # TODO: write the index and the number of needed iterations to the result
            result[i, j] = np.array([index, n_iters+1])

    print("Result:" , result)
    return result


####################################################################################################
# Exercise 3: Minimal Surfaces

def surface_area(v: np.ndarray, f: np.ndarray) -> float:
    """
    Calculate the area of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    area: the total surface area
    """

    # initialize area
    area = 0.0

    # TODO: iterate over all triangles and sum up their area
    for i in range(f.shape[0]):
        a = v[f[i][0]]
        b = v[f[i][1]]
        c = v[f[i][2]]

        area += np.linalg.norm(np.cross(b - a, c - a)) / 2

    return area


def surface_area_gradient(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Calculate the area gradient of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    gradient: the surface area gradient of all vertices in v
    """

    # intialize the gradient
    gradient = np.zeros(v.shape)

    # TODO: iterate over all triangles and sum up the vertices gradients
    for i in range(f.shape[0]):
        a = v[f[i, 0]]
        b = v[f[i, 1]]
        c = v[f[i, 2]]

        ac = a - c
        bc = b - c
        n0 = np.cross(bc, ac) / np.linalg.norm(np.cross(bc, ac))
        gradient[f[i, 0]] -= np.cross(n0, bc)

        ca = c - a
        ba = b - a
        n1 = np.cross(ca, ba) / np.linalg.norm(np.cross(ca, ba))
        gradient[f[i, 1]] -= np.cross(n1, ca)

        ab = a - b
        cb = c - b
        n2 = np.cross(ab, cb) / np.linalg.norm(np.cross(ab, cb))
        gradient[f[i, 2]] -= np.cross(n2, ab)

    return gradient


def gradient_descent_step(v: np.ndarray, f: np.ndarray, c: np.ndarray, epsilon: float=1e-6, ste: float=1.0, fac: float=0.5) -> (bool, float, np.ndarray, np.ndarray):
    """
    Calculate the minimal area surface for the given triangles in v/f and boundary representation in c.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i
    c: list of vertex indices which are fixed and can't be moved
    epsilon: difference tolerance between old area and new area

    Return:
    converged: flag that determines whether the function converged
    area: new surface area after the gradient descent step
    updated_v: vertices with changed positions
    gradient: calculated gradient
    """


    # TODO: calculate gradient and area before changing the surface
    gradient = surface_area_gradient(v, f)
    area = surface_area(v, f)

    # TODO: calculate indices of vertices whose position can be changed
    indices = np.array([i for i in range(v.shape[0]) if i not in c])
    print(c)
    print(f)
    print(indices)
    # TODO: find suitable step size so that area can be decreased, don't change v yet
    step = ste
    new_v = v.copy()
    for i in range(len(indices)):
        new_v[indices[i]] = v[indices[i]] + step * gradient[indices[i]]
    
    new_area = surface_area(new_v, f)

    while new_area > area and step > epsilon:
        step *= fac
        for i in range(len(indices)):
            new_v[indices[i]] = v[indices[i]] + step * gradient[indices[i]]
        new_area = surface_area(new_v, f)


    # TODO: now update vertex positions in v
    for i in range(len(indices)):
        v[indices[i]] = new_v[indices[i]]
    # TODO: Check if new area differs only epsilon from old area
    # Return (True, area, v, gradient) to show that we converged and otherwise (False, area, v, gradient)
    if abs(new_area - area) < epsilon:
        return True, new_area, v, gradient
    else:
        return False, area, v, gradient


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
