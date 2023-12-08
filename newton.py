import numpy as np
import numdifftools as nd
from deap import benchmarks


def f(x):
    """Function to optimize."""
    return benchmarks.ackley(x)[0]

def gradient(x):
    """Gradient of f."""
    return nd.Gradient(f)(x)

def hessian(x):
    """Hessian of f."""
    return nd.Hessian(f)(x)

def newton(x0, tol=1e-10, maxiter=1000):
    """Newton's method."""
    x = x0
    for i in range(maxiter):
        x = x - np.linalg.pinv(hessian(x)) @ gradient(x)
        if np.linalg.norm(gradient(x)) < tol:
            break
    else:
        print(f'Failed to reach tolerance {tol} in {maxiter} iterations.')
    return x

def main():
    x0 = np.array([1.2, 1.2])
    x = newton(x0)
    print(x)


if __name__ == "__main__":
    main()
