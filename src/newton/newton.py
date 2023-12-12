import numpy as np
import numdifftools as nd
from deap import benchmarks


def newton(f, x0, tol=1e-10, maxiter=1000):
    """Newton's method."""
    def gradient(x):
        return nd.Gradient(f)(x)

    def hessian(x):
        return nd.Hessian(f)(x)

    count = 0
    x = x0
    prev = x0
    for i in range(maxiter):
        count += 1
        x = x - np.linalg.pinv(hessian(x)) @ gradient(x)
        if np.linalg.norm(x - prev) < tol:
            break
        prev = x
    else:
        print(f'Failed to reach tolerance {tol} in {maxiter} iterations.')
    return x, count

def main():
    x0 = np.array([1e-9, 1e-9])
    x = newton(lambda x: benchmarks.ackley(x)[0], x0)
    print(x)


if __name__ == "__main__":
    main()
