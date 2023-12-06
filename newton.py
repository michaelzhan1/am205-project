import autograd.numpy as np
from autograd import grad, jacobian


def f(x):
    """Function to optimize."""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def gradient(x):
    """Gradient of f."""
    return grad(f)(x)

def hessian(x):
    """Hessian of f."""
    return jacobian(gradient)(x)

def newton(x0, tol=1e-15, maxiter=1000):
    """Newton's method."""
    x = x0
    for i in range(maxiter):
        x = x - np.linalg.pinv(hessian(x)) @ gradient(x)
        if np.linalg.norm(gradient(x)) < tol:
            break
    return x

def main():
    x0 = np.array([1.2, 1.2])
    x = newton(x0)
    print(x)


if __name__ == "__main__":
    main()
