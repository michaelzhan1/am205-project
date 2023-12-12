import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots()
all_x = []
all_y = []


def initialize_plot(f):
    global X, Y, Z
    X = np.arange(-1, 1, 0.5)
    Y = np.arange(-1, 1, 0.5)
    X, Y = np.meshgrid(X, Y)
    Z = np.fromiter(map(f, zip(X.flat,Y.flat)),  dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    ax.contourf(X, Y, Z)


def plot_points(f, x, fig_title):
    global X, Y, Z
    ax.clear()
    ax.contourf(X, Y, Z)
    all_x.append(x[0])
    all_y.append(x[1])
    ax.plot(all_x, all_y, 'r-')
    plt.savefig(fig_title)
