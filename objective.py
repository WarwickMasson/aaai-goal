import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

def centre_function(theta):
    return np.sin(theta)*np.cos(theta)**2

def line_between(x1, y1, x2, y2, z, colour):
    x = np.arange(x1, x2, 0.1)
    y = np.arange(y1, y2, 0.1)
    if x.size == 0:
        x = x1 * np.ones(y.size)
    if y.size == 0:
        y = y1 * np.ones(x.size)
    ax.plot(y, x, '--'+colour, zs = z)

def curve_between(x1, y1, x2, y2, colour):
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x = np.arange(x1, x2, 0.1)
    y = np.arange(y1, y2, 0.1)
    if x.size == 0:
        x = x1 * np.ones(y.size)
    if y.size == 0:
        y = y1 * np.ones(x.size)
    z = objective(x, y)
    ax.plot(y, x, '-'+colour, zs = z)

def function(theta):
    A = 1
    B = -1
    C = 5
    return A*theta**3 + B*theta**2 + C*theta

def objective(theta, omega):
    A = 300
    B = 5
    return A + function(theta) - B*(centre_function(theta) - omega)**2

xs = np.arange(-5, 5, 0.1)
ys = centre_function(xs)
ax.plot(ys, xs, '-r', zs = objective(xs, ys))
ax.plot(0*np.ones(xs.size)+5, xs, '-g', zs = 100*ys)
curve_between(-5, 4.9, -5, 5, 'c')
curve_between(5, 4.9, 5, 5, 'y')
ax.plot(ys, xs, '-g', zs = 0*xs)
ax.plot(0*ys + 5, xs, '-r', zs = objective(xs, ys))
line_between(-5, 0, -5, 5, 125, 'r')
line_between(5, 0, 5, 5, 410, 'r')
line_between(-5, 0, -5, 5, 0, 'g')
line_between(5, 0, 5, 5, 0, 'g')
line_between(-5, 0, -5, 0.1, np.arange(0, 125, 0.1), 'g')
line_between(5, 0, 5, 0.01, np.arange(0, 410, 0.1), 'g')

def plot_diffs(diff, x, y, colour):
    while x < 6 - diff:
        curve_between(x, y, x, centre_function(x), colour)
        y = centre_function(x)
        curve_between(x, y, x+diff, y, colour)
        x += diff
    curve_between(x, y, x, centre_function(x), colour)

ax.yaxis.set_ticklabels([])
ax.xaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
plot_diffs(2.5, -3.5, -4.0, 'c')
plot_diffs(0.25, -2, -2.5, 'y')

X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
R = objective(X, Y)
ax.plot_surface(Y, X, R)
plt.title('$J(\\theta, \\omega)$')
ax.set_xlabel('$\\omega$')
ax.set_ylabel('$\\theta$')
ax.set_zlabel('Expected Return')
plt.legend(['$H(\\theta)$', '$W(\\theta)$', 'AO', 'Q-PAMDP'], loc = 'upper left')
plt.savefig('runs/objective', bbox_inches = 'tight')
