from matplotlib import pyplot as plt
from matplotlib import colors as col
from math import pi
import numpy as np


def rastrigin(x):
    a = 10
    n = 10
    return a*n + x ** 2 - a * np.cos(2 * pi * x)


def dx_rastrigin(x):
    n = 10
    a = 10
    return 2 * x + a * np.sin(2 * pi * x)


n = 200
width = 5.12 * 2
xg = np.linspace(-width, width, n)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1 = ax.plot(xg, rastrigin(xg))
step_size_l = [0.1, 0.05, 0.02, 0.01, 0.001]
colors = ['r', 'y', 'g', 'b', 'm']

for run in range(5):
    step_size = step_size_l[run]
    x = np.random.uniform(-width, width, 1)
    line2 = ax.scatter(x, rastrigin(x), label='step_size ' + str(step_size), c=col.to_rgb(colors[run]))
    ax.legend()

    for _ in range(20):
        step = dx_rastrigin(x) * step_size
        x = x - step

        r = rastrigin(x)
        line2.set_offsets(np.c_[x, rastrigin(x)])

        fig.canvas.draw_idle()
        plt.pause(0.2)