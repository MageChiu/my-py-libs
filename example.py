import math

import numpy as np

from py_math import plot_function, plot_functions


@plot_function(x_range=(-5, 5), samples=300, color="tomato", title="y = x^3")
def f(x: float) -> float:
    return x ** 3


@plot_function(x_range=(-5, 5), samples=300, color="tab:orange", title="y = sin(x)")
def sin(x: float) -> float:
    return math.sin(x)


@plot_function(x_range=(-5 * np.pi, 5 * np.pi), samples=300, color="tab:blue", title="y = cos(x)")
def cos(x: float) -> float:
    return math.cos(x)


@plot_functions(
    x_range=(-5, 5),
    samples=300,
    labels=["x^3", "sin(x)", "cos(x)"],
    colors=["tomato", "tab:orange", "tab:blue"],
    title="multi function plot",
)
def p1(x: float):
    return [f(x), sin(x), cos(x)]


if __name__ == "__main__":
    p1.plot()
