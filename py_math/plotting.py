from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import update_wrapper
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class PlotConfig:
    x_min: float = -10.0
    x_max: float = 10.0
    samples: int = 400
    color: str = "tab:blue"
    title: Optional[str] = None
    xlabel: str = "x"
    ylabel: str = "f(x)"
    grid: bool = True

    def validate(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.samples < 2:
            raise ValueError("samples must be at least 2")


def _build_x_values(config: PlotConfig) -> list[float]:
    config.validate()
    step = (config.x_max - config.x_min) / (config.samples - 1)
    return [config.x_min + index * step for index in range(config.samples)]


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc
    return plt


class PlotFunction:
    """Callable wrapper that keeps plotting configuration on the function."""

    def __init__(self, func: Callable[[float], float], config: PlotConfig):
        update_wrapper(self, func)
        self._func = func
        self.plot_config = config

    def __call__(self, x: float) -> float:
        return self._func(x)

    def sample_points(self) -> tuple[list[float], list[float]]:
        xs = _build_x_values(self.plot_config)
        ys = [self._func(x) for x in xs]
        return xs, ys

    def plot(self, *, show: bool = True, save_path: str | Path | None = None):
        plt = _load_matplotlib()

        xs, ys = self.sample_points()

        fig, ax = plt.subplots()
        ax.plot(xs, ys, color=self.plot_config.color, label=self.__name__)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(self.plot_config.xlabel)
        ax.set_ylabel(self.plot_config.ylabel)
        ax.set_title(self.plot_config.title or f"Graph of {self.__name__}(x)")
        ax.grid(self.plot_config.grid)
        ax.legend()

        if save_path is not None:
            fig.savefig(Path(save_path), bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax


class PlotFunctions:
    """Callable wrapper for functions returning multiple y values."""

    def __init__(
        self,
        func: Callable[[float], Sequence[float]],
        config: PlotConfig,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ):
        update_wrapper(self, func)
        self._func = func
        self.plot_config = config
        self.labels = list(labels) if labels is not None else None
        self.colors = list(colors) if colors is not None else None

    def __call__(self, x: float) -> Sequence[float]:
        return self._func(x)

    def sample_points(self) -> tuple[list[float], list[list[float]]]:
        xs = _build_x_values(self.plot_config)
        series_values: list[list[float]] | None = None

        for x in xs:
            y_values = list(self._func(x))
            if not y_values:
                raise ValueError("multi-function plot expects at least one y value")
            if series_values is None:
                series_values = [[] for _ in range(len(y_values))]
            if len(y_values) != len(series_values):
                raise ValueError("each sample must return the same number of y values")
            for index, value in enumerate(y_values):
                series_values[index].append(value)

        if series_values is None:
            raise ValueError("unable to sample plot data")
        return xs, series_values

    def plot(self, *, show: bool = True, save_path: str | Path | None = None):
        plt = _load_matplotlib()
        xs, series_values = self.sample_points()

        if self.labels is not None and len(self.labels) != len(series_values):
            raise ValueError("labels length must match the number of plotted functions")
        if self.colors is not None and len(self.colors) != len(series_values):
            raise ValueError("colors length must match the number of plotted functions")

        fig, ax = plt.subplots()
        for index, ys in enumerate(series_values):
            label = self.labels[index] if self.labels is not None else f"{self.__name__}[{index}]"
            color = self.colors[index] if self.colors is not None else None
            ax.plot(xs, ys, color=color, label=label)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(self.plot_config.xlabel)
        ax.set_ylabel(self.plot_config.ylabel)
        ax.set_title(self.plot_config.title or f"Graph of {self.__name__}(x)")
        ax.grid(self.plot_config.grid)
        ax.legend()

        if save_path is not None:
            fig.savefig(Path(save_path), bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax


def plot_function(
    *,
    x_range: tuple[float, float] = (-10.0, 10.0),
    samples: int = 400,
    color: str = "tab:blue",
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "f(x)",
    grid: bool = True,
) -> Callable[[Callable[[float], float]], PlotFunction]:
    """Decorator used to attach plot configuration to a single-variable function."""

    def decorator(func: Callable[[float], float]) -> PlotFunction:
        config = PlotConfig(
            x_min=x_range[0],
            x_max=x_range[1],
            samples=samples,
            color=color,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
        )
        return PlotFunction(func, config)

    return decorator


def plot_functions(
    *,
    x_range: tuple[float, float] = (-10.0, 10.0),
    samples: int = 400,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "f(x)",
    grid: bool = True,
) -> Callable[[Callable[[float], Sequence[float]]], PlotFunctions]:
    """Decorator used to draw multiple function curves from one function."""

    def decorator(func: Callable[[float], Sequence[float]]) -> PlotFunctions:
        config = PlotConfig(
            x_min=x_range[0],
            x_max=x_range[1],
            samples=samples,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
        )
        return PlotFunctions(func, config, labels=labels, colors=colors)

    return decorator


def draw_function(
    func: Callable[[float], float] | PlotFunction,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """Draw a function graph. Plain functions use default configuration."""

    plot_target = func if isinstance(func, PlotFunction) else plot_function()(func)
    return plot_target.plot(show=show, save_path=save_path)


def draw_functions(
    func: Callable[[float], Sequence[float]] | PlotFunctions,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """Draw multiple function curves. Plain functions use default configuration."""

    plot_target = func if isinstance(func, PlotFunctions) else plot_functions()(func)
    return plot_target.plot(show=show, save_path=save_path)
