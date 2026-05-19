from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import update_wrapper
from pathlib import Path
from typing import Any

import numpy as np


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc
    return plt


@dataclass(frozen=True)
class Plot2DConfig:
    x_range: tuple[float, float] = (-10.0, 10.0)
    y_range: tuple[float, float] = (-10.0, 10.0)
    samples: int = 400
    title: str | None = None
    xlabel: str = "x"
    ylabel: str = "y"
    grid: bool = True

    def validate(self) -> None:
        if self.x_range[1] <= self.x_range[0]:
            raise ValueError("x_range max must be greater than min")
        if self.y_range[1] <= self.y_range[0]:
            raise ValueError("y_range max must be greater than min")
        if self.samples < 2:
            raise ValueError("samples must be at least 2")


@dataclass(frozen=True)
class Plot3DConfig:
    x_range: tuple[float, float] = (-2.0, 2.0)
    y_range: tuple[float, float] = (-2.0, 2.0)
    z_range: tuple[float, float] = (-2.0, 2.0)
    samples: int = 24
    title: str | None = None
    xlabel: str = "x"
    ylabel: str = "y"
    zlabel: str = "z"

    def validate(self) -> None:
        if self.x_range[1] <= self.x_range[0]:
            raise ValueError("x_range max must be greater than min")
        if self.y_range[1] <= self.y_range[0]:
            raise ValueError("y_range max must be greater than min")
        if self.z_range[1] <= self.z_range[0]:
            raise ValueError("z_range max must be greater than min")
        if self.samples < 2:
            raise ValueError("samples must be at least 2")


@dataclass(frozen=True)
class Curve2D:
    func: Callable[[float], float]
    label: str | None = None
    color: str | None = None
    linestyle: str = "-"
    linewidth: float = 1.5
    samples: int | None = None


@dataclass(frozen=True)
class VerticalLine2D:
    x: float
    label: str | None = None
    color: str | None = None
    linestyle: str = "--"
    linewidth: float = 1.2


@dataclass(frozen=True)
class HorizontalLine2D:
    y: float
    label: str | None = None
    color: str | None = None
    linestyle: str = "--"
    linewidth: float = 1.2


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    label: str | None = None
    color: str | None = None
    marker: str = "o"
    size: float = 40.0


@dataclass(frozen=True)
class Implicit2D:
    func: Callable[[float, float], float]
    level: float = 0.0
    label: str | None = None
    color: str | None = None
    linestyle: str = "-"
    linewidth: float = 1.5
    samples: int = 250


@dataclass(frozen=True)
class PlaneX:
    x: float
    label: str | None = None
    color: str = "tab:red"
    alpha: float = 0.3


@dataclass(frozen=True)
class PlaneY:
    y: float
    label: str | None = None
    color: str = "tab:green"
    alpha: float = 0.3


@dataclass(frozen=True)
class PlaneZ:
    z: float
    label: str | None = None
    color: str = "tab:blue"
    alpha: float = 0.3


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float
    label: str | None = None
    color: str | None = None
    marker: str = "o"
    size: float = 36.0


@dataclass(frozen=True)
class Implicit3D:
    func: Callable[[float, float, float], float]
    level: float = 0.0
    label: str | None = None
    color: str = "tab:purple"
    alpha: float = 0.35
    samples: int = 24
    tolerance: float | None = None
    marker_size: float = 6.0


Plot2DItem = Curve2D | VerticalLine2D | HorizontalLine2D | Point2D | Implicit2D
Plot3DItem = PlaneX | PlaneY | PlaneZ | Point3D | Implicit3D


def curve(
    func: Callable[[float], float],
    *,
    label: str | None = None,
    color: str | None = None,
    linestyle: str = "-",
    linewidth: float = 1.5,
    samples: int | None = None,
) -> Curve2D:
    return Curve2D(
        func=func,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        samples=samples,
    )


def vline(
    x: float,
    *,
    label: str | None = None,
    color: str | None = None,
    linestyle: str = "--",
    linewidth: float = 1.2,
) -> VerticalLine2D:
    return VerticalLine2D(
        x=x,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def hline(
    y: float,
    *,
    label: str | None = None,
    color: str | None = None,
    linestyle: str = "--",
    linewidth: float = 1.2,
) -> HorizontalLine2D:
    return HorizontalLine2D(
        y=y,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def point(
    x: float,
    y: float,
    *,
    label: str | None = None,
    color: str | None = None,
    marker: str = "o",
    size: float = 40.0,
) -> Point2D:
    return Point2D(x=x, y=y, label=label, color=color, marker=marker, size=size)


def implicit2d(
    func: Callable[[float, float], float],
    *,
    level: float = 0.0,
    label: str | None = None,
    color: str | None = None,
    linestyle: str = "-",
    linewidth: float = 1.5,
    samples: int = 250,
) -> Implicit2D:
    return Implicit2D(
        func=func,
        level=level,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        samples=samples,
    )


def plane_x(
    x: float,
    *,
    label: str | None = None,
    color: str = "tab:red",
    alpha: float = 0.3,
) -> PlaneX:
    return PlaneX(x=x, label=label, color=color, alpha=alpha)


def plane_y(
    y: float,
    *,
    label: str | None = None,
    color: str = "tab:green",
    alpha: float = 0.3,
) -> PlaneY:
    return PlaneY(y=y, label=label, color=color, alpha=alpha)


def plane_z(
    z: float,
    *,
    label: str | None = None,
    color: str = "tab:blue",
    alpha: float = 0.3,
) -> PlaneZ:
    return PlaneZ(z=z, label=label, color=color, alpha=alpha)


def point3d(
    x: float,
    y: float,
    z: float,
    *,
    label: str | None = None,
    color: str | None = None,
    marker: str = "o",
    size: float = 36.0,
) -> Point3D:
    return Point3D(x=x, y=y, z=z, label=label, color=color, marker=marker, size=size)


def implicit3d(
    func: Callable[[float, float, float], float],
    *,
    level: float = 0.0,
    label: str | None = None,
    color: str = "tab:purple",
    alpha: float = 0.35,
    samples: int = 24,
    tolerance: float | None = None,
    marker_size: float = 6.0,
) -> Implicit3D:
    return Implicit3D(
        func=func,
        level=level,
        label=label,
        color=color,
        alpha=alpha,
        samples=samples,
        tolerance=tolerance,
        marker_size=marker_size,
    )


def _ensure_items(items: Sequence[Any], *, dimension: str) -> list[Any]:
    resolved = list(items)
    if not resolved:
        raise ValueError(f"{dimension} plot expects at least one item")
    return resolved


def _add_legend_proxy_2d(ax, *, label: str | None, color: str | None, linestyle: str) -> None:
    if label:
        ax.plot([], [], color=color, linestyle=linestyle, label=label)


def _add_legend_proxy_3d(ax, *, label: str | None, color: str) -> None:
    if label:
        ax.plot([], [], [], color=color, label=label)


def _vectorize_2d(func: Callable[[float, float], float]):
    return np.vectorize(func, otypes=[float])


def _vectorize_3d(func: Callable[[float, float, float], float]):
    return np.vectorize(func, otypes=[float])


def _auto_tolerance(values: np.ndarray) -> float:
    return max(float(np.percentile(np.abs(values), 2.5)), 1e-3)


def _render_2d_item(ax, item: Plot2DItem, config: Plot2DConfig) -> None:
    if isinstance(item, Curve2D):
        sample_count = item.samples or config.samples
        xs = np.linspace(config.x_range[0], config.x_range[1], sample_count)
        ys = [item.func(float(x)) for x in xs]
        ax.plot(
            xs,
            ys,
            label=item.label,
            color=item.color,
            linestyle=item.linestyle,
            linewidth=item.linewidth,
        )
        return

    if isinstance(item, VerticalLine2D):
        ax.axvline(
            item.x,
            label=item.label,
            color=item.color,
            linestyle=item.linestyle,
            linewidth=item.linewidth,
        )
        return

    if isinstance(item, HorizontalLine2D):
        ax.axhline(
            item.y,
            label=item.label,
            color=item.color,
            linestyle=item.linestyle,
            linewidth=item.linewidth,
        )
        return

    if isinstance(item, Point2D):
        ax.scatter(item.x, item.y, label=item.label, color=item.color, marker=item.marker, s=item.size)
        return

    if isinstance(item, Implicit2D):
        xs = np.linspace(config.x_range[0], config.x_range[1], item.samples)
        ys = np.linspace(config.y_range[0], config.y_range[1], item.samples)
        grid_x, grid_y = np.meshgrid(xs, ys)
        values = _vectorize_2d(item.func)(grid_x, grid_y)
        ax.contour(
            grid_x,
            grid_y,
            values,
            levels=[item.level],
            colors=[item.color] if item.color else None,
            linewidths=item.linewidth,
            linestyles=item.linestyle,
        )
        _add_legend_proxy_2d(
            ax,
            label=item.label,
            color=item.color or "tab:blue",
            linestyle=item.linestyle,
        )
        return

    raise TypeError(f"unsupported 2D plot item: {type(item)!r}")


def _render_3d_item(ax, item: Plot3DItem, config: Plot3DConfig) -> None:
    grid_y = np.linspace(config.y_range[0], config.y_range[1], config.samples)
    grid_z = np.linspace(config.z_range[0], config.z_range[1], config.samples)
    grid_x = np.linspace(config.x_range[0], config.x_range[1], config.samples)

    if isinstance(item, PlaneX):
        yy, zz = np.meshgrid(grid_y, grid_z)
        xx = np.full_like(yy, item.x)
        ax.plot_surface(xx, yy, zz, color=item.color, alpha=item.alpha, linewidth=0, shade=False)
        _add_legend_proxy_3d(ax, label=item.label, color=item.color)
        return

    if isinstance(item, PlaneY):
        xx, zz = np.meshgrid(grid_x, grid_z)
        yy = np.full_like(xx, item.y)
        ax.plot_surface(xx, yy, zz, color=item.color, alpha=item.alpha, linewidth=0, shade=False)
        _add_legend_proxy_3d(ax, label=item.label, color=item.color)
        return

    if isinstance(item, PlaneZ):
        xx, yy = np.meshgrid(grid_x, grid_y)
        zz = np.full_like(xx, item.z)
        ax.plot_surface(xx, yy, zz, color=item.color, alpha=item.alpha, linewidth=0, shade=False)
        _add_legend_proxy_3d(ax, label=item.label, color=item.color)
        return

    if isinstance(item, Point3D):
        ax.scatter(
            item.x,
            item.y,
            item.z,
            label=item.label,
            color=item.color,
            marker=item.marker,
            s=item.size,
        )
        return

    if isinstance(item, Implicit3D):
        xs = np.linspace(config.x_range[0], config.x_range[1], item.samples)
        ys = np.linspace(config.y_range[0], config.y_range[1], item.samples)
        zs = np.linspace(config.z_range[0], config.z_range[1], item.samples)
        grid_x3, grid_y3, grid_z3 = np.meshgrid(xs, ys, zs, indexing="ij")
        values = _vectorize_3d(item.func)(grid_x3, grid_y3, grid_z3)
        tolerance = item.tolerance or _auto_tolerance(values - item.level)
        mask = np.abs(values - item.level) <= tolerance

        if not np.any(mask):
            nearest = np.abs(values - item.level)
            mask = nearest == np.min(nearest)

        ax.scatter(
            grid_x3[mask],
            grid_y3[mask],
            grid_z3[mask],
            color=item.color,
            alpha=item.alpha,
            s=item.marker_size,
        )
        _add_legend_proxy_3d(ax, label=item.label, color=item.color)
        return

    raise TypeError(f"unsupported 3D plot item: {type(item)!r}")


class PlotItems:
    """Callable wrapper for 2D plot item collections."""

    def __init__(self, func: Callable[[], Sequence[Plot2DItem]], config: Plot2DConfig):
        update_wrapper(self, func)
        self._func = func
        self.plot_config = config

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def items(self) -> list[Plot2DItem]:
        self.plot_config.validate()
        return _ensure_items(self._func(), dimension="2D")

    def plot(self, *, show: bool = True, save_path: str | Path | None = None):
        plt = _load_matplotlib()
        fig, ax = plt.subplots()

        for item in self.items():
            _render_2d_item(ax, item, self.plot_config)

        ax.set_xlim(*self.plot_config.x_range)
        ax.set_ylim(*self.plot_config.y_range)
        ax.set_xlabel(self.plot_config.xlabel)
        ax.set_ylabel(self.plot_config.ylabel)
        ax.set_title(self.plot_config.title or self.__name__)
        ax.grid(self.plot_config.grid)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

        if save_path is not None:
            fig.savefig(Path(save_path), bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax


class Plot3DItems:
    """Callable wrapper for 3D plot item collections."""

    def __init__(self, func: Callable[[], Sequence[Plot3DItem]], config: Plot3DConfig):
        update_wrapper(self, func)
        self._func = func
        self.plot_config = config

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def items(self) -> list[Plot3DItem]:
        self.plot_config.validate()
        return _ensure_items(self._func(), dimension="3D")

    def plot(self, *, show: bool = True, save_path: str | Path | None = None):
        plt = _load_matplotlib()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for item in self.items():
            _render_3d_item(ax, item, self.plot_config)

        ax.set_xlim(*self.plot_config.x_range)
        ax.set_ylim(*self.plot_config.y_range)
        ax.set_zlim(*self.plot_config.z_range)
        ax.set_xlabel(self.plot_config.xlabel)
        ax.set_ylabel(self.plot_config.ylabel)
        ax.set_zlabel(self.plot_config.zlabel)
        ax.set_title(self.plot_config.title or self.__name__)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

        if save_path is not None:
            fig.savefig(Path(save_path), bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax


def plot_items(
    *,
    x_range: tuple[float, float] = (-10.0, 10.0),
    y_range: tuple[float, float] = (-10.0, 10.0),
    samples: int = 400,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    grid: bool = True,
) -> Callable[[Callable[[], Sequence[Plot2DItem]]], PlotItems]:
    """Decorator for 2D mixed plot items."""

    def decorator(func: Callable[[], Sequence[Plot2DItem]]) -> PlotItems:
        config = Plot2DConfig(
            x_range=x_range,
            y_range=y_range,
            samples=samples,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
        )
        return PlotItems(func, config)

    return decorator


def plot3d_items(
    *,
    x_range: tuple[float, float] = (-2.0, 2.0),
    y_range: tuple[float, float] = (-2.0, 2.0),
    z_range: tuple[float, float] = (-2.0, 2.0),
    samples: int = 24,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
) -> Callable[[Callable[[], Sequence[Plot3DItem]]], Plot3DItems]:
    """Decorator for 3D mixed plot items."""

    def decorator(func: Callable[[], Sequence[Plot3DItem]]) -> Plot3DItems:
        config = Plot3DConfig(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            samples=samples,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
        )
        return Plot3DItems(func, config)

    return decorator


def draw_items(
    func: Callable[[], Sequence[Plot2DItem]] | PlotItems,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    plot_target = func if isinstance(func, PlotItems) else plot_items()(func)
    return plot_target.plot(show=show, save_path=save_path)


def draw3d_items(
    func: Callable[[], Sequence[Plot3DItem]] | Plot3DItems,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    plot_target = func if isinstance(func, Plot3DItems) else plot3d_items()(func)
    return plot_target.plot(show=show, save_path=save_path)
