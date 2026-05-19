from py_math import (
    constraint_gt,
    constraint_lt,
    implicit2d,
    plot_items,
    region2d,
)


@plot_items(x_range=(-4, 4), y_range=(-4, 4), title="Single constrained region")
def single_region():
    return [
        region2d(
            [constraint_lt(lambda x, y: x * x + y * y, 10)],
            label="x^2 + y^2 < 10",
            color="tab:blue",
            alpha=0.25,
        ),
        implicit2d(lambda x, y: x * x + y * y - 10, label="x^2 + y^2 = 10", color="tab:blue"),
    ]


@plot_items(x_range=(-1, 5), y_range=(-1, 5), title="Multiple constrained regions")
def multi_regions():
    return [
        region2d(
            [
                constraint_lt(lambda x, y: x * x + y * y, 10),
                constraint_gt(lambda x, y: x, 1),
                constraint_gt(lambda x, y: y, 1),
            ],
            label="x^2 + y^2 < 10, x > 1, y > 1",
            color="tab:orange",
            alpha=0.30,
        ),
        region2d(
            [
                constraint_lt(lambda x, y: (x - 2) * (x - 2) + y * y, 4),
                constraint_gt(lambda x, y: y, 0),
            ],
            label="(x-2)^2 + y^2 < 4, y > 0",
            color="tab:green",
            alpha=0.25,
        ),
        implicit2d(lambda x, y: x * x + y * y - 10, label="x^2 + y^2 = 10", color="tab:orange"),
        implicit2d(lambda x, y: (x - 2) * (x - 2) + y * y - 4, label="(x-2)^2 + y^2 = 4", color="tab:green"),
    ]


if __name__ == "__main__":
    single_region.plot()
    multi_regions.plot()
