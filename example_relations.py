from py_math import (
    curve,
    implicit2d,
    implicit3d,
    plane_x,
    plane_y,
    plane_z,
    plot3d_items,
    plot_items,
    point,
    point3d,
    vline,
)


@plot_items(x_range=(-2, 2), y_range=(-2, 2), title="2D relations")
def relation_2d():
    return [
        curve(lambda x: x**3, label="y = x^3", color="tomato"),
        vline(0.5, label="x = 0.5", color="tab:red"),
        implicit2d(
            lambda x, y: x * x + y * y - 1,
            label="x^2 + y^2 = 1",
            color="tab:blue",
        ),
        point(0.5, 0.5**3, label="(0.5, f(0.5))", color="black"),
    ]


@plot3d_items(
    x_range=(-1.5, 1.5),
    y_range=(-1.5, 1.5),
    z_range=(-1.5, 1.5),
    samples=28,
    title="3D relations",
)
def relation_3d():
    return [
        plane_x(0.3, label="x = 0.3"),
        plane_y(0.5, label="y = 0.5"),
        plane_z(0.4, label="z = 0.4"),
        point3d(0.3, 0.5, 0.4, label="intersection", color="black"),
        implicit3d(
            lambda x, y, z: x * x * x + y * y * y + z * z * z - 1,
            label="x^3 + y^3 + z^3 = 1",
            color="tab:purple",
            samples=26,
        ),
    ]



@plot_items(x_range=(-1, 2), y_range=(-1, 2), title="2D relations")
def sanjiaoxing_2d():
    return [
        implicit2d(
            lambda x, y: y-x - 0.5,
            label="y-x = 0.5",
            color="tab:blue",
        ),
        implicit2d(
            lambda x, y: y - 0.5,
            label="y = 0.5",
            color="tab:blue",
        ),
        implicit2d(
            lambda x, y: x - 0.5,
            label="x = 0.5",
            color="tab:blue",
        ),
        # new 
        implicit2d(
            lambda x, y: y-x ,
            label="y > x",
            color="tab:red",
        ),
        implicit2d(
            lambda x, y: x  ,
            label="x > 0",
            color="tab:red",
        ),
        implicit2d(
            lambda x, y: y - 1 ,
            label="y < 1",
            color="tab:red",
        ),
    ]


@plot_items(x_range=(-2, 2), y_range=(-2, 2), title="2D relations")
def p2():
    return [
        implicit2d(
            lambda x, y: x * x / 4 + y*y -1 ,
            label="x^2/4 + y^2 = 1",
            color="tab:blue",
        ),
    ]


if __name__ == "__main__":
    # relation_2d.plot()
    # relation_3d.plot()
    # sanjiaoxing_2d.plot()
    p2.plot()
