from py_math import (
    constraint3d_gt,
    constraint3d_lt,
    plane_x,
    plane_y,
    plot3d_items,
    point3d,
    region3d,
)


@plot3d_items(
    x_range=(-2, 2),
    y_range=(-2, 2),
    z_range=(-2, 2),
    samples=28,
    title="3D constrained volume surface",
)
def single_volume():
    return [
        region3d(
            [constraint3d_lt(lambda x, y, z: x * x + y * y + z * z, 2.5)],
            label="x^2 + y^2 + z^2 < 2.5",
            color="tab:blue",
            alpha=0.20,
            samples=32,
            mode="surface",
            show_mesh=True,
            mesh_color="midnightblue",
            mesh_linewidth=0.25,
            shade=True,
            light_azimuth=35,
            light_altitude=42,
            show_outline=True,
            outline_color="black",
            outline_linewidth=1.1,
        ),
        point3d(0.0, 0.0, 0.0, label="origin", color="black"),
    ]


@plot3d_items(
    x_range=(-2, 2),
    y_range=(-2, 2),
    z_range=(-2, 2),
    samples=24,
    title="Multiple constrained volumes",
)
def multi_volume():
    return [
        region3d(
            [
                constraint3d_lt(lambda x, y, z: x * x + y * y + z * z, 2.5),
                constraint3d_gt(lambda x, y, z: x, 0.0),
            ],
            label="sphere & x > 0",
            color="tab:orange",
            alpha=0.18,
            samples=26,
            mode="surface",
            show_mesh=True,
            mesh_color="saddlebrown",
            mesh_linewidth=0.22,
            shade=True,
            light_azimuth=30,
            light_altitude=38,
            show_outline=True,
            outline_color="black",
            outline_linewidth=0.9,
        ),
        region3d(
            [
                constraint3d_lt(lambda x, y, z: (x - 0.5) ** 2 + y * y + z * z, 1.2),
                constraint3d_gt(lambda x, y, z: y, -0.2),
            ],
            label="shifted sphere cap",
            color="tab:green",
            alpha=0.14,
            samples=24,
            mode="surface",
            show_mesh=True,
            mesh_color="darkgreen",
            mesh_linewidth=0.22,
            shade=True,
            light_azimuth=55,
            light_altitude=35,
            show_outline=True,
            outline_color="black",
            outline_linewidth=0.9,
        ),
        plane_x(0.0, label="x = 0", color="tab:red", alpha=0.12),
        plane_y(0.0, label="y = 0", color="tab:purple", alpha=0.12),
    ]


if __name__ == "__main__":
    single_volume.plot()
    multi_volume.plot()
