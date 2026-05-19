import unittest

from py_math import (
    Constraint2D,
    Constraint3D,
    Curve2D,
    Implicit2D,
    Implicit3D,
    PlaneX,
    PlaneY,
    PlaneZ,
    Point3D,
    Region2D,
    Region3D,
    constraint3d_gt,
    constraint3d_lt,
    constraint_gt,
    constraint_lt,
    curve,
    draw3d_items,
    draw_items,
    implicit2d,
    implicit3d,
    plane_x,
    plane_y,
    plane_z,
    plot3d_items,
    plot_items,
    point3d,
    region2d,
    region3d,
    vline,
)


class RelationPlotTests(unittest.TestCase):
    def test_plot_items_keeps_callable_and_items(self) -> None:
        @plot_items(x_range=(-2, 2), y_range=(-2, 2), title="relations")
        def relation():
            return [
                curve(lambda x: x * x, label="square"),
                vline(0.5, label="x = 0.5"),
                implicit2d(lambda x, y: x * x + y * y - 1, label="circle"),
            ]

        items = relation.items()

        self.assertEqual(relation.__name__, "relation")
        self.assertEqual(relation.plot_config.title, "relations")
        self.assertEqual(len(items), 3)
        self.assertIsInstance(items[0], Curve2D)
        self.assertIsInstance(items[2], Implicit2D)

    def test_region2d_supports_multiple_constraints(self) -> None:
        @plot_items(x_range=(-1, 5), y_range=(-1, 5), title="region")
        def relation():
            return [
                region2d(
                    [
                        constraint_lt(lambda x, y: x * x + y * y, 10),
                        constraint_gt(lambda x, y: x, 1),
                        constraint_gt(lambda x, y: y, 1),
                    ],
                    label="intersection",
                    color="tab:orange",
                )
            ]

        items = relation.items()

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], Region2D)
        self.assertEqual(len(items[0].constraints), 3)
        self.assertTrue(all(isinstance(constraint, Constraint2D) for constraint in items[0].constraints))

    def test_draw_items_wraps_plain_function(self) -> None:
        def relation():
            return [curve(lambda x: x), vline(0.25)]

        wrapped = plot_items()(relation)

        self.assertEqual(len(wrapped.items()), 2)
        self.assertTrue(callable(draw_items))

    def test_plot3d_items_keeps_callable_and_items(self) -> None:
        @plot3d_items(
            x_range=(-1, 1),
            y_range=(-1, 1),
            z_range=(-1, 1),
            title="3d relations",
        )
        def relation3d():
            return [
                plane_x(0.3, label="x = 0.3"),
                plane_y(0.5, label="y = 0.5"),
                plane_z(0.4, label="z = 0.4"),
                point3d(0.3, 0.5, 0.4, label="point"),
                implicit3d(lambda x, y, z: x + y + z - 1, label="surface"),
            ]

        items = relation3d.items()

        self.assertEqual(relation3d.__name__, "relation3d")
        self.assertEqual(relation3d.plot_config.title, "3d relations")
        self.assertEqual(len(items), 5)
        self.assertIsInstance(items[0], PlaneX)
        self.assertIsInstance(items[1], PlaneY)
        self.assertIsInstance(items[2], PlaneZ)
        self.assertIsInstance(items[3], Point3D)
        self.assertIsInstance(items[4], Implicit3D)

    def test_region3d_supports_multiple_constraints(self) -> None:
        @plot3d_items(
            x_range=(-2, 2),
            y_range=(-2, 2),
            z_range=(-2, 2),
            title="volume",
        )
        def relation3d():
            return [
                region3d(
                    [
                        constraint3d_lt(lambda x, y, z: x * x + y * y + z * z, 2.5),
                        constraint3d_gt(lambda x, y, z: x, 0.0),
                    ],
                    label="volume",
                    color="tab:cyan",
                    samples=10,
                )
            ]

        items = relation3d.items()

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], Region3D)
        self.assertEqual(len(items[0].constraints), 2)
        self.assertTrue(all(isinstance(constraint, Constraint3D) for constraint in items[0].constraints))
        self.assertEqual(items[0].mode, "voxels")

    def test_region3d_surface_mode_is_available(self) -> None:
        @plot3d_items(
            x_range=(-2, 2),
            y_range=(-2, 2),
            z_range=(-2, 2),
            title="surface volume",
        )
        def relation3d():
            return [
                region3d(
                    [constraint3d_lt(lambda x, y, z: x * x + y * y + z * z, 2.5)],
                    label="surface sphere",
                    color="tab:blue",
                    samples=12,
                    mode="surface",
                    show_mesh=True,
                    mesh_color="navy",
                    mesh_linewidth=0.2,
                    shade=True,
                    light_azimuth=30,
                    light_altitude=40,
                    show_outline=True,
                    outline_color="black",
                    outline_linewidth=0.8,
                )
            ]

        items = relation3d.items()

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], Region3D)
        self.assertEqual(items[0].mode, "surface")
        self.assertTrue(items[0].show_mesh)
        self.assertEqual(items[0].mesh_color, "navy")
        self.assertEqual(items[0].mesh_linewidth, 0.2)
        self.assertTrue(items[0].shade)
        self.assertEqual(items[0].light_azimuth, 30)
        self.assertEqual(items[0].light_altitude, 40)
        self.assertTrue(items[0].show_outline)
        self.assertEqual(items[0].outline_color, "black")
        self.assertEqual(items[0].outline_linewidth, 0.8)

    def test_draw3d_items_wraps_plain_function(self) -> None:
        def relation3d():
            return [plane_x(0.0), point3d(0.0, 0.0, 0.0)]

        wrapped = plot3d_items()(relation3d)

        self.assertEqual(len(wrapped.items()), 2)
        self.assertTrue(callable(draw3d_items))


if __name__ == "__main__":
    unittest.main()
