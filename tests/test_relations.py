import unittest

from py_math import (
    Curve2D,
    Implicit2D,
    Implicit3D,
    PlaneX,
    PlaneY,
    PlaneZ,
    Point3D,
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

    def test_draw3d_items_wraps_plain_function(self) -> None:
        def relation3d():
            return [plane_x(0.0), point3d(0.0, 0.0, 0.0)]

        wrapped = plot3d_items()(relation3d)

        self.assertEqual(len(wrapped.items()), 2)
        self.assertTrue(callable(draw3d_items))


if __name__ == "__main__":
    unittest.main()
