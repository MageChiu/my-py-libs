import unittest

from py_math import draw_function, draw_functions, plot_function, plot_functions


class PlottingDecoratorTests(unittest.TestCase):
    def test_decorator_keeps_function_callable(self) -> None:
        @plot_function(x_range=(-2, 2), samples=5, title="square")
        def square(x: float) -> float:
            return x * x

        self.assertEqual(square(3), 9)
        self.assertEqual(square.__name__, "square")
        self.assertEqual(square.plot_config.title, "square")

    def test_sample_points_follow_config(self) -> None:
        @plot_function(x_range=(-1, 1), samples=3)
        def cube(x: float) -> float:
            return x * x * x

        xs, ys = cube.sample_points()

        self.assertEqual(xs, [-1.0, 0.0, 1.0])
        self.assertEqual(ys, [-1.0, 0.0, 1.0])

    def test_draw_function_wraps_plain_function(self) -> None:
        def linear(x: float) -> float:
            return 2 * x + 1

        wrapped = plot_function()(linear)
        xs, ys = wrapped.sample_points()

        self.assertEqual(xs[0], -10.0)
        self.assertEqual(xs[-1], 10.0)
        self.assertEqual(ys[0], -19.0)
        self.assertEqual(ys[-1], 21.0)
        self.assertTrue(callable(draw_function))

    def test_multi_plot_decorator_samples_multiple_series(self) -> None:
        @plot_functions(
            x_range=(-1, 1),
            samples=3,
            labels=["linear", "square"],
            colors=["tab:red", "tab:green"],
        )
        def bundle(x: float):
            return [x, x * x]

        xs, series = bundle.sample_points()

        self.assertEqual(xs, [-1.0, 0.0, 1.0])
        self.assertEqual(series, [[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        self.assertEqual(bundle.labels, ["linear", "square"])

    def test_draw_functions_wraps_plain_multi_function(self) -> None:
        def bundle(x: float):
            return [x, 2 * x]

        wrapped = plot_functions()(bundle)
        xs, series = wrapped.sample_points()

        self.assertEqual(xs[0], -10.0)
        self.assertEqual(xs[-1], 10.0)
        self.assertEqual(series[0][0], -10.0)
        self.assertEqual(series[1][-1], 20.0)
        self.assertTrue(callable(draw_functions))


if __name__ == "__main__":
    unittest.main()
