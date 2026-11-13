"""Focused tests for quantities plotted from a model atmosphere."""
import unittest

import numpy as np

from plot_model_file import radiation_anisotropy, stokes_percent


class PlotQuantitiesTests(unittest.TestCase):
    def test_stokes_percent_and_unit_invariance(self):
        stokes = np.array([[2., 4.], [0.2, -0.8], [-0.1, 0.4]])
        expected = np.array([[2., 4.], [10., -20.], [-5., 10.]])
        actual = stokes_percent(stokes)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(stokes_percent(stokes * 1000)[1:], expected[1:])
        np.testing.assert_array_equal(stokes[0], [2., 4.])

    def test_nonpositive_intensity_is_omitted_with_warning(self):
        stokes = np.array([[1., 0., -1.], [0.2, 0., 0.1], [0.1, 0., 0.2]])
        with self.assertWarnsRegex(RuntimeWarning, "non-positive I"):
            result = stokes_percent(stokes)
        np.testing.assert_allclose(result[1:, 0], [20., 10.])
        self.assertTrue(np.all(np.isnan(result[1:, 1:])))

    def test_anisotropy_uses_both_prescribed_moments(self):
        col = {"Jbar00": np.array([2., 4.]), "Jbar20": np.array([1., -1.])}
        np.testing.assert_allclose(radiation_anisotropy(col), np.sqrt(2) * np.array([0.5, -0.25]))
        col["Jbar00"][0] = 0.
        with self.assertRaisesRegex(ValueError, "Jbar00"):
            radiation_anisotropy(col)


if __name__ == "__main__":
    unittest.main()
