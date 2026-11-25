"""Focused tests for quantities plotted from a model atmosphere."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import tempfile

import numpy as np

from atmosphere import C_LIGHT, LineModel
from grid_opacity import H_PLANCK, eta0
from line_profiles import voigt_function
from plot_model_file import (DEFAULT_MODEL, build_grid, plot_all_mu_spectra_doppler,
                             plot_spectrum_doppler, radiation_anisotropy, read_model,
                             resolve_model, stokes_percent)


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

    def test_population_opacity_and_spontaneous_emission(self):
        line = LineModel(6.5e14, 2e8, 1., 1., 1., 0., 87.6)
        col = {"height_km": np.array([1., 0.]), "T_K": np.array([5000., 5000.]),
               "n_lower": np.array([1e16, 1e16]), "n_upper": np.array([3e15, 3e15]),
               "C_ul": np.array([1e5, 1e5]), "a_damp": np.array([0.01, 0.01]),
               "Jbar00": np.array([1e-8, 1e-8]), "Jbar20": np.zeros(2)}
        grid = build_grid(col, line, np.zeros(2), np.array([0.5]), np.array([0.]),
                          np.ones(1), np.zeros(2), 0., use_populations=True)
        width = line.doppler_width(5000.)
        profile = voigt_function(0., 0.01).real / width
        n_l, n_u = 1e10, 3e9
        expected_chi = eta0(line, n_l - n_u / 3) * profile
        expected_j = H_PLANCK * line.nu0 / (4 * np.pi) * n_u * line.A_ul * profile
        np.testing.assert_allclose(grid.eta[:, 0, 0, 0, 0], expected_chi, rtol=1e-12)
        np.testing.assert_allclose(grid.emis[:, 0, 0, 0, 0], expected_j, rtol=1e-12)
        expected_source = (2 * H_PLANCK * line.nu0**3 / C_LIGHT**2
                           / (3 * n_l / n_u - 1))
        np.testing.assert_allclose(grid.emis[:, 0, 0, 0, 0] / grid.eta[:, 0, 0, 0, 0],
                                   expected_source, rtol=1e-12)
        grid.line_state.n_u[:] = 3 * n_l
        with self.assertRaisesRegex(ValueError, "opacity must be positive"):
            line.calc_j(grid)
        with self.assertRaisesRegex(ValueError, "n_u"):
            replace(grid.line_state, n_u=np.array([-1., 1.]))
        with self.assertRaisesRegex(ValueError, "does not update"):
            grid.solve_rte_sc(np.ones(1))

    def test_population_change_preserves_aligned_emissivity(self):
        line = LineModel(6.5e14, 2e8, 1., 1., 1., 0., 87.6)
        col = {"height_km": np.array([1., 0.]), "T_K": np.array([5000., 5000.]),
               "n_lower": np.array([1e16, 1e16]), "n_upper": np.array([3e15, 3e15]),
               "C_ul": np.array([1e5, 1e5]), "a_damp": np.array([0.01, 0.01]),
               "Jbar00": np.array([1e-8, 1e-8]), "Jbar20": np.array([1e-9, 1e-9])}
        grid = build_grid(col, line, np.zeros(2), np.array([0.5]), np.array([0.]),
                          np.ones(1), np.zeros(2), 0., use_populations=True)
        original = line.calc_j(grid)
        grid.line_state.n_u *= 0.5
        changed = line.calc_j(grid)
        self.assertTrue(np.all(original[:, 0] > changed[:, 0]))
        np.testing.assert_allclose(original[:, 1:], changed[:, 1:], rtol=1e-12, atol=1e-30)

    def test_updated_model_columns_and_top_offset(self):
        col, _ = read_model(resolve_model(DEFAULT_MODEL))
        self.assertIn("n_upper", col)
        self.assertIn("log_tau_line", col)
        total = 10 ** col["log_tau_line"]
        self.assertGreater(total[0], 0)
        self.assertTrue(np.all(np.diff(total) >= 0))

    def test_doppler_plot_titles_include_reference_width_in_both_units(self):
        line = LineModel(C_LIGHT / (460.8624e-7), 2e8, 1., 1., 1., 0., 87.6)
        width = 2.34e9
        stokes = np.zeros((2, 3, 3, 1, 1))
        stokes[:, 0] = 1.
        titles = []

        def capture(fig, *args, **kwargs):
            titles.append(fig._suptitle.get_text())

        with tempfile.TemporaryDirectory() as outdir, patch("matplotlib.figure.Figure.savefig", capture):
            for prefix in ("line", "continuum"):
                plot_spectrum_doppler(np.array([-width, 0., width]), stokes, 1., "unit",
                                      np.array([0.5]), width, line, outdir, prefix)
                plot_all_mu_spectra_doppler(np.array([-width, 0., width]), stokes, 1., "unit",
                                             np.array([0.5]), width, line, outdir, prefix)
        self.assertEqual(len(titles), 4)
        for title in titles:
            self.assertIn("0.0016578 nm", title)
            self.assertIn("2.34 GHz", title)


if __name__ == "__main__":
    unittest.main()
