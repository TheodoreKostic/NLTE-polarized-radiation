import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import plot_model_file as plots
from formal_solver import short_characteristics, solve_short_characteristics, solve_unpolarized_transfer


class TransferTests(unittest.TestCase):
    def test_sc_wrapper_uses_scalar_solver_for_both_directions(self):
        tau = np.array([0.0, 0.0001, 0.1, 0.7, 3.0])[:, None]
        mu = np.array([-0.5, 0.7])
        source = np.empty((5, 3, 1, 2, 1))
        source[:, :, 0, :, 0] = np.array([1.0, 1.2, 2.0, 0.9, 0.7])[:, None, None]
        top = np.full((3, 1, 2, 1), 0.3)
        bottom = np.full_like(top, 2.0)
        result = solve_short_characteristics(source, tau, mu, top, bottom)
        for k, ray in enumerate(mu):
            boundary = top[0, 0, k, 0] if ray < 0 else bottom[0, 0, k, 0]
            expected, _ = short_characteristics(tau[:, 0], source[:, 0, 0, k, 0], ray, boundary)
            np.testing.assert_allclose(result[:, 0, 0, k, 0], expected, rtol=1e-13)

    def test_sc_matches_independent_linear_source_solver(self):
        tau = np.array([0.0, 0.0001, 0.1, 0.7, 3.0])[:, None]
        mu = np.array([-0.5, 0.7])
        source = np.broadcast_to(1 + 0.2 * tau[:, None, :, None, None], (5, 3, 1, 2, 1))
        top = np.full((3, 1, 2, 1), 0.3)
        bottom = np.full_like(top, 2.0)
        np.testing.assert_allclose(
            solve_short_characteristics(source, tau, mu, top, bottom),
            solve_unpolarized_transfer(source, tau, mu, top, bottom),
            rtol=1e-12,
        )

    def test_constant_source_and_boundaries(self):
        tau = np.array([0.0, 0.001, 0.2, 1.0])[:, None]
        mu = np.array([-0.5, 0.5])
        source = np.full((4, 3, 1, 2, 1), 2.0)
        top, bottom = np.ones((3, 1, 2, 1)), np.full((3, 1, 2, 1), 3.0)
        for solver in (solve_short_characteristics, solve_unpolarized_transfer):
            result = solver(source, tau, mu, top, bottom)
            expected_down = 2.0 - np.exp(-tau[:, 0] / 0.5)
            expected_up = 2.0 + np.exp(-(1.0 - tau[:, 0]) / 0.5)
            np.testing.assert_allclose(result[:, 0, 0, 0, 0], expected_down, rtol=1e-10)
            np.testing.assert_allclose(result[:, 0, 0, 1, 0], expected_up, rtol=1e-10)


class SpectrumPlotTests(unittest.TestCase):
    def test_coordinates_and_emergent_layer(self):
        nu = np.array([-2e9, 0.0, 2e9])
        stokes = np.zeros((2, 3, 3, 1, 1))
        stokes[0, 0, :, 0, 0] = [1, 2, 3]
        stokes[0, 1, :, 0, 0] = [0.1, 0.4, 0.9]
        stokes[0, 2, :, 0, 0] = [-0.2, 0.2, 0.6]
        stokes[-1, 0, :, 0, 0] = 100
        line = SimpleNamespace(nu0=6.5e14)
        captured = []

        def capture(fig, *args, **kwargs):
            captured.append([(curve.get_xdata().copy(), curve.get_ydata().copy())
                             for ax in fig.axes for curve in ax.lines])

        with tempfile.TemporaryDirectory() as outdir, patch("matplotlib.figure.Figure.savefig", capture):
            plots.plot_spectrum_frequency(nu, stokes, 1, "unit", [0.5], outdir)
            plots.plot_spectrum_normalized(nu, stokes, 1, "unit", [0.5], outdir)
            plots.plot_spectrum_wavelength(nu, stokes, 1, "unit", [0.5], line, outdir)
            plots.plot_spectrum_doppler(nu, stokes, 1, "unit", [0.5], 2e9, line, outdir)
            plots.plot_all_mu_spectra_frequency(nu, stokes, 1, "unit", [0.5], outdir)
            plots.plot_all_mu_spectra_doppler(nu, stokes, 1, "unit", [0.5], 2e9, line, outdir)

        np.testing.assert_allclose(captured[0][0][0], [-2, 0, 2])
        np.testing.assert_allclose(captured[0][0][1], [1, 2, 3])
        np.testing.assert_allclose(captured[1][0][1], [10, 20, 30])
        for idx in (0, 3, 4, 5):
            np.testing.assert_allclose(captured[idx][1][1], [10, 20, 30])
            np.testing.assert_allclose(captured[idx][2][1], [-20, 10, 20])
        np.testing.assert_allclose(captured[1][1][1], [-20, 10, 20])
        np.testing.assert_allclose(captured[2][0][0], (plots.C_LIGHT / (line.nu0 + nu) * 1e7)[::-1])
        np.testing.assert_allclose(captured[2][0][1], [3, 2, 1])
        np.testing.assert_allclose(captured[2][1][1], [30, 20, 10])
        np.testing.assert_allclose(captured[2][2][1], [20, 10, -20])
        for idx in (3, 5):
            np.testing.assert_allclose(captured[idx][0][0], [-1, 0, 1])
            np.testing.assert_allclose(captured[idx][0][1], [1, 2, 3])
        np.testing.assert_allclose(captured[4][0][1], [1, 2, 3])

    def test_all_spectra_mask_invalid_intensity_and_label_ratios(self):
        nu = np.array([-1e9, 0.0, 1e9])
        stokes = np.zeros((2, 3, 3, 2, 1))
        stokes[0, 0, :, :, 0] = [[2, 4], [0, -1], [4, 2]]
        stokes[0, 1, :, :, 0] = [[1, 1], [1, 1], [1, 1]]
        stokes[0, 2, :, :, 0] = [[-0.5, 0.5], [1, 1], [2, -1]]
        line = SimpleNamespace(nu0=6.5e14)
        figures = []

        def capture(fig, *args, **kwargs):
            figures.append([(ax.get_ylabel(), [curve.get_ydata().copy() for curve in ax.lines])
                            for ax in fig.axes])

        with tempfile.TemporaryDirectory() as outdir, patch("matplotlib.figure.Figure.savefig", capture):
            calls = [
                lambda: plots.plot_spectrum_frequency(nu, stokes, 2, "unit", [0.5, 1.0], outdir),
                lambda: plots.plot_spectrum_normalized(nu, stokes, 2, "unit", [0.5, 1.0], outdir),
                lambda: plots.plot_spectrum_wavelength(nu, stokes, 2, "unit", [0.5, 1.0], line, outdir),
                lambda: plots.plot_spectrum_doppler(nu, stokes, 2, "unit", [0.5, 1.0], 1e9, line, outdir),
                lambda: plots.plot_all_mu_spectra_frequency(nu, stokes, 2, "unit", [0.5, 1.0], outdir),
                lambda: plots.plot_all_mu_spectra_doppler(nu, stokes, 2, "unit", [0.5, 1.0], 1e9, line, outdir),
            ]
            for call in calls:
                with self.assertWarnsRegex(RuntimeWarning, "non-positive I"):
                    call()

        for fig in figures:
            ratios = fig if len(fig) == 2 else fig[1:]
            self.assertEqual([axis[0] for axis in ratios], ["Q/I [%]", "U/I [%]"])
            for _, curves in ratios:
                for curve in curves:
                    self.assertTrue(np.isnan(curve[1]))
        for fig in (figures[-2], figures[-1]):
            np.testing.assert_allclose(fig[0][1][0], [4, 0, 8])
            np.testing.assert_allclose(fig[1][1][0], [50, np.nan, 25], equal_nan=True)
            np.testing.assert_allclose(fig[1][1][1], [25, np.nan, 50], equal_nan=True)
            np.testing.assert_allclose(fig[2][1][1], [12.5, np.nan, -50], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
