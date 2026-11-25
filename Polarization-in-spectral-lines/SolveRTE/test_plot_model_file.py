"""Focused tests for quantities plotted from a model atmosphere."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import tempfile

import numpy as np

from atmosphere import C_LIGHT, K_B, LineModel
from grid_opacity import (H_PLANCK, eta0, planck, point_emission_and_propagation,
                          point_emission_and_propagation_unresolved)
from line_profiles import voigt_function
from plot_model_file import (DEFAULT_MODEL, build_grid, plot_all_mu_spectra_doppler,
                             plot_spectrum_doppler, radiation_anisotropy,
                             radiation_tensors_from_model, read_model, resolve_model,
                             stokes_percent)


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

    def test_radiation_tensor_columns_and_conjugate_symmetry(self):
        col = {"height_km": np.array([1., 0.]),
               "Jbar00": np.array([10., 20.]), "Jbar20": np.array([1., -2.]),
               "Jbar21_re": np.array([2., 3.]), "Jbar21_im": np.array([4., -5.]),
               "Jbar22_re": np.array([6., -7.]), "Jbar22_im": np.array([-8., 9.])}
        jkq = radiation_tensors_from_model(col)
        np.testing.assert_array_equal(jkq[:, 0, 2], col["Jbar00"] * 1e3)
        np.testing.assert_array_equal(jkq[:, 2, 2], col["Jbar20"] * 1e3)
        for q in (1, 2):
            positive = (col[f"Jbar2{q}_re"] + 1j * col[f"Jbar2{q}_im"]) * 1e3
            np.testing.assert_array_equal(jkq[:, 2, 2 + q], positive)
            np.testing.assert_array_equal(jkq[:, 2, 2 - q], (-1) ** q * np.conj(positive))
        np.testing.assert_array_equal(jkq[:, 1], 0.)
        partial = {name: col[name] for name in ("height_km", "Jbar00", "Jbar20", "Jbar21_im")}
        jkq = radiation_tensors_from_model(partial)
        np.testing.assert_array_equal(jkq[:, 2, 3], 1j * col["Jbar21_im"] * 1e3)
        np.testing.assert_array_equal(jkq[:, 2, 1], 1j * col["Jbar21_im"] * 1e3)
        np.testing.assert_array_equal(jkq[:, 2, (0, 4)], 0.)
        for name, value in (("Jbar21_re", np.array([1.])),
                            ("Jbar22_im", np.array([1., np.nan])),
                            ("Jbar21_real", np.ones(2))):
            with self.assertRaises(ValueError):
                radiation_tensors_from_model({**partial, name: value})

    def test_model_file_optional_tensor_columns_feed_grid(self):
        header = ("# height_km log_tau500 T_K Jbar00 Jbar20 n_lower C_ul epsilon "
                  "a_damp Jbar21_re Jbar21_im Jbar22_re Jbar22_im\n")
        rows = ("1 -1 5000 1e-8 1e-9 1e16 1e5 0.01 0.01 2e-9 -3e-9 4e-9 5e-9\n"
                "0 0 5000 2e-8 2e-9 1e16 1e5 0.01 0.01 -1e-9 2e-9 3e-9 -4e-9\n")
        with tempfile.TemporaryDirectory() as folder:
            path = f"{folder}/model.txt"
            with open(path, "w") as handle:
                handle.write(header + rows)
            col, _ = read_model(path)
            with open(path, "w") as handle:
                handle.write(header.rstrip() + " Jbar21_re\n" + rows)
            with self.assertRaisesRegex(ValueError, "column names must be unique"):
                read_model(path)
        line = LineModel(6.5e14, 2e8, 1., 1., 1., 0., 87.6)
        grid = build_grid(col, line, np.zeros(2), np.array([0.5]), np.array([0.]),
                          np.ones(1), np.zeros(2), 0.)
        np.testing.assert_allclose(grid.JKQ[:, 2, 3], [2e-6 - 3e-6j, -1e-6 + 2e-6j])
        np.testing.assert_allclose(grid.JKQ[:, 2, 0], [4e-6 - 5e-6j, 3e-6 + 4e-6j])

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

    def test_population_alignment_matches_two_level_pumping(self):
        line = LineModel(6.5e14, 2e8, 1., 1., 1., 0., 87.6)
        jkq = np.zeros((3, 5), complex)
        jkq[2, 2] = 1e-5
        mu, temperature, n_l, c_ul, d2 = 0.5, 100000., 1e10, 2e7, 3e7
        args = (jkq, line, np.array([mu]), np.array([0.]), np.array([0.]),
                temperature, n_l, 0., 0., 0., 0., 0., d2, c_ul)
        emis, _ = point_emission_and_propagation(*args, n_u=3e9, damping=0.01)
        width = line.doppler_width(temperature)
        expected = (eta0(line, n_l) / width * voigt_function(0., 0.01).real
                    * jkq[2, 2].real * line.A_ul / (line.A_ul + c_ul + d2)
                    * (-3 / (2 * np.sqrt(2)) * (1 - mu**2)))
        np.testing.assert_allclose(emis[1, 0, 0, 0], expected, rtol=1e-12)
        no_stim, _ = point_emission_and_propagation(*args, n_u=3e9, damping=0.01,
                                                    stimulated=False)
        np.testing.assert_allclose(emis[1:], no_stim[1:], rtol=1e-12, atol=1e-30)

    def test_population_magnetic_coefficients_and_unresolved_limits(self):
        line = LineModel(6.5e14, 2e8, 1., 1., 1., 0., 87.6)
        jkq = np.zeros((3, 5), complex)
        jkq[0, 2], jkq[2, 2] = 1e-5, 1e-6
        prefix = (jkq, line, np.array([0.5]), np.array([0.]),
                  np.linspace(-2e9, 2e9, 7), 6000., 1e10)
        suffix = (0., 0., 0., 1e6)
        def single(B, theta, chi, moments=jkq):
            return point_emission_and_propagation(
                moments, *prefix[1:], B, theta, chi, *suffix, n_u=3e8, damping=0.01)

        zero = single(0., 0., 0.)
        near = single(1e-7, 1.1, 0.7)
        for actual, expected in zip(near, zero):
            np.testing.assert_allclose(actual, expected, rtol=1e-7,
                                       atol=np.max(np.abs(expected)) * 1e-8)

        isotropic = jkq.copy()
        isotropic[2, 2] = 0.
        forward = single(1000., 0.8, 0.7, isotropic)
        reverse = single(1000., np.pi - 0.8, 0.7 + np.pi, isotropic)
        for a, b in zip(forward, reverse):
            np.testing.assert_allclose(a[:3], b[:3], rtol=1e-10,
                                       atol=np.max(np.abs(a)) * 1e-12)
            np.testing.assert_allclose(a[3], -b[3], rtol=1e-10,
                                       atol=np.max(np.abs(a)) * 1e-12)
        self.assertGreater(np.max(np.abs(forward[0][3])), 0.)

        direct = single(50., 1.1, 0.7)
        averaged = point_emission_and_propagation_unresolved(
            *prefix, 50., 1.1, 0.7, 0., *suffix, n_u=3e8, damping=0.01,
            n_theta=4, n_chi=6)
        for actual, expected in zip(averaged, direct):
            np.testing.assert_allclose(actual, expected, rtol=1e-12,
                                       atol=np.max(np.abs(expected)) * 1e-13)
        random = point_emission_and_propagation_unresolved(
            *prefix, 0., 0., 0., 50., *suffix, n_u=3e8, damping=0.01,
            n_theta=4, n_chi=8)
        for coeff in random:
            self.assertLess(np.max(np.abs(coeff[3])), np.max(np.abs(coeff[0])) * 1e-12)

    def test_population_lte_kirchhoff_for_zeeman_components(self):
        line = LineModel(6.5e14, 2e8, 1., 1., 1., 0., 87.6)
        temperature, n_l = 6000., 1e10
        n_u = 3 * n_l * np.exp(-H_PLANCK * line.nu0 / (K_B * temperature))
        emis, eta = point_emission_and_propagation(
            np.zeros((3, 5), complex), line, np.array([0.6]), np.array([0.]),
            np.linspace(-2e9, 2e9, 7), temperature, n_l, 1000., 0.8, 0.7,
            0., 0., 0., 0., n_u=n_u, damping=0.01)
        np.testing.assert_allclose(emis, eta[:4] * planck(line.nu0, temperature),
                                   rtol=1e-12, atol=np.max(np.abs(emis)) * 1e-13)

    def test_updated_model_columns_and_top_offset(self):
        col, _ = read_model(resolve_model(DEFAULT_MODEL))
        self.assertIn("n_upper", col)
        self.assertIn("log_tau_line", col)
        total = 10 ** col["log_tau_line"]
        self.assertGreater(total[0], 0)
        self.assertTrue(np.all(np.diff(total) >= 0))

    def test_doppler_plot_titles_omit_reference_width(self):
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
            self.assertIn("Doppler", title)
            self.assertNotIn("Reference Doppler width", title)
            self.assertNotIn("0.0016578 nm", title)
            self.assertNotIn("2.34 GHz", title)


if __name__ == "__main__":
    unittest.main()
