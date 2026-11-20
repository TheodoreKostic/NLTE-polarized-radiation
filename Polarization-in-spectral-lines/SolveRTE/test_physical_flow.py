import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atmosphere import C_LIGHT
from grid_opacity import eta0, planck, stimulated_factor
from line_profiles import voigt_function
from plot_model_file import (
    DEFAULT_MODEL, build_grid, continuum_from_tau500, cumulative_tau, read_model,
    resolve_model, transfer,
)
from plot_physical_flow import model_parameters, physical_flow, plot_spectra, wavelength_grid


class PhysicalFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.col, header = read_model(resolve_model(DEFAULT_MODEL))
        cls.line, cls.v_turb = model_parameters(cls.col, header, 87.62)
        cls.wavelength, cls.width = wavelength_grid(
            cls.line, cls.line.doppler_width(cls.col["T_K"], cls.v_turb), 31, 8, None)
        cls.mu = np.array([0.5, 1.0])

    def test_static_flow_matches_existing_pipeline(self):
        nu, x, source, tau, stokes = physical_flow(
            self.col, self.line, self.v_turb, self.wavelength, self.mu,
            continuum=True, gamma=np.pi / 2)
        _, chi_c = continuum_from_tau500(self.col)
        grid = build_grid(self.col, self.line, self.v_turb, self.mu, nu,
                          np.ones_like(nu), chi_c, np.pi / 2)
        expected_source = grid.emis[:, :3] / grid.eta[:, 0, None]
        line_grid = build_grid(self.col, self.line, self.v_turb, self.mu, nu,
                               np.ones_like(nu), np.zeros_like(chi_c), np.pi / 2)
        expected_tau = cumulative_tau(line_grid.eta[:, 0, :, 0, 0],
                                      -np.diff(grid.atm.z))
        expected_tau += continuum_from_tau500(self.col)[0][:, None]
        expected_stokes = transfer(expected_source, expected_tau, self.mu, self.line,
                                   grid.atm.T[-1], None, "sc")
        self.assertEqual(x.shape, (len(grid.atm.z), 31, 2, 1))
        np.testing.assert_allclose(source, expected_source, rtol=1e-12)
        np.testing.assert_allclose(tau[:, :, 0, 0], expected_tau, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(tau[:, :, 1, 0], expected_tau, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(stokes, expected_stokes, rtol=1e-11, atol=1e-20)
        self.assertGreater(np.ptp(stokes[0, 0, :, 0, 0]), 0.01 * stokes[0, 0, 0, 0, 0])

    def test_projected_velocity_changes_profile_and_ray_optical_depth(self):
        velocity = 3e5
        nu, x, _, tau, stokes = physical_flow(
            self.col, self.line, self.v_turb, self.wavelength, self.mu, v_z=velocity)
        width = self.line.doppler_width(self.col["T_K"], self.v_turb)
        expected = (-nu[None, :, None] + self.line.nu0 * velocity * self.mu / C_LIGHT)
        expected = expected / width[:, None, None]
        np.testing.assert_allclose(x[..., 0], expected)
        amplitude = (eta0(self.line, self.col["n_lower"] * 1e-6)
                     * stimulated_factor(self.line.nu0, self.col["T_K"]) / width)
        opacity = amplitude[:, None, None] * voigt_function(
            expected, self.col["a_damp"][:, None, None]).real
        dz = -np.diff(self.col["height_km"] * 1e5)
        for k in range(len(self.mu)):
            expected_tau = cumulative_tau(opacity[:, :, k], dz)
            np.testing.assert_allclose(tau[:, :, k, 0], expected_tau, rtol=1e-12, atol=1e-14)
        self.assertFalse(np.allclose(tau[-1, :, 0, 0], tau[-1, :, 1, 0]))
        np.testing.assert_allclose(stokes[-1, 0, :, :, 0],
                                   planck(self.line.nu0, self.col["T_K"][-1]))
        np.testing.assert_array_equal(stokes[-1, 1:, :, :, 0], 0)

    def test_wavelength_intensity_conversion_and_emergent_boundary(self):
        wavelength = np.array([4.5e-5, 4.6e-5, 4.7e-5])
        nu = C_LIGHT / wavelength - self.line.nu0
        stokes = np.zeros((2, 3, 3, 1, 1))
        stokes[0, :, :, 0, 0] = [[2, 4, 8], [0.2, 0.8, 2.4], [-0.4, 0.4, 1.6]]
        stokes[1, 0, :, 0, 0] = 100
        figures = []
        titles = []

        def capture(fig, *args, **kwargs):
            figures.append([[curve.get_ydata().copy() for curve in ax.lines] for ax in fig.axes])
            titles.append(fig._suptitle.get_text())

        with tempfile.TemporaryDirectory() as outdir, patch("matplotlib.figure.Figure.savefig", capture):
            plot_spectra(wavelength, nu, stokes, [0.5], self.line, self.width, outdir, False)
        expected = stokes[0, 0, :, 0, 0] / 1e3 * C_LIGHT / wavelength**2 * 1e-7
        np.testing.assert_allclose(figures[0][0][0], expected)
        np.testing.assert_allclose(figures[0][1][0], [10, 20, 30])
        np.testing.assert_allclose(figures[0][2][0], [-20, 10, 20])
        np.testing.assert_allclose(figures[1][0][0], [8e-3, 4e-3, 2e-3])
        delta_nu_ref = self.line.nu0 * self.width / self.line.lambda0
        for title in titles:
            self.assertIn(f"{self.width * 1e7:.5g}", title)
            self.assertIn(f"{delta_nu_ref / 1e9:.5g}", title)
            self.assertIn("nm", title)
            self.assertIn("GHz", title)

    def test_wavelength_grid_rejects_bad_limits(self):
        with self.assertRaisesRegex(ValueError, "wavelength limits"):
            wavelength_grid(self.line, np.array([1e9]), 5, 5, [500, 400])


if __name__ == "__main__":
    unittest.main()
