import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atmosphere import C_LIGHT
from grid_opacity import H_PLANCK, eta0, fill_grid, planck
from line_profiles import voigt_function
from plot_model_file import (
    DEFAULT_MODEL, build_grid, cumulative_tau, read_model,
    resolve_model, transfer,
)
from plot_physical_flow import (continuum_from_line_center, model_parameters, physical_flow,
                                plot_spectra, wavelength_grid)


class PhysicalFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.col, header = read_model(resolve_model(DEFAULT_MODEL))
        cls.line, cls.v_turb = model_parameters(cls.col, header, 87.62)
        cls.wavelength, cls.width = wavelength_grid(
            cls.line, cls.line.doppler_width(cls.col["T_K"], cls.v_turb), 31, 8, None)
        cls.mu = np.array([0.5, 1.0])

    def test_static_flow_matches_tabulated_depth_and_source(self):
        nu, x, source, tau, stokes = physical_flow(
            self.col, self.line, self.v_turb, self.wavelength, self.mu,
            continuum=True, gamma=np.pi / 2)
        chi_c = continuum_from_line_center(self.col, self.line, self.v_turb)
        grid = build_grid(self.col, self.line, self.v_turb, self.mu, nu,
                          np.ones_like(nu), chi_c, np.pi / 2, use_populations=True)
        fill_grid(grid, gamma=np.pi / 2, wavelength_profile=True)
        expected_source = grid.emis[:, :3] / grid.eta[:, 0, None]
        dz = -np.diff(grid.atm.z)
        opacity = grid.eta[:, 0, :, 0, 0]
        expected_tau = cumulative_tau(opacity, dz) + opacity[0] * dz[0] / 2
        expected_stokes = transfer(expected_source, expected_tau, self.mu, self.line,
                                   grid.atm.T[-1],
                                   planck(C_LIGHT / self.wavelength, grid.atm.T[-1])[:, None, None],
                                   "sc")
        self.assertEqual(x.shape, (len(grid.atm.z), 31, 2, 1))
        np.testing.assert_allclose(source, expected_source, rtol=1e-12)
        np.testing.assert_allclose(tau[:, :, 0, 0], expected_tau, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(tau[:, :, 1, 0], expected_tau, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(stokes, expected_stokes, rtol=1e-11, atol=1e-20)
        center = len(nu) // 2
        np.testing.assert_allclose(tau[:, center, :, 0],
                                   np.broadcast_to((10 ** self.col["log_tau_line"])[:, None],
                                                   (len(dz) + 1, len(self.mu))), rtol=1e-9)
        self.assertGreater(np.ptp(stokes[0, 0, :, 0, 0]), 0.01 * stokes[0, 0, 0, 0, 0])

    def test_near_limb_ray_preserves_existing_ray_solutions(self):
        wavelength = self.wavelength[::10]
        for solver in ("sc", "delo"):
            combined = physical_flow(self.col, self.line, self.v_turb, wavelength,
                                     [0.01, 0.05, 0.1], continuum=False, solver=solver)[-1]
            previous = physical_flow(self.col, self.line, self.v_turb, wavelength,
                                     [0.1], continuum=False, solver=solver)[-1]
            self.assertTrue(np.all(np.isfinite(combined)))
            self.assertTrue(np.all(combined[0, 0] > 0))
            np.testing.assert_allclose(combined[:, :, :, 2, :], previous[:, :, :, 0, :],
                                       rtol=1e-12, atol=1e-20)

    def test_projected_velocity_changes_profile_and_ray_optical_depth(self):
        velocity = 3e5
        nu, x, _, tau, stokes = physical_flow(
            self.col, self.line, self.v_turb, self.wavelength, self.mu, v_z=velocity,
            continuum=False)
        width = self.line.doppler_width(self.col["T_K"], self.v_turb)
        delta_lambda = self.line.lambda0 * width / self.line.nu0
        expected = (self.wavelength[None, :, None] - self.line.lambda0
                    * (1 - velocity * self.mu / C_LIGHT))
        expected = expected / delta_lambda[:, None, None]
        np.testing.assert_allclose(x[..., 0], expected)
        amplitude = eta0(self.line, (self.col["n_lower"] - self.col["n_upper"] / 3) * 1e-6) / width
        opacity = amplitude[:, None, None] * voigt_function(
            expected, self.col["a_damp"][:, None, None]).real
        dz = -np.diff(self.col["height_km"] * 1e5)
        for k in range(len(self.mu)):
            expected_tau = cumulative_tau(opacity[:, :, k], dz) + opacity[0, :, k] * dz[0] / 2
            np.testing.assert_allclose(tau[:, :, k, 0], expected_tau, rtol=2e-10, atol=1e-14)
        self.assertFalse(np.allclose(tau[-1, :, 0, 0], tau[-1, :, 1, 0]))
        np.testing.assert_allclose(
            stokes[-1, 0, :, :, 0],
            np.broadcast_to(planck(C_LIGHT / self.wavelength, self.col["T_K"][-1])[:, None],
                            (len(self.wavelength), len(self.mu))))
        np.testing.assert_array_equal(stokes[-1, 1:, :, :, 0], 0)

    def test_isotropic_source_matches_line_plus_continuum_formula(self):
        # With Jbar20 = 0 only the isotropic part remains, so
        # S = (chi_L S_L + chi_c B_lambda) / (chi_L + chi_c) must hold exactly.
        col = dict(self.col)
        col["Jbar20"] = np.zeros_like(col["Jbar20"])
        _, x, source, _, _ = physical_flow(
            col, self.line, self.v_turb, self.wavelength, self.mu, continuum=True)
        n_l, n_u = col["n_lower"] * 1e-6, col["n_upper"] * 1e-6
        width = self.line.doppler_width(col["T_K"], self.v_turb)
        chi_line = (eta0(self.line, n_l - n_u / 3) / width)[:, None, None] * voigt_function(
            x[..., 0], col["a_damp"][:, None, None]).real
        two_h_nu3_c2 = 2 * H_PLANCK * self.line.nu0**3 / C_LIGHT**2
        source_line = (two_h_nu3_c2 / (3 * n_l / n_u - 1))[:, None, None]
        chi_c = continuum_from_line_center(col, self.line, self.v_turb)[:, None, None]
        b_lambda = planck(C_LIGHT / self.wavelength, col["T_K"][:, None])[:, :, None]
        expected = (chi_line * source_line + chi_c * b_lambda) / (chi_line + chi_c)
        np.testing.assert_allclose(source[:, 0, :, :, 0], expected, rtol=1e-10)
        np.testing.assert_allclose(source[:, 1:], 0, atol=1e-30)

    def test_inconsistent_line_center_depth_is_rejected(self):
        col = dict(self.col)
        col["log_tau_line"] = np.full_like(col["log_tau_line"], -9.)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            continuum_from_line_center(col, self.line, self.v_turb)
        col["log_tau_line"] = self.col["log_tau_line"] - 6.
        with self.assertRaisesRegex(ValueError, "negative or nonfinite continuum"):
            continuum_from_line_center(col, self.line, self.v_turb)

    def test_delo_zero_field_matches_linear_scalar_transfer(self):
        nu, _, source, tau, scalar = physical_flow(
            self.col, self.line, self.v_turb, self.wavelength, self.mu,
            continuum=True, gamma=np.pi / 2, solver="sc")
        _, _, _, _, delo = physical_flow(
            self.col, self.line, self.v_turb, self.wavelength, self.mu,
            continuum=True, gamma=np.pi / 2, solver="delo")
        grid = build_grid(
            self.col, self.line, self.v_turb, self.mu, nu, np.ones_like(nu),
            continuum_from_line_center(self.col, self.line, self.v_turb), np.pi / 2,
            use_populations=True)
        fill_grid(grid, gamma=np.pi / 2, wavelength_profile=True)
        scalar_source = grid.emis[:, :3] / grid.eta[:, 0, None]
        dz = -np.diff(grid.atm.z)
        scalar_tau = cumulative_tau(grid.eta[:, 0, :, 0, 0], dz)
        scalar_tau += grid.eta[0, 0, :, 0, 0][None, :] * dz[0] / 2
        top = np.zeros((3, len(nu), len(self.mu), 1))
        bottom = np.zeros_like(top)
        bottom[0] = planck(C_LIGHT / self.wavelength, grid.atm.T[-1])[:, None, None]
        linear = transfer(scalar_source, scalar_tau, self.mu, self.line,
                          grid.atm.T[-1], bottom[0], "fs")
        self.assertEqual(delo.shape[1], 4)
        np.testing.assert_allclose(delo[:, :3], linear, rtol=2e-12, atol=1e-20)
        np.testing.assert_array_equal(delo[:, 3], 0.0)
        self.assertEqual(scalar.shape[1], 3)

    def test_magnetic_delo_spectra_and_depth_dependent_fields(self):
        wavelength = self.wavelength[::3]
        n = len(self.col["height_km"])
        zero = physical_flow(self.col, self.line, self.v_turb, wavelength, self.mu,
                             solver="delo")
        field = np.tile([40., 25., 30.], (n, 1))
        magnetic = physical_flow(self.col, self.line, self.v_turb, wavelength, self.mu,
                                 solver="delo", B=field, D2=np.zeros(n))
        emergent = magnetic[-1][0, :, :, 0, 0]
        self.assertTrue(np.all(np.isfinite(emergent)))
        self.assertGreater(np.max(np.abs(emergent[2])), 1e-12 * np.max(emergent[0]))
        self.assertGreater(np.max(np.abs(emergent[3])), 1e-12 * np.max(emergent[0]))
        self.assertGreater(np.max(np.abs(emergent[1] - zero[-1][0, 1, :, 0, 0])),
                           1e-8 * np.max(emergent[0]))
        constant = physical_flow(self.col, self.line, self.v_turb, wavelength, self.mu,
                                 solver="delo", B=[40., 25., 30.], D2=0.)
        np.testing.assert_allclose(magnetic[-1], constant[-1], rtol=1e-12, atol=1e-30)
        with self.assertRaisesRegex(ValueError, "require solver"):
            physical_flow(self.col, self.line, self.v_turb, wavelength, self.mu, B=field)
        with self.assertRaisesRegex(ValueError, "D2"):
            physical_flow(self.col, self.line, self.v_turb, wavelength, self.mu, D2=-1.)

    def test_each_optional_tensor_component_affects_emergent_polarization(self):
        wavelength = self.wavelength[::5]
        _, _, _, tau, stokes = physical_flow(
            self.col, self.line, self.v_turb, wavelength, [0.5], solver="delo")
        for name in ("Jbar21_re", "Jbar21_im", "Jbar22_re", "Jbar22_im"):
            col = dict(self.col)
            col[name] = self.col["Jbar00"] * 0.02
            _, _, _, changed_tau, changed = physical_flow(
                col, self.line, self.v_turb, wavelength, [0.5], solver="delo")
            np.testing.assert_allclose(changed_tau, tau, rtol=1e-12)
            difference = np.max(np.abs(changed[0, 1:3] - stokes[0, 1:3]))
            self.assertGreater(difference, 1e-7 * np.max(stokes[0, 0]), name)

    def test_unresolved_field_and_depolarization_flow(self):
        col = {key: value[::20] for key, value in self.col.items()}
        v_turb = self.v_turb[::20]
        wavelength = self.wavelength[::5]
        zero = physical_flow(col, self.line, v_turb, wavelength, [0.5],
                             solver="delo", continuum=False)[-1]
        damped = physical_flow(col, self.line, v_turb, wavelength, [0.5],
                               D2=10 * self.line.A_ul, solver="delo", continuum=False)[-1]
        random = physical_flow(col, self.line, v_turb, wavelength, [0.5], B_unres=50.,
                               n_theta=4, n_chi=8, solver="delo", continuum=False)[-1]
        self.assertLess(np.max(np.abs(damped[0, 1])), np.max(np.abs(zero[0, 1])) / 10)
        self.assertTrue(np.all(np.isfinite(random)))
        self.assertLess(np.max(np.abs(random[0, 3])), np.max(np.abs(random[0, 0])) * 1e-12)
        self.assertGreater(np.max(np.abs(random[0, 1] - zero[0, 1])), 0.)
        with self.assertRaisesRegex(ValueError, "require solver"):
            physical_flow(col, self.line, v_turb, wavelength, [0.5], B_unres=1.)
        with self.assertRaisesRegex(ValueError, "B_unres"):
            physical_flow(col, self.line, v_turb, wavelength, [0.5],
                          B_unres=-1., solver="delo", continuum=False)

    def test_inferred_continuum_tracks_tabulated_depth(self):
        baseline = continuum_from_line_center(self.col, self.line, self.v_turb)
        dz = -np.diff(self.col["height_km"] * 1e5)
        extra_chi = 1e-6
        extra_tau = extra_chi * (dz[0] / 2 + np.r_[0., np.cumsum(dz)])
        col = dict(self.col)
        col["log_tau_line"] = np.log10(10 ** self.col["log_tau_line"] + extra_tau)
        inferred = continuum_from_line_center(col, self.line, self.v_turb)
        np.testing.assert_allclose(inferred, baseline + extra_chi, rtol=1e-10, atol=1e-16)

    def test_wavelength_intensity_conversion_and_emergent_boundary(self):
        wavelength = np.array([4.5e-5, 4.6e-5, 4.7e-5])
        nu = C_LIGHT / wavelength - self.line.nu0
        stokes = np.zeros((2, 3, 3, 1, 1))
        stokes[0, :, :, 0, 0] = [[2, 4, 8], [0.2, 0.8, 2.4], [-0.4, 0.4, 1.6]]
        stokes[1, 0, :, 0, 0] = 100
        figures = []
        titles = []
        paths = []
        coordinates = []

        def capture(fig, path, *args, **kwargs):
            figures.append([[curve.get_ydata().copy() for curve in ax.lines] for ax in fig.axes])
            titles.append(fig._suptitle.get_text())
            paths.append(str(path))
            coordinates.append((fig.axes[0].get_xlabel(), fig.axes[0].lines[0].get_xdata().copy()))

        with tempfile.TemporaryDirectory() as outdir, patch("matplotlib.figure.Figure.savefig", capture):
            plot_spectra(wavelength, nu, stokes, [0.5], self.line, self.width, outdir, False,
                         field_label="B=(40, 25, 30) G, D2=1e7 s^-1", B=[40., 25., 30.])
        expected = stokes[0, 0, :, 0, 0] / 1e3 * C_LIGHT / wavelength**2 * 1e-7
        np.testing.assert_allclose(figures[0][0][0], expected)
        np.testing.assert_allclose(figures[0][1][0], [10, 20, 30])
        np.testing.assert_allclose(figures[0][2][0], [-20, 10, 20])
        np.testing.assert_allclose(figures[1][0][0], [8e-3, 4e-3, 2e-3])
        self.assertEqual(len(figures), 2)
        self.assertEqual(coordinates[0][0], "Wavelength [nm]")
        np.testing.assert_allclose(coordinates[0][1], wavelength * 1e7, rtol=0, atol=1e-10)
        self.assertEqual(coordinates[1][0], "Frequency [GHz]")
        np.testing.assert_allclose(coordinates[1][1], np.sort(C_LIGHT / wavelength / 1e9),
                                   rtol=0, atol=1e-8)
        for title in titles:
            self.assertNotIn(r"\Delta", title)
            self.assertNotIn("Doppler", title)
            self.assertIn("B=(40, 25, 30) G, D2=1e7 s^-1", title)
            self.assertIn("|B|=55.902 G, inclination=57.543692 deg, azimuth=32.005383 deg", title)
        for path in paths:
            self.assertTrue(path.endswith("_B55.901699G_inc57.543692deg_azi32.005383deg.png"))

    def test_plot_field_angles_when_direction_is_undefined(self):
        wavelength = self.wavelength[::15]
        nu = C_LIGHT / wavelength - self.line.nu0
        stokes = np.zeros((2, 4, len(wavelength), 1, 1))
        stokes[:, 0] = 1.
        with tempfile.TemporaryDirectory() as outdir, patch("matplotlib.figure.Figure.savefig") as save:
            for field in ([0., 0., 0.], [0., 0., -10.]):
                plot_spectra(wavelength, nu, stokes, [0.5], self.line, self.width,
                             outdir, False, B=field)
            paths = [str(call.args[0]) for call in save.call_args_list]
        self.assertIn("_B0.000000G_incNAdeg_aziNAdeg.png", paths[0])
        self.assertIn("_B10.000000G_inc180.000000deg_aziNAdeg.png", paths[2])

    def test_tiny_v_percent_is_zero_only_in_plots(self):
        wavelength = self.wavelength[::5]
        nu = C_LIGHT / wavelength - self.line.nu0
        stokes = np.zeros((1, 4, len(wavelength), 1, 1))
        stokes[0, 0, :, 0, 0] = 1.
        original_v_percent = np.array([0.5e-15, -0.5e-15, 2e-15, -2e-15, 0., 0., 0.])
        stokes[0, 3, :, 0, 0] = original_v_percent / 100
        original = stokes.copy()
        curves = []

        def capture(fig, *args, **kwargs):
            curves.append(fig.axes[3].lines[0].get_ydata().copy())

        with tempfile.TemporaryDirectory() as outdir, patch("matplotlib.figure.Figure.savefig", capture):
            plot_spectra(wavelength, nu, stokes, [0.05], self.line, self.width, outdir, False)
        expected = np.array([0., 0., 2e-15, -2e-15, 0., 0., 0.])
        np.testing.assert_allclose(curves[0], expected, rtol=1e-12, atol=0)
        np.testing.assert_allclose(curves[1], expected[::-1], rtol=1e-12, atol=0)
        np.testing.assert_array_equal(stokes, original)

    def test_plot_spectral_axis_selection(self):
        wavelength = self.wavelength[::15]
        nu = C_LIGHT / wavelength - self.line.nu0
        stokes = np.zeros((2, 4, len(wavelength), 1, 1))
        stokes[:, 0] = 1.
        with tempfile.TemporaryDirectory() as outdir, patch("matplotlib.figure.Figure.savefig") as save:
            for axis in ("frequency", "wavelength"):
                plot_spectra(wavelength, nu, stokes, [0.5], self.line, self.width,
                             outdir, False, spectral_axis=axis)
            paths = [str(call.args[0]) for call in save.call_args_list]
            self.assertEqual(len(paths), 2)
            self.assertTrue(paths[0].endswith("_frequency.png"))
            self.assertTrue(paths[1].endswith("_wavelength.png"))
            with self.assertRaisesRegex(ValueError, "spectral_axis"):
                plot_spectra(wavelength, nu, stokes, [0.5], self.line, self.width,
                             outdir, False, spectral_axis="doppler")

    def test_spectral_tick_labels_do_not_overlap_on_model_grid(self):
        wavelength = self.wavelength
        nu = C_LIGHT / wavelength - self.line.nu0
        stokes = np.ones((1, 4, len(wavelength), 1, 1))
        captured = []

        def capture(fig, *args, **kwargs):
            fig.canvas.draw()
            captured.append(fig)
            self.assertEqual(tuple(fig.get_size_inches()), (19, 5.2))
            for ax in fig.axes:
                formatter = ax.xaxis.get_major_formatter()
                self.assertAlmostEqual(formatter.offset,
                                       650500 if ax.get_xlabel() == "Frequency [GHz]" else 460.8)
                self.assertTrue(ax.xaxis.get_offset_text().get_text().startswith("+"))
                labels = [label for label in ax.get_xticklabels() if label.get_visible()]
                self.assertTrue(all(label.get_rotation() == 0 for label in labels))
                boxes = [label.get_window_extent() for label in labels]
                self.assertGreater(len(boxes), 4)
                self.assertFalse(any(a.overlaps(b) for a, b in zip(boxes, boxes[1:])))

        with tempfile.TemporaryDirectory() as outdir, patch("matplotlib.figure.Figure.savefig", capture):
            plot_spectra(wavelength, nu, stokes, [0.5], self.line, self.width, outdir, False)
        self.assertEqual(len(captured), 2)

    def test_wavelength_grid_rejects_bad_limits(self):
        with self.assertRaisesRegex(ValueError, "wavelength limits"):
            wavelength_grid(self.line, np.array([1e9]), 5, 5, [500, 400])


if __name__ == "__main__":
    unittest.main()
