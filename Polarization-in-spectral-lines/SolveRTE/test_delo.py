import unittest

import numpy as np

from formal_solver import (
    _delo_weights, delo_polarized_ray, propagate_constant_slab, solve_delo,
    solve_unpolarized_transfer,
)

# eta_I > |(eta_Q, eta_U, eta_V)| as for a physical medium; every coefficient is non-zero
ETA = np.array([1.0, 0.1, -0.05, 0.2, 0.3, -0.2, 0.4])  # cm^-1
EMIS = np.array([0.7, 0.05, 0.02, -0.03])
INCIDENT = np.array([1.0, 0.2, -0.1, 0.05])


def homogeneous_error(n_layers, mu, thickness=2.0):
    z = np.linspace(thickness, 0.0, n_layers)  # cm, index 0 is the top
    eta = np.tile(ETA, (n_layers, 1))
    emis = np.tile(EMIS, (n_layers, 1))
    result = delo_polarized_ray(eta, emis, -np.diff(z), mu, INCIDENT)
    emergent = result[0] if mu > 0 else result[-1]
    exact = propagate_constant_slab(EMIS, ETA, thickness / abs(mu), INCIDENT)
    return np.max(np.abs(emergent - exact))


class DeloTests(unittest.TestCase):
    def test_homogeneous_slab_converges_to_exact_solution_in_both_directions(self):
        for mu in (0.5, -0.5, 1.0, -0.2):
            coarse, fine = homogeneous_error(101, mu), homogeneous_error(201, mu)
            self.assertLess(homogeneous_error(401, mu), 2e-3)
            self.assertGreater(coarse / fine, 3.0)  # second order: about 4 per doubling

    def test_zero_coefficients_match_linear_source_scalar_solver(self):
        rng = np.random.default_rng(3)
        nd = 30
        z = np.cumsum(rng.uniform(0.05, 0.4, nd))[::-1].copy()
        eta = np.zeros((nd, 7))
        eta[:, 0] = rng.uniform(0.3, 2.0, nd)
        emis = np.zeros((nd, 4))
        emis[:, :3] = rng.uniform(-0.5, 1.5, (nd, 3))
        dz = -np.diff(z)
        mu = np.array([0.3, -0.7])
        tau = np.r_[0.0, np.cumsum(0.5 * (eta[:-1, 0] + eta[1:, 0]) * dz)][:, None]
        source = (emis[:, :3] / eta[:, :1])[:, :, None, None, None] * np.ones((1, 1, 1, 2, 1))
        top = rng.uniform(0.0, 1.0, (3, 1, 2, 1))
        bottom = rng.uniform(0.0, 1.0, (3, 1, 2, 1))
        expected = solve_unpolarized_transfer(source, tau, mu, top, bottom)
        for k, ray in enumerate(mu):
            boundary = np.r_[(bottom if ray > 0 else top)[:, 0, k, 0], 0.0]
            result = delo_polarized_ray(eta, emis, dz, ray, boundary)
            np.testing.assert_allclose(result[:, :3], expected[:, :, 0, k, 0], rtol=1e-12)
            np.testing.assert_array_equal(result[:, 3], 0.0)

    def test_weights_are_continuous_across_series_switch(self):
        for delta in (1e-8, 1e-3, 0.0099999, 0.01, 0.0100001, 0.5, 30.0):
            exact_d = np.longdouble(delta)
            ratio = -np.expm1(-exact_d) / exact_d
            expected = (np.exp(-exact_d), ratio - np.exp(-exact_d), 1 - ratio)
            np.testing.assert_allclose(_delo_weights(delta), np.array(expected, dtype=float), rtol=1e-9)

    def test_rejects_invalid_inputs(self):
        eta, emis, dz = np.tile(ETA, (3, 1)), np.tile(EMIS, (3, 1)), np.array([0.1, 0.1])
        bad_eta = eta.copy()
        bad_eta[1, 0] = 0.0
        for args in ((bad_eta, emis, dz, 0.5, INCIDENT), (eta, emis, np.array([0.1, 0.0]), 0.5, INCIDENT),
                     (eta, emis, dz, 0.0, INCIDENT), (eta[:1], emis[:1], dz[:0], 0.5, INCIDENT)):
            with self.assertRaises(ValueError):
                delo_polarized_ray(*args)


class SolveDeloTests(unittest.TestCase):
    @staticmethod
    def field(rng, depth=12, nnu=3, nphi=2):
        mu = np.array([0.4, -0.6, 1.0])
        eta = rng.uniform(-0.1, 0.1, (depth, 7, nnu, len(mu), nphi))
        eta[:, 0] = rng.uniform(0.5, 2.0, (depth, nnu, len(mu), nphi))
        emis = rng.uniform(-0.2, 1.0, (depth, 4, nnu, len(mu), nphi))
        z = np.cumsum(rng.uniform(0.05, 0.3, depth))[::-1].copy()
        top = rng.uniform(0.0, 1.0, (4, nnu, len(mu), nphi))
        bottom = rng.uniform(0.0, 1.0, (4, nnu, len(mu), nphi))
        return emis, eta, z, mu, top, bottom

    def test_field_wrapper_matches_ray_kernel_on_every_axis(self):
        emis, eta, z, mu, top, bottom = self.field(np.random.default_rng(7))
        result = solve_delo(emis, eta, z, mu, top, bottom)
        self.assertEqual(result.shape, emis.shape)
        for f in range(emis.shape[2]):
            for m, ray in enumerate(mu):
                for p in range(emis.shape[4]):
                    incoming = (bottom if ray > 0 else top)[:, f, m, p]
                    expected = delo_polarized_ray(
                        np.ascontiguousarray(eta[:, :, f, m, p]),
                        np.ascontiguousarray(emis[:, :, f, m, p]), -np.diff(z), ray, incoming)
                    np.testing.assert_allclose(result[:, :, f, m, p], expected, rtol=1e-13)

    def test_field_wrapper_rejects_inconsistent_inputs(self):
        emis, eta, z, mu, top, bottom = self.field(np.random.default_rng(8))
        bad_eta = eta.copy()
        bad_eta[3, 0, 1, 1, 1] = 0.0
        cases = (
            (emis, eta[:, :6], z, mu, top, bottom),
            (emis, eta, z[::-1], mu, top, bottom),
            (emis, eta, z, np.array([0.4, 0.0, 1.0]), top, bottom),
            (emis, eta, z, mu, top[:3], bottom),
            (emis, bad_eta, z, mu, top, bottom),
            (np.where(np.arange(emis.size).reshape(emis.shape) == 5, np.nan, emis), eta, z, mu, top, bottom),
        )
        for args in cases:
            with self.assertRaises(ValueError):
                solve_delo(*args)


if __name__ == "__main__":
    unittest.main()
