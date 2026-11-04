"""Plots for a homogeneous slab with a prescribed pump: anisotropy and Stokes profiles.

Model (single pass, NOT a self-consistent NLTE solution)
-------------------------------------------------------
The slab has constant temperature, density and field. At every depth the local
emissivity eps and propagation matrix eta are computed from one prescribed radiation
tensor J^K_Q (the "pump", the same at all depths). The Stokes vector at each depth,
frequency and ray is then obtained from the exact constant-slab propagator with no
radiation entering from outside:
    outward rays (mu > 0) start at the bottom, inward rays (mu < 0) at the top.
J^K_Q is recomputed from that Stokes field with jkq_from_stokes. The anisotropy
    w(z) = sqrt(2) J^2_0 / J^0_0
therefore shows how the slab itself builds up (or loses) anisotropy under the given
pump, which is what a depth-dependent solver will have to iterate to consistency.

Run from this directory:  python plot_homogeneous_slab.py   (writes PNGs to plots_slab/)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atmosphere import Atmosphere1D, Grid, LineModel
from formal_solver import propagate_constant_slab
from grid_opacity import fill_grid
from grid_tensors import jkq_from_stokes

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_slab")

# Sr I 460.7 nm-like Ju=1 -> Jl=0 line
LINE = LineModel(2.9979e10 / 4.607e-5, 2.01e8, 1.0, 1.0, 1.0, 0.0, 87.6)
N_DEPTH = 21
TAU_SLAB = 5.0   # vertical optical thickness at line centre (B = 0, no collisions)
T = 6000.0
GAMMA = np.pi / 2

# Prescribed pump: J^0_0 = 1, J^2_0 = -0.1 (anisotropy w = sqrt(2) * (-0.1) ~ -0.14)
PUMP = np.zeros((3, 5), complex)
PUMP[0, 2] = 1.0
PUMP[2, 2] = -0.1

CASES = {
    "B = 0": dict(B=0.0, th=0.0, B_unres=0.0, D2=0.0),
    "B = 30 G, 60 deg": dict(B=30.0, th=np.pi / 3, B_unres=0.0, D2=0.0),
    "30 G + unresolved 30 G": dict(B=30.0, th=np.pi / 3, B_unres=30.0, D2=0.0),
    "B = 0, D2 = A": dict(B=0.0, th=0.0, B_unres=0.0, D2=LINE.A_ul),
}


def build(case, z, n_l):
    n = len(z)
    atm = Atmosphere1D(
        z=z, T=np.full(n, T), n_l=np.full(n, n_l),
        B=np.full(n, case["B"]), theta_B=np.full(n, case["th"]), chi_B=np.full(n, 0.3),
        v_los=np.zeros(n), D2=np.full(n, case["D2"]), C_ul=np.zeros(n),
        vmic=np.full(n, 1e5), B_unres=np.full(n, case["B_unres"]),
    )
    grid = Grid.build(atm, LINE, n_mu_half=4, n_phi=8, n_nu=121, nu_max_doppler=6.0).allocate()
    grid.JKQ[:] = PUMP
    fill_grid(grid, gamma=GAMMA, n_theta=12, n_chi=12)
    return grid


def slab_stokes(grid, z):
    """Exact constant-slab Stokes at every depth, frequency and ray: [N, 4, nnu, nmu, nphi]."""
    top, bottom = z[0], z[-1]
    nnu, nmu, nphi = len(grid.nu), len(grid.mu), len(grid.phi)
    out = np.zeros((len(z), 4, nnu, nmu, nphi))
    for im, mu in enumerate(grid.mu):
        for k, zk in enumerate(z):
            dist = ((zk - bottom) if mu > 0 else (top - zk)) / abs(mu)
            for inu in range(nnu):
                for ip in range(nphi):
                    out[k, :, inu, im, ip] = propagate_constant_slab(
                        grid.eps[k, :, inu, im, ip], grid.eta[k, :, inu, im, ip],
                        dist, np.zeros(4),
                    )
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    # Choose the slab length so that tau_slab holds at line centre for B = 0.
    probe = build(CASES["B = 0"], np.array([1.0, 0.0]), 1.0)
    eta_c = probe.eta[0, 0, len(probe.nu) // 2, -1, 0]  # eta_I at line centre (independent of mu at B = 0)
    length = TAU_SLAB / eta_c  # vertical tau = eta_I(centre) * length, with n_l = 1
    z = np.linspace(length, 0.0, N_DEPTH)
    tau = TAU_SLAB * (z[0] - z) / length  # line-centre optical depth from the top

    results = {}
    for name, case in CASES.items():
        grid = build(case, z, 1.0)
        stokes = slab_stokes(grid, z)
        J = jkq_from_stokes(grid, stokes)
        results[name] = (grid, stokes, J)
        print(f"{name}: J00(top/mid/bottom) = {J[0, 0, 2].real:.3f} {J[N_DEPTH // 2, 0, 2].real:.3f}"
              f" {J[-1, 0, 2].real:.3f};  w(mid) = {np.sqrt(2) * J[N_DEPTH // 2, 2, 2].real / J[N_DEPTH // 2, 0, 2].real:+.3f}")

    # 1. Anisotropy and J00 versus depth
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, (grid, stokes, J) in results.items():
        j00 = J[:, 0, 2].real
        ax[0].plot(tau, j00, label=name)
        ax[1].plot(tau, np.sqrt(2) * J[:, 2, 2].real / j00, label=name)
        ax[2].plot(tau, np.hypot(J[:, 2, 1].real, J[:, 2, 1].imag) / j00 * np.sqrt(2), label=name)
    ax[0].set(xlabel=r"line-centre $\tau$ from top", ylabel=r"$J^0_0$", title="Mean intensity (profile-weighted)")
    ax[1].set(xlabel=r"line-centre $\tau$ from top", ylabel=r"$w=\sqrt{2}\,J^2_0/J^0_0$", title="Anisotropy")
    ax[2].set(xlabel=r"line-centre $\tau$ from top", ylabel=r"$\sqrt{2}\,|J^2_1|/J^0_0$", title="Transverse $|J^2_1|$ component")
    ax[1].axhline(np.sqrt(2) * PUMP[2, 2].real, color="k", ls=":", lw=1, label="prescribed pump")
    for a in ax:
        a.grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "anisotropy_vs_depth.png"), dpi=130)
    plt.close(fig)

    # 2. All J^2_Q components for the inclined-field case
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in ("B = 0", "B = 30 G, 60 deg"):
        J = results[name][2]
        for Q in range(0, 3):
            ax[0 if name == "B = 0" else 1].plot(tau, np.abs(J[:, 2, Q + 2]) / J[:, 0, 2].real, label=fr"$|J^2_{Q}|/J^0_0$")
        ax[0 if name == "B = 0" else 1].set(title=name, xlabel=r"line-centre $\tau$ from top")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "J2Q_components.png"), dpi=130)
    plt.close(fig)

    # 3. Emergent fractional polarization profiles (top of slab, outward rays)
    imu = len(results["B = 0"][0].mu) - 2  # an outward ray with intermediate mu
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for name, (grid, stokes, J) in results.items():
        x = grid.nu / LINE.doppler_width(T, 1e5)
        S = stokes[0, :, :, imu, 0]
        for i, lab in enumerate(("Q/I", "U/I", "V/I")):
            ax[i].plot(x, S[i + 1] / S[0], label=name)
            ax[i].set_ylabel(lab)
    ax[0].set_title(fr"Emergent profiles, $\mu$ = {grid.mu[imu]:.2f}, $\phi$ = 0")
    ax[2].set_xlabel(r"$(\nu-\nu_0)/\Delta\nu_D$")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "emergent_stokes.png"), dpi=130)
    plt.close(fig)

    # 4. Stokes versus depth at line centre for the inclined-field case
    grid, stokes, J = results["B = 30 G, 60 deg"]
    ic = len(grid.nu) // 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for im in (len(grid.mu) - 1, len(grid.mu) - 3):
        S = stokes[:, :, ic, im, 0]
        ax[0].plot(tau, S[:, 0], label=fr"I, $\mu$={grid.mu[im]:.2f}")
        for i, lab in ((1, "Q"), (2, "U")):
            ok = S[:, 0] > 0  # the boundary point of the incoming side has I = 0
            ax[1].plot(tau[ok], S[ok, i] / S[ok, 0], label=fr"{lab}/I, $\mu$={grid.mu[im]:.2f}")
    ax[0].set(xlabel=r"line-centre $\tau$ from top", ylabel="I at line centre")
    ax[1].set(xlabel=r"line-centre $\tau$ from top", ylabel="fractional polarization at line centre")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stokes_vs_depth_line_centre.png"), dpi=130)
    plt.close(fig)
    print("Saved plots to", OUT)


if __name__ == "__main__":
    main()
