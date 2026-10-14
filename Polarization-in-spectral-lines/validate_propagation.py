"""Validation of the grid propagation matrix (eta, rho) and emissivity, including Stokes V.

Run:  MPLBACKEND=Agg PYTHONPATH=.:.. python validate_propagation.py
Known cases used:
  1. B = 0: eta_I = eta0 * Voigt, nothing else.
  2. Zeeman triplet (LL04 Eq. 9.x): eta_{I,Q,U,V} and rho_{Q,U,V} for arbitrary (theta, chi).
  3. Weak-field limits: V = nu_L B cos(theta) dI/dnu ; Q = -1/4 (nu_L B)^2 sin^2 cos2chi d2I/dnu2.
  4. Kirchhoff: eps = eta * B_nu (C_ul >> A) and eps = eta * J00 (isotropic, no collisions).
  5. Milne-Eddington (Unno-Rachkovsky): I(0) = B0 e1 + mu B1 K^-1 e1 versus the closed form.
Figure: validate_propagation.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

from atmosphere import LineModel, C_LIGHT
from grid_opacity import point_emission_and_propagation, eta0, planck, NU_LARMOR

NU0, DND = 4.3e14, 4e9
LINE = LineModel(NU0, 5e7, 1.0, 1.0, 1, 0, 40)
VMIC = DND * C_LIGHT / NU0
A_DAMP = LINE.A_ul / (4 * np.pi * DND)
X = np.linspace(-8, 8, 1601)
NU = X * DND
MU, PHI = np.array([1.0]), np.array([0.0])
AMP = eta0(LINE, 1.0) / DND
ZERO_J = np.zeros((3, 5), complex)
results = []


def check(name, err, tol):
    ok = err < tol
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {err:.2e} (tol {tol:.0e})")


def run(B, th, chi, J=ZERO_J, C_ul=0.0, T=1e-12, gamma=0.0):
    eps, eta = point_emission_and_propagation(
        J, LINE, MU, PHI, NU, T, 1.0, B, th, chi, 0.0, VMIC, 0.0, C_ul, gamma=gamma)
    return eps[:, :, 0, 0], eta[:, :, 0, 0] / AMP  # eta in units of eta0/dnu_D


def triplet(B, th, chi):
    """LL04 normal-Zeeman-triplet eta, rho (units eta0/dnu_D); b/r assigned as M=+1/-1 shifts."""
    vH = NU_LARMOR * B / DND
    w = lambda M: wofz(X - M * vH + 1j * A_DAMP) / np.sqrt(np.pi)
    wp, wb, wr = w(0), w(1), w(-1)  # blue component = M=+1 shift (sign fixed by weak-field V)
    s, c = np.sin(th), np.cos(th)
    d_lin = lambda f: f(wp) - 0.5 * (f(wb) + f(wr))
    out = []
    for f in (np.real, np.imag):
        eI = 0.5 * f(wp) * s**2 + 0.25 * (f(wb) + f(wr)) * (1 + c**2)
        eQ = 0.5 * d_lin(f) * s**2 * np.cos(2 * chi)
        eU = 0.5 * d_lin(f) * s**2 * np.sin(2 * chi)
        eV = 0.5 * (f(wr) - f(wb)) * c
        out.append((eI, eQ, eU, eV))
    (eI, eQ, eU, eV), (_, rQ, rU, rV) = out
    return np.array([eI, eQ, eU, eV, rQ, rU, rV])


# 1. B = 0
_, eta = run(0.0, 0.3, 0.2)
phi0 = wofz(X + 1j * A_DAMP).real / np.sqrt(np.pi)
check("B=0: eta_I - Voigt", np.abs(eta[0] - phi0).max() / phi0.max(), 1e-12)
check("B=0: polarisation absent", np.abs(eta[1:]).max(), 1e-12)
check("B=0: int eta_I dnu / eta0", abs(np.trapezoid(eta[0], NU) / DND - 1) , 2e-3)

# 2. Zeeman triplet, all seven components, many geometries
worst = 0.0
for B in (30.0, 1000.0, 3000.0):
    for th, chi in [(0, 0), (0.4, 0.0), (0.9, 0.7), (np.pi / 2, 0.3), (2.0, 2.5), (np.pi, 0)]:
        ref = triplet(B, th, chi)
        _, eta = run(B, th, chi)
        worst = max(worst, np.abs(eta - ref).max() / np.abs(ref).max())
check("Zeeman triplet eta_{I,Q,U,V}, rho_{Q,U,V} (3 B, 6 geometries)", worst, 1e-9)

# 3. Weak-field limits (B = 5 G)
B = 5.0
dl = NU_LARMOR * B
th, chi = 0.8, 0.5
_, eta = run(B, th, chi)
dI = np.gradient(eta[0], NU)
d2I = np.gradient(dI, NU)
Vwf = dl * np.cos(th) * dI * 1.0
check("weak field: eta_V = +nu_L B cos(th) d eta_I/dnu", np.abs(eta[3] - Vwf).max() / np.abs(Vwf).max(), 2e-3)
Qwf = -0.25 * dl**2 * np.sin(th)**2 * np.cos(2 * chi) * d2I
check("weak field: eta_Q = -1/4 (nu_L B)^2 sin^2 cos2chi eta_I''", np.abs(eta[1] - Qwf).max() / np.abs(Qwf).max(), 5e-3)
Uwf = -0.25 * dl**2 * np.sin(th)**2 * np.sin(2 * chi) * d2I
check("weak field: eta_U analogue", np.abs(eta[2] - Uwf).max() / np.abs(Uwf).max(), 5e-3)
psi = wofz(X + 1j * A_DAMP).imag / np.sqrt(np.pi)
dpsi = np.gradient(psi, NU)
rVwf = dl * np.cos(th) * dpsi
check("weak field: rho_V = nu_L B cos(th) d psi/dnu", np.abs(eta[6] - rVwf).max() / np.abs(rVwf).max(), 2e-3)

# 4. Kirchhoff checks
J00 = 1e-5
Jiso = ZERO_J.copy()
Jiso[0, 2] = J00
for B_, th_, chi_ in [(0.0, 0.0, 0.0), (300.0, 0.7, 0.4), (2000.0, 2.2, 1.1)]:
    eps, eta = run(B_, th_, chi_, J=Jiso)
    sc = AMP
    err = np.abs(eps - J00 * eta[:4] * sc).max() / np.abs(J00 * eta[:4] * sc).max()
    check(f"eps = eta*J00 (isotropic, B={B_:g}, incl. V)", err, 1e-9)
T = 6000.0
Bnu = planck(NU0, T)
eps, eta = run(300.0, 0.7, 0.4, J=Jiso, C_ul=1e12, T=T)
ref = Bnu * eta[:4] * AMP
check("eps = B_nu * eta (C_ul >> A), incl. V", np.abs(eps - ref).max() / np.abs(ref).max(), 1e-4)

# 5. Milne-Eddington / Unno-Rachkovsky
def kmat(e):
    eI, eQ, eU, eV, rQ, rU, rV = e
    return np.array([[eI, eQ, eU, eV], [eQ, eI, rV, -rU], [eU, -rV, eI, rQ], [eV, rU, -rQ, eI]])


def milne_eddington(eta, beta, B0, B1, mu=1.0):
    out = np.zeros((4, eta.shape[1]))
    cl = np.zeros((4, eta.shape[1]))
    for k in range(eta.shape[1]):
        K = np.eye(4) + beta * kmat(eta[:, k])
        out[:, k] = mu * B1 * np.linalg.solve(K, np.array([1.0, 0, 0, 0]))
        e = beta * eta[:, k]
        d = (1 + e[0])**2 * ((1 + e[0])**2 - e[1]**2 - e[2]**2 - e[3]**2 + e[4]**2 + e[5]**2 + e[6]**2) \
            - (e[1] * e[4] + e[2] * e[5] + e[3] * e[6])**2
        ip = e[1] * e[4] + e[2] * e[5] + e[3] * e[6]
        cl[0, k] = (1 + e[0]) * ((1 + e[0])**2 + e[4]**2 + e[5]**2 + e[6]**2) / d
        cl[1, k] = ((1 + e[0])**2 * (-e[1]) + (1 + e[0]) * (e[2] * e[6] - e[3] * e[5]) - e[4] * ip) / d
        cl[2, k] = ((1 + e[0])**2 * (-e[2]) + (1 + e[0]) * (e[3] * e[4] - e[1] * e[6]) - e[5] * ip) / d
        cl[3, k] = ((1 + e[0])**2 * (-e[3]) + (1 + e[0]) * (e[1] * e[5] - e[2] * e[4]) - e[6] * ip) / d
    return out, mu * B1 * cl


cases = [("B=1500 G, θ_B=0°, χ_B=0°", 1500.0, 0.0, 0.0),
         ("B=1500 G, θ_B=90°, χ_B=17°", 1500.0, np.pi / 2, np.deg2rad(17.0)),
         ("B=1500 G, θ_B=57°, χ_B=34°", 1500.0, np.deg2rad(57.0), np.deg2rad(34.0))]
fig, axes = plt.subplots(len(cases), 4, figsize=(15, 8), sharex=True)
for row, (lab, B_, th_, chi_) in enumerate(cases):
    _, eta = run(B_, th_, chi_)
    num, closed = milne_eddington(eta, 5.0, 1.0, 1.0)
    cont = 1.0 + 1.0
    err = np.abs(num - closed).max() / np.abs(num).max()
    check(f"Unno-Rachkovsky closed form vs K^-1 ({lab})", err, 1e-9)
    for i, nm in enumerate("IQUV"):
        ax = axes[row, i]
        off = 1.0 if i == 0 else 0.0
        y_num, y_cl = (num[i] + off) / cont, (closed[i] + off) / cont
        ax.plot(X, y_num, "k", label="K$^{-1}$ solve")
        ax.plot(X, y_cl, "r--", label="closed form")
        name = "I/I_c" if i == 0 else f"{nm}/I_c"
        ax.set_title(f"{lab}: {name}", fontsize=8)
        if i > 0 and np.abs(y_num).max() < 1e-8:
            ax.set_ylim(-0.15, 0.15)
            ax.text(0.5, 0.5, "zero (|signal| < 1e-8)", transform=ax.transAxes, ha="center", fontsize=8)
    axes[row, 0].legend(fontsize=7)
for ax in axes[-1]:
    ax.set_xlabel(r"$(\nu-\nu_0)/\Delta\nu_D$")
fig.tight_layout()
fig.savefig("validate_propagation.png", dpi=110)

# Milne-Eddington weak-field V against the derivative of Stokes I (end-to-end Stokes V)
B_, th_, chi_ = 5.0, 0.8, 0.0
_, eta = run(B_, th_, chi_)
num, _ = milne_eddington(eta, 5.0, 1.0, 1.0)
I_prof = 1.0 + num[0]
Vwf = NU_LARMOR * B_ * np.cos(th_) * np.gradient(I_prof, NU)
check("ME profile: V = nu_L B cos(th) dI/dnu (weak field)", np.abs(num[3] - Vwf).max() / np.abs(Vwf).max(), 5e-3)

print(f"\n{sum(results)}/{len(results)} checks passed; figure: validate_propagation.png")
