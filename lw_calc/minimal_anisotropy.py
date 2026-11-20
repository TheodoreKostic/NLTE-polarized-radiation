## Minimal: line-averaged radiation field tensors (J00, J20), anisotropy and level populations
## of one line in a 1D atmosphere, with Lightweaver. Default: Sr I 4607 in FALC.
##
##   python minimal_anisotropy.py          -> sr4607_falc_Jbar_anisotropy.txt
##
## To use another 1D model, pass any lw.Atmosphere to anisotropy() (see the bottom of the file).

import numpy as np
import lightweaver as lw
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, Si_atom, Al_atom, CaII_atom, \
    Fe23_atom, He_9_atom, Mg_atom, N_atom, NaI_fine_atom, S_atom
import lightweaver.constants as Const

from custom_atoms import sr_tanausu


def vertical_tau(chi, z):
    #Optical depth from the top (z[0]), trapezoidal
    return 0.5 * chi[0] * (z[0] - z[1]) + np.r_[0.0, np.cumsum(0.5 * (chi[1:] + chi[:-1]) * (z[:-1] - z[1:]))]


def anisotropy(atmos, line=(8, 0), nmu=16, sr_dex=2.93, Nthreads=4):
    '''
    Converge Sr (sr_tanausu, active, CRD) in atmos, then compute the full I(lambda, mu, z)
    around the line with nmu Gauss-Legendre mu, and its profile-averaged moments.

    Parameters
    ----------
    atmos : lw.Atmosphere - 1D model atmosphere
    line : (j, i) - upper and lower level index of the transition, (8, 0) = Sr I 4607
    nmu : number of Gauss-Legendre mu points in (0, 1] for the moments
    sr_dex : Sr abundance, log eps

    Returns
    -------
    dict of depth arrays (index 0 = top of the atmosphere) plus line constants.
    '''
    j, i = line # Default is Sr I 4607 (8, 0) (check the atomic model)

    # 1. NLTE populations (5 rays), same atom set as minimal_synth.synth()
    atmos.quadrature(5)
    abund = lw.AtomicAbundance({lw.PeriodicTable['Sr']: sr_dex})  
    
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(),
                            Fe23_atom(), He_9_atom(), Mg_atom(), N_atom(), NaI_fine_atom(),
                            S_atom(), sr_tanausu()], abundance=abund)
    aSet.set_active('Sr') # only Sr is in non-LTE
    spect = aSet.compute_wavelength_grid()
    eqPops = aSet.compute_eq_pops(atmos)
    
    ctx = lw.Context(atmos, spect, eqPops, conserveCharge=False, Nthreads=Nthreads)
    lw.iterate_ctx_se(ctx, prd=False, quiet=True, popsTol=1e-4, JTol=1e-4)
    atom = ctx.activeAtoms[0] # This stores the info on the active atom:
    tl = next(t for t in atom.trans if (t.j, t.i) == line)

    # 2. Full I[lambda, mu, d, z] (d = 0 down, 1 up) on line +- 0.1 nm, populations fixed.
    #    Only formal_sol_gamma_matrices() fills depthData.
    x, wx = np.polynomial.legendre.leggauss(nmu)
    mu, wmu = 0.5 * (x + 1), 0.5 * wx
    wave = np.linspace(tl.lambda0 - 0.1, tl.lambda0 + 0.1, 401)
    
    _, rctx = ctx.compute_rays(wave, {'muz': list(mu), 'wmu': list(wmu)},
                               upOnly=False, returnCtx=True)
    
    rctx.depthData.fill = True
    rctx.formal_sol_gamma_matrices()
    t = next(t for t in rctx.activeAtoms[0].trans if (t.j, t.i) == line)
    lam = np.array(t.wavelength)
    I = np.array(rctx.depthData.I)[t.Nblue:t.Nblue + lam.size]

    # 3. Profile average with the weights Lightweaver uses for the rates (so Jbar00 = Rij/Bij), basically we calculate things on our own
    lp = np.r_[lam[0], lam, lam[-1]]
    W = (0.5 * (lp[2:] - lp[:-2]))[:, None, None, None] * wmu[None, :, None, None] * np.array(t.phi)
    norm = W.sum(axis=(0, 1, 2))
    W = W * (t.lambda0 / lam)[:, None, None, None]
    T20 = ((3 * mu**2 - 1) / (2 * np.sqrt(2)))[None, :, None, None]
    Jb00 = (W * I).sum(axis=(0, 1, 2)) / norm
    Jb20 = (W * T20 * I).sum(axis=(0, 1, 2)) / norm

    # 4. tau_line: total opacity (line + continuum) at lambda0 from the formal solution, integrated
    #    vertically from the top. Ray index -1 (mu ~ 1, upgoing); with vz = 0 every ray has the same chi.
    chi0 = np.array(rctx.depthData.chi)[np.argmin(np.abs(wave - t.lambda0)), -1, 1]
    z = np.array(atmos.z)
    tau_line = vertical_tau(chi0, z)
    n = np.array(atom.n)
    C_ul = np.array(atom.C)[i, j]                  # Lightweaver: C[i, j] is the rate j -> i
    T = np.array(atmos.temperature)
    lev = atom.atomicModel.levels
    Grad = next(l for l in atom.atomicModel.lines if (l.j, l.i) == line).broadening.natural[0].gamma
    Cst = C_ul * (1 - np.exp(-Const.HC / (tl.lambda0 * 1e-9 * Const.KBoltzmann * T)))

    # 5. tau500 and height (z = 0 at tau500 = 1) from the background opacity at 500 nm
    #    (Lightweaver's own atmos.tauRef is 1e4 off in 0.16.1, so don't use it)
    _, c500 = ctx.compute_rays(np.array([500.0]), [1.0], returnCtx=True)
    chi500 = np.array(c500.background.chi)[0]
    tau500 = vertical_tau(chi500, z)
    height = z - np.interp(0.0, np.log10(tau500), z)

    return dict(height=height, log_tau500=np.log10(tau500), T=T, log_tau_line=np.log10(tau_line), Jbar00=Jb00, Jbar20=Jb20, w=np.sqrt(2) * Jb20 / Jb00,
                n_lower=n[i], n_upper=n[j], C_ul=C_ul, epsilon=Cst / (tl.Aji + Cst),
                a_damp=np.array(tl.aDamp), Q_elast=np.array(tl.Qelast),
                lambda0=tl.lambda0, A_ul=tl.Aji, Gamma_rad=Grad, nmu=nmu, sr_dex=sr_dex,
                lower=lev[i].label, upper=lev[j].label,
                g_lower=lev[i].g, g_upper=lev[j].g)


def write_txt(fname, r, model):
    '''Write the depth table returned by anisotropy(); model is a one-line description.'''
    cols = ['height_km', 'log_tau500', 'T_K', 'log_tau_line', 'Jbar00', 'Jbar20', 'w',
            'n_lower', 'n_upper', 'C_ul', 'epsilon', 'a_damp', 'Q_elast']
    data = np.column_stack([r['height'] / 1e3, r['log_tau500']] + [r[c] for c in
                           ['T', 'log_tau_line', 'Jbar00', 'Jbar20', 'w', 'n_lower', 'n_upper',
                            'C_ul', 'epsilon', 'a_damp', 'Q_elast']])
    fmt = ['%11.2f', '%10.4f', '%9.1f', '%12.4f', '%13.6e', '%13.6e', '%10.6f',
           '%12.4e', '%12.4e', '%11.4e', '%11.4e', '%11.4e', '%11.4e']
    width = [9, 10, 9, 12, 13, 13, 10, 12, 12, 11, 11, 11, 11]
    hdr = f'''Line {r['lower']} - {r['upper']}, lambda0 = {r['lambda0']:.4f} nm (vacuum): line-averaged radiation field, populations and line parameters
Model: {model}, 1D plane-parallel.
NLTE: Lightweaver, Sr I/II model atom sr_tanausu (16 levels, 12 lines) active, CRD, log eps(Sr) = {r['sr_dex']:.2f}.
      Populations converged with 5 Gauss-Legendre rays; moments below from a final formal solution
      with {r['nmu']} Gauss-Legendre mu points.
      PRELIMINARY: the collisional rates of this transition are being revised (see C_ul below).
Definitions (dOmega/4pi normalised, axisymmetric):
  J00 = (1/2) int_{{-1}}^{{1}} I dmu,   J20 = 1/(4 sqrt2) int_{{-1}}^{{1}} (3mu^2 - 1) I dmu
  line-averaged: JbarK0 = int dlambda oint dOmega/4pi phi(lambda, mu, z) T^K_0(mu) I(lambda, mu, z),
  absorption profile phi normalised to 1, T^0_0 = 1, T^2_0 = (3mu^2 - 1)/(2 sqrt2)
  (Jbar00 = R_ij / B_ij of the line).   w = sqrt2 Jbar20 / Jbar00   (fractional anisotropy).
Line data: A_ul = {r['A_ul']:.4e} s^-1, Gamma_rad = {r['Gamma_rad']:.3e} s^-1, g_l = {r['g_lower']:g}, g_u = {r['g_upper']:g}.
n_lower, n_upper: NLTE populations of the lower ({r['lower']}) and upper ({r['upper']}) level
      [m^-3] (x 1e-6 for cm^-3).
C_ul: collisional de-excitation rate upper -> lower [s^-1], as used in the statistical equilibrium:
      electrons (CE, Bergemann) + neutral H (CH). The CH term is currently ~1e5 too small (pending fix),
      so C_ul is effectively electron-only. The electron term (Bergemann values x 1e-6, units not yet
      confirmed) is ~150x the Van Regemorter estimate at 5000 K, so epsilon may be strongly overestimated.
epsilon: photon destruction probability, eps = C_ul (1 - exp(-h nu/kT)) / (A_ul + C_ul (1 - exp(-h nu/kT))).
a_damp: Voigt damping parameter a = (Gamma_rad + Q_elast) lambda0 / (4 pi v_D), v_D = sqrt(2kT/m + vturb^2).
Q_elast: elastic collision rate [s^-1] = van der Waals broadening by neutral H and He (Unsold x 1.5);
      no Stark term. This is the elastic rate Lightweaver uses for PRD (not a depolarisation rate).
Units: J in W m^-2 Hz^-1 sr^-1 (x 1e3 for erg s^-1 cm^-2 Hz^-1 sr^-1).
Height: z = 0 at tau500 = 1, increasing outwards.
log_tau500, log_tau_line: log10 of the vertical (mu = 1) optical depth at 500 nm and at the line-centre
      wavelength lambda0. Both are integrated from the top in the same way (trapezoidal rule on the model grid,
      half a cell at the top point):  tau_0 = chi_0 (z_0 - z_1)/2,  tau_k = tau_0 + sum_(m<k) (chi_m + chi_m+1)/2 (z_m - z_m+1).
      500 nm: chi = continuum (background) opacity [m^-1].
      line centre: chi [m^-1] = total opacity at lambda0 that Lightweaver uses in the formal solution, with the
      converged NLTE populations: line opacity (h nu/4pi) (n_lower B_lu - n_upper B_ul) phi(lambda0, z)  (Voigt
      profile with a_damp above, corrected for stimulated emission) PLUS the continuum at lambda0 (H, H-, metal
      bound-free and free-free, Thomson and Rayleigh scattering, Sr bound-free). So tau_line is the total
      optical depth at line centre, not the line-only one. With vz = 0, chi is the same for every direction.
Rows go from the top of the atmosphere downwards.

''' + ' '.join(f'{c:>{wd}s}' for c, wd in zip(cols, width))
    np.savetxt(fname, data, header=hdr, fmt=fmt)
    print('wrote', fname)


if __name__ == '__main__':
    # Any 1D lw.Atmosphere works here, e.g. lw.Atmosphere.make_1d(lw.ScaleType.Geometric,
    # depthScale=z, temperature=T, vlos=vz, vturb=vturb, Pgas=pgas)   (SI units, z[0] = top)
    r = anisotropy(Falc82(), line=(8, 0), nmu=16)
    write_txt('sr4607_falc_Jbar_anisotropy.txt', r,
              model='FALC (Fontenla et al. 1993, lightweaver.fal.Falc82), vz = 0')
