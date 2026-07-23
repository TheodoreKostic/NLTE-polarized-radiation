# Stokes Frame And V Diagnostics Summary

## Goal

This note documents what was changed in the Stokes profile workflow, why the old approach failed in the vertical-field case, and why the new frame-consistent branch works.

The original issue was:

- Fig. 13.8-like geometry looked good.
- Fig. 13.6-like geometry (vertical field case) gave Stokes V that was too small by orders of magnitude and had extra lobe structure.

## Root Causes Found

### 1. Profile convention mismatch between generalized and Appendix forms

A direct comparison showed that Phi_generalized and Phi_appendix did not match for K=2, Kp=1, Q=plus/minus 1 in the real part.

Observed mismatch before fix:

- max absolute real difference about 9.6e-4

Fix:

- In Profile_fun.py, phi_q and psi_q were changed to use x - q*vH (not x + q*vH), making Appendix profile component centers consistent with the generalized-profile convention used in Eq. (10.40).

Result:

- Phi_generalized and Phi_appendix now agree to machine precision for the tested channels.

### 2. Mixed-frame contraction in Eq. (13.20)

The contraction for emissivity uses products of T, rho, and Phi. If these objects are not represented in one consistent frame, cancellation and amplitude errors appear, especially in weak-V regimes.

The problematic pattern was:

- Hanle treatment and profile components effectively in a magnetic-field-oriented convention.
- Stokes geometry tensors evaluated with a fixed observer-angle set that did not move consistently with the frame conversion.

This inconsistency was most damaging for the vertical-field-like setup, where the V signal depends on delicate term balance.

### 3. Q/U basis (gamma) not transformed when changing frame

After introducing a magnetic-frame branch, V improved, but Q/U showed sign or swap-like behavior for some geometries.

Reason:

- gamma is the Stokes reference-axis angle in the plane perpendicular to LOS.
- Rotating LOS/frame without rotating the +Q reference axis changes the Q/U basis.
- This creates apparent Q/U sign/swap differences even when underlying physics is correct.

## Implemented Changes

## A. New frame-contraction branch toggle

Added a branch switch in Stokes_V_profile.py:

- USE_LL04_MAG_FRAME_BRANCH

Modes:

- False: original vertical-frame full-Hanle path.
- True: LL04-style magnetic-frame contraction path.

In magnetic-frame mode:

1. LOS is rotated into magnetic frame.
2. J rank-2 tensor is rotated into magnetic frame.
3. Hanle is applied diagonally there as rho_Q = J_Q / (1 + iQHu).
4. Contraction is done in that same frame.

This removes mixed-frame inconsistencies in Eq. (13.20).

## B. Q/U reference-mode toggle

Added:

- Q_U_REFERENCE_MODE

Supported options:

- transport_gamma
- fixed_gamma_rotate_qu_back

Meaning:

- transport_gamma: transports the +Q axis with frame change and recomputes gamma in the new frame.
- fixed_gamma_rotate_qu_back: keeps original gamma in contraction and applies an explicit post-rotation in the Q-U plane.

Both are valid conventions if applied consistently.

## C. Fractional-polarization plotting (Fig. 13.7-like)

Added dedicated plotting helper for:

- pQ = Q/I
- pU = U/I
- pV = V/I

Also plots dashed integrated reference levels:

- tilde pQ = integral(Q) / integral(I)
- tilde pU = integral(U) / integral(I)
- tilde pV = integral(V) / integral(I)

Produced outputs:

- Fractional_polarization_fig13_7_like_generalized.png
- Fractional_polarization_fig13_7_like_appendix.png

## D. Diagnostic controls

Added diagnostic switches and checks to isolate cancellation and symmetry behavior:

- RUN_DIAG
- OLD_DEBUG
- decomposition plotting for V channels
- parity check for V(x) oddness
- pair-conjugation residual checks

These diagnostics showed:

- Very strong odd parity consistency in V.
- Dominant prior mismatch was profile convention and frame/basis consistency, not random numerical instability.

## Why The New Approach Works

The new approach works because it enforces consistency in three places simultaneously:

1. Profile convention consistency:

- Generalized and Appendix profile definitions are now aligned.

2. Frame consistency:

- T, rho, and geometry are evaluated in one coherent frame for the Eq. (13.20) contraction.

3. Polarization-basis consistency:

- Q/U reference axis handling is explicit, preventing accidental sign/swap artifacts.

In short, the old workflow could look correct for some geometries but fail badly for others because hidden inconsistencies only become obvious when cancellation sensitivity is high (as in the problematic vertical-field case).

## Why One Configuration Could Match While Another Failed

This is expected when cancellations differ by geometry:

- In one geometry (Fig. 13.8-like), the signal can be less sensitive to small convention mismatch.
- In another (Fig. 13.6-like), the same mismatch can suppress or distort V strongly.

So partial success in one configuration does not guarantee frame/profile correctness globally.

## Practical Usage Notes

1. Use USE_LL04_MAG_FRAME_BRANCH = True to test frame-consistent Eq. (13.20) mode.
2. If Q/U orientation appears off, switch Q_U_REFERENCE_MODE between:

- transport_gamma
- fixed_gamma_rotate_qu_back

3. Compare both:

- absolute Stokes profiles (I,Q,U,V)
- fractional profiles (pQ,pU,pV)

because Fig. 13.7 discussion is in terms of fractional polarization.

## Final Takeaway

The main issue was not a single algebraic sign in rho*T*Phi, but a chain of consistency issues:

- profile center convention,
- contraction frame consistency,
- and Q/U basis transport.

Once those were made consistent, the vertical-field V behavior became physically reasonable and diagnostics became self-consistent across configurations.
