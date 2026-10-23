# 1D response functions for scattering polarization and Hanle effect in normal Zeeman triplet

## GOAL: Devise RF to vector and microturbulent B for Sr 460.7 in a 1D atmosphere where $J_Q^K$ are considered given. 

## Assumptions: 

We know the structure of the atmosphere in thermodynamic parameters: $T, p, \vec{v}, \vec{B}$. Also $\rho$, if needed (can be inferred from EOS).

Given SNAPI or even better Lightweaver, we can calculate the lower level population of the line, continuum opacity, damping coefficient $a$, scattering probability $\epsilon$, and anisotropy $J_Q^K$ (or course in 1D case it will be only 00, and 02).

Flow:

1) Calculate locally the polarized emission vector and absorption matrix. This is pretty much done. 

    a) Include the turbulent depolarization as an option. 

    b) Include obligatory collisional depolarization. 

2) Implement DELO-like polarized radiative transfer solver to integrate this and get the emergent Stokes profile for given $\mu$. This solver you can take from SNAPI or Lightweaver. Be mindful that none of these probably contain polarized emission. 

3) Implement local reponse to $\vec{B}$. (JAX should be able to get you this one for free?)

4) Propagate the local response through the formal solution. JAX can probably also get this but the question is how much memory / computing time it takes. Maybe we can come up with our own analytical solution (see the appendix of Milic & van Noort, 2017 or was it 2018).

5) Write the paper. 

6) Publish the paper. 

7) Prepare to celebrate! :-)

