# -*- coding: utf-8 -*-
"""
Effective thermophysical properties of the Cu-Al2O3/water hybrid nanofluid.

Implements the single-phase property model of the paper:
  * density        - volumetric mixture rule
  * specific heat  - mass-weighted average
  * viscosity      - Brinkman correlation
  * conductivity   - sequential Hamilton-Crosser (Al2O3 first, then Cu)

All quantities are SI. `phi1` = Cu volume fraction, `phi2` = Al2O3 volume
fraction, total loading phi = phi1 + phi2.

Author:  Muhammed Anaz Khan  (ORCID: 0000-0002-8837-9865)
License: MIT
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    rho: float      # density            [kg/m^3]
    cp: float       # specific heat       [J/kg.K]
    k: float        # thermal conductivity[W/m.K]
    mu: float = 0.0 # dynamic viscosity   [Pa.s] (base fluid only)


# baseline properties at 300 K (Section 2.4 of the paper)
WATER = Material(rho=998.0, cp=4182.0, k=0.613, mu=8.91e-4)
CU    = Material(rho=8933.0, cp=385.0,  k=401.0)
AL2O3 = Material(rho=3970.0, cp=765.0,  k=40.0)
SILICON_K = 148.0   # substrate conductivity [W/m.K]


def hamilton_crosser(kp, kbf, phi, n=3.0):
    """Single-step Hamilton-Crosser conductivity ratio knf/kbf.
    n = 3/psi is the shape factor (n = 3 for spheres, psi = 1)."""
    num = kp + (n - 1.0) * kbf - (n - 1.0) * phi * (kbf - kp)
    den = kp + (n - 1.0) * kbf + phi * (kbf - kp)
    return num / den


def k_hybrid(phi1, phi2, bf=WATER, p1=CU, p2=AL2O3, n=3.0):
    """Effective conductivity of the Cu-Al2O3 hybrid, applied sequentially:
    Al2O3 is dispersed first, then Cu into that intermediate fluid."""
    # step 1: Al2O3 in water at fraction phi2 / (1 - phi1)
    r1 = hamilton_crosser(p2.k, bf.k, phi2 / (1.0 - phi1), n)
    k_int = r1 * bf.k
    # step 2: Cu in the intermediate fluid at fraction phi1
    r2 = hamilton_crosser(p1.k, k_int, phi1, n)
    return r2 * k_int


def properties(phi1, phi2, bf=WATER, p1=CU, p2=AL2O3, n=3.0):
    """Return the effective (rho, cp, mu, k) of the hybrid nanofluid."""
    phi = phi1 + phi2
    rho = (1.0 - phi) * bf.rho + phi1 * p1.rho + phi2 * p2.rho
    cp = ((1.0 - phi) * bf.rho * bf.cp
          + phi1 * p1.rho * p1.cp
          + phi2 * p2.rho * p2.cp) / rho
    mu = bf.mu / (1.0 - phi) ** 2.5                      # Brinkman
    k = k_hybrid(phi1, phi2, bf, p1, p2, n)              # sequential H-C
    return dict(rho=rho, cp=cp, mu=mu, k=k)


def dimensionless(phi1, phi2, u_in, Dh, **kw):
    """Reynolds, Prandtl and Peclet numbers at the given inlet velocity."""
    pr = properties(phi1, phi2, **kw)
    Re = pr["rho"] * u_in * Dh / pr["mu"]
    Pr = pr["mu"] * pr["cp"] / pr["k"]
    return dict(Re=Re, Pr=Pr, Pe=Re * Pr, **pr)


if __name__ == "__main__":
    # reproduce the additivity check of Section 5.5 (knf/kbf at phi = 0.02)
    kbf = WATER.k
    print("Conductivity ratio knf/kbf at total phi = 0.02")
    print(f"  hybrid (phi1=phi2=0.01): {k_hybrid(0.01, 0.01)/kbf:.4f}")
    print(f"  Cu-only  (phi=0.02)    : {k_hybrid(0.02, 0.0)/kbf:.4f}")
    print(f"  Al2O3-only (phi=0.02)  : {k_hybrid(0.0, 0.02)/kbf:.4f}")
    lin = 0.5 * (k_hybrid(0.02, 0.0) + k_hybrid(0.0, 0.02)) / kbf
    print(f"  linear mix of the two  : {lin:.4f}")
    print()
    for phi in (0.0, 0.01, 0.02, 0.03):
        pr = properties(phi / 2, phi / 2)
        print(f"phi={phi:4.2f}  rho={pr['rho']:7.1f}  cp={pr['cp']:6.1f}  "
              f"mu={pr['mu']:.3e}  k={pr['k']:.4f}  knf/kbf={pr['k']/kbf:.4f}")
