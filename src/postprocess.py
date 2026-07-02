# -*- coding: utf-8 -*-
"""
Closed-form performance metrics fitted to the PINN predictions (Section 5 of the
paper): Darcy friction factor, pumping power, Nusselt-number enhancement and the
performance evaluation criterion (PEC). Running this file reproduces Table 2.

Author:  Muhammed Anaz Khan  (ORCID: 0000-0002-8837-9865)
License: MIT
"""
import numpy as np


def friction_factor(phi, Re):
    """f = C(phi) * Re^-0.96,  C(phi) = 79.4 (1 + 16.4 phi + 84.6 phi^2)."""
    C = 79.4 * (1.0 + 16.4 * phi + 84.6 * phi ** 2)
    return C * Re ** (-0.96)


def pec(Nu_nf, Nu_bf, f_nf, f_bf):
    """Performance evaluation criterion = (Nu_nf/Nu_bf) / (f_nf/f_bf)^(1/3)."""
    return (Nu_nf / Nu_bf) / (f_nf / f_bf) ** (1.0 / 3.0)


# PINN-predicted average Nusselt number at Re = 500, q'' = 50 W/cm^2 (Table 2)
NU_AVG = {0.000: 6.42, 0.005: 6.71, 0.010: 6.98, 0.020: 7.37, 0.030: 7.67}
DT_BASE = {0.000: 38.2, 0.005: 36.7, 0.010: 35.3, 0.020: 33.6, 0.030: 32.9}
PPUMP_MW = {0.000: 1.38, 0.005: 1.47, 0.010: 1.57, 0.020: 1.78, 0.030: 2.01}
# heat-transfer enhancement Nu_nf/Nu_bf at phi = 0.02 for three Reynolds numbers
ENHANCE_2PCT = {200: 1.118, 500: 1.148, 1000: 1.174}


if __name__ == "__main__":
    Re = 500.0
    f_bf = friction_factor(0.0, Re)
    Nu_bf = NU_AVG[0.0]
    print(f"{'phi(vol%)':>9} {'Nu_avg':>7} {'f':>7} {'dTbase(K)':>10} "
          f"{'Ppump(mW)':>10} {'enhance':>8} {'PEC':>6}")
    for phi in sorted(NU_AVG):
        f = friction_factor(phi, Re)
        enh = NU_AVG[phi] / Nu_bf
        P = pec(NU_AVG[phi], Nu_bf, f, f_bf)
        print(f"{phi*100:9.1f} {NU_AVG[phi]:7.2f} {f:7.3f} {DT_BASE[phi]:10.1f} "
              f"{PPUMP_MW[phi]:10.2f} {enh:8.3f} {P:6.3f}")

    # PEC at phi = 2% rises with Reynolds number because the enhancement grows
    fr = friction_factor(0.02, Re) / f_bf                    # f-ratio is Re-independent
    frcube = fr ** (1 / 3)
    print(f"\nPEC at phi = 2 vol.% vs Reynolds number (f-ratio^(1/3) = {frcube:.3f}):")
    for R, enh in ENHANCE_2PCT.items():
        print(f"  Re={R:5d}:  PEC = {enh / fr ** (1/3):.3f}")
    print("\nNote: with the definition above and the Table 2 values, PEC peaks at")
    print("about 1.04 (Re=500) and 1.06 (Re=1000) - a modest net gain. This is the")
    print("value the manuscript should quote; see README 'Known consistency notes'.")
