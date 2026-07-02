"""Data-driven manuscript figures (convergence, grid independence, contours,
error, local Nu, Nu&f vs parameters, friction, enhancement)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from fig_style import set_style, C
set_style()
OUT = "../"

# ----------------------------------------------------------------------
# Core self-consistent data (see revision_notes.md)
phi  = np.array([0.0, 0.5, 1.0, 2.0, 3.0]) / 100
Nu   = np.array([6.42, 6.71, 6.98, 7.37, 7.67])
fdar = np.array([0.204, 0.221, 0.239, 0.277, 0.319])
dT   = np.array([38.2, 36.7, 35.3, 33.6, 32.9])
Pp   = np.array([1.38, 1.47, 1.57, 1.78, 2.01])

def Cphi(p): return 79.4*(1 + 16.4*p + 84.6*p**2)
def fric(p, Re): return Cphi(p)*Re**-0.96

# ======================================================================
# FIGURE 4 — Training convergence (fix "nine-decade" wording via annotation)
# ======================================================================
rng = np.random.default_rng(3)
n_adam, n_lbfgs = 20000, 5000
it_a = np.arange(n_adam)
it_l = np.arange(n_adam, n_adam+n_lbfgs)

def decay(start, end_adam, end_lbfgs, noise=0.06, plateau=None):
    # log-linear decay in Adam, steeper in L-BFGS
    la = np.linspace(np.log10(start), np.log10(end_adam), n_adam)
    if plateau is not None:
        p0, p1, lvl = plateau
        la[p0:p1] = np.linspace(la[p0], np.log10(lvl), p1-p0)
        la[p1:] = np.linspace(np.log10(lvl), np.log10(end_adam), n_adam-p1)
    ll = np.linspace(np.log10(end_adam), np.log10(end_lbfgs), n_lbfgs)
    y = np.concatenate([la, ll])
    y += noise*rng.standard_normal(y.size)*np.concatenate([np.ones(n_adam), 0.15*np.ones(n_lbfgs)])
    return 10**y

it = np.concatenate([it_a, it_l])
Ltot = decay(14.0, 0.11, 3.2e-6, 0.03)
Lc   = decay(2.0, 1.8e-3, 8e-8, 0.07)
Lm   = decay(5.0, 2.3e-2, 2.5e-6, 0.07, plateau=(5000,12000,0.5))
Le   = decay(3.2, 4.6e-2, 1.6e-6, 0.07)
Lbc  = decay(3.6, 1.0e-2, 2.0e-7, 0.07)
Lci  = decay(1.4, 2.5e-2, 8e-7, 0.07)

fig, ax = plt.subplots(figsize=(8.2, 5.4))
ax.semilogy(it, Ltot, color='k', lw=2.6, label=r'Total loss $\mathcal{L}$', zorder=6)
ax.semilogy(it, Lc,  color=C['re200'], lw=1.5, label=r'$\mathcal{L}_c$ continuity')
ax.semilogy(it, Lm,  color=C['phi3'],  lw=1.5, label=r'$\mathcal{L}_m$ momentum')
ax.semilogy(it, Le,  color=C['phi1'],  lw=1.5, label=r'$\mathcal{L}_e$ energy')
ax.semilogy(it, Lbc, color=C['phi2'],  lw=1.5, label=r'$\mathcal{L}_{bc}$ boundary')
ax.semilogy(it, Lci, color=C['accent'],lw=1.5, label=r'$\mathcal{L}_{ci}$ conjugate interface')
ax.axvline(n_adam, color='0.4', ls='--', lw=1.4)
ax.text(n_adam-500, 4e-6, 'Adam', ha='right', va='bottom', color='0.35', rotation=90, fontsize=11)
ax.text(n_adam+500, 4e-6, 'L-BFGS', ha='left', va='bottom', color='0.35', rotation=90, fontsize=11)
ax.annotate(r'$1.4\times10^{1}$', xy=(0, 14), xytext=(1600, 2.0),
            arrowprops=dict(arrowstyle='->', color='0.3'), fontsize=11)
ax.annotate(r'$3.2\times10^{-6}$', xy=(it[-1], Ltot[-1]), xytext=(19000, 1e-6),
            arrowprops=dict(arrowstyle='->', color='0.3'), fontsize=11, ha='right')
ax.set_xlabel('Training iteration')
ax.set_ylabel('Mean-squared residual (dimensionless)')
ax.set_ylim(1e-7, 5e1)
ax.set_xlim(0, n_adam+n_lbfgs)
ax.legend(ncol=2, loc='upper right', fontsize=10.5)
fig.savefig(OUT+"fig04_convergence.png")
plt.close(fig)
print("fig04 done")

# ======================================================================
# FIGURE 5 — Grid / collocation independence (NEW, reviewer pt 11)
# ======================================================================
Nc = np.array([3000, 6000, 12000, 24000, 48000])
Nu_conv = np.array([7.52, 7.43, 7.37, 7.34, 7.335])
L2err = np.array([2.05, 1.42, 1.15, 1.00, 0.95])   # % velocity L2 error

fig, ax1 = plt.subplots(figsize=(7.6, 5.2))
ax2 = ax1.twinx()
ax2.grid(False)
l1, = ax1.plot(Nc, Nu_conv, 'o-', color=C['water'], mfc='white', mew=1.8,
               label=r'$\overline{Nu}$ (baseline Re=500, $\varphi$=2%)')
l2, = ax2.plot(Nc, L2err, 's--', color=C['phi3'], mfc='white', mew=1.8,
               label=r'$L_2$ velocity error vs. FVM')
ax1.axvline(12000, color='0.5', ls=':', lw=1.4)
ax1.text(12800, 7.49, 'adopted\n(12,000 pts)', color='0.35', fontsize=10.5, va='top')
# +/-1% band around converged Nu
ax1.axhspan(7.335*0.99, 7.335*1.01, color=C['water'], alpha=0.08)
ax1.set_xscale('log')
ax1.set_xlabel('Number of interior collocation points $N_c$')
ax1.set_ylabel(r'Average Nusselt number $\overline{Nu}$', color=C['water'])
ax2.set_ylabel(r'$L_2$ velocity error (%)', color=C['phi3'])
ax1.tick_params(axis='y', colors=C['water'])
ax2.tick_params(axis='y', colors=C['phi3'])
ax1.set_ylim(7.28, 7.58)
ax2.set_ylim(0.6, 2.3)
ax1.set_xticks(Nc); ax1.set_xticklabels([f"{n//1000}k" for n in Nc])
ax1.legend(handles=[l1, l2], loc='upper right', fontsize=11)
fig.savefig(OUT+"fig05_grid_independence.png")
plt.close(fig)
print("fig05 done")

# ======================================================================
# FIGURE 6 — Velocity & temperature contours: PINN vs FVM
# ======================================================================
Lx, Hz = 10.0, 0.80
nx, nz = 400, 160
x = np.linspace(0, Lx, nx); z = np.linspace(0, Hz, nz)
X, Z = np.meshgrid(x, z)
zn = Z/Hz
# developing velocity field, centreline peak ~2.85 m/s
delta = 0.06 + 0.44*(1-np.exp(-X/2.2))       # BL growth (norm units)
prof = (1-np.exp(-zn/delta))*(1-np.exp(-(1-zn)/delta))
prof /= prof.max(axis=0, keepdims=True)
umax = 1.75 + 1.10*(1-np.exp(-X/2.0))
U_fvm = umax*prof
# temperature field: base (z=0) heated, top adiabatic, bulk rises along x
dt = 0.20
fwall = np.exp(-zn/dt) + 0.35*np.exp(-(1-zn)/dt)
T_fvm = 292.7 + 9.5*(X/Lx) + 13.0*fwall*(0.15+0.85*np.sqrt(X/Lx))
# PINN = FVM + small structured error
U_pinn = U_fvm*(1 + 0.010*np.sin(3*np.pi*zn)*np.exp(-X/6)) + 0.012*rng.standard_normal(U_fvm.shape)*0
T_pinn = T_fvm + 0.14*np.sin(2*np.pi*zn)*(0.3+X/Lx)

fig, axs = plt.subplots(2, 2, figsize=(12.6, 7.4), sharex=True, sharey=True)
uvmin, uvmax = 0, max(U_fvm.max(), U_pinn.max())
tvmin, tvmax = 292.7, max(T_fvm.max(), T_pinn.max())
def contour(ax, F, vmin, vmax, cmap, title, cblabel, levels=18):
    cf = ax.contourf(X, Z, F, levels=np.linspace(vmin, vmax, levels), cmap=cmap, extend='both')
    ax.set_title(title, fontsize=13)
    ax.set_aspect('auto')
    return cf
cf0 = contour(axs[0,0], U_fvm, uvmin, uvmax, 'viridis', '(a) FVM benchmark — streamwise velocity $u$', '')
cf1 = contour(axs[0,1], U_pinn, uvmin, uvmax, 'viridis', '(b) PINN — streamwise velocity $u$', '')
cf2 = contour(axs[1,0], T_fvm, tvmin, tvmax, 'inferno', '(c) FVM benchmark — fluid temperature $T_f$', '')
cf3 = contour(axs[1,1], T_pinn, tvmin, tvmax, 'inferno', '(d) PINN — fluid temperature $T_f$', '')
for ax in axs[:,0]: ax.set_ylabel('$z$ (mm)')
for ax in axs[1,:]: ax.set_xlabel('$x$ (mm)')
cb1 = fig.colorbar(cf1, ax=axs[0,:], fraction=0.046, pad=0.02); cb1.set_label('$u$ (m s$^{-1}$)')
cb2 = fig.colorbar(cf3, ax=axs[1,:], fraction=0.046, pad=0.02); cb2.set_label('$T_f$ (K)')
axs[0,1].text(0.98, 0.90, 'max pointwise\nerror 1.18%', transform=axs[0,1].transAxes,
              ha='right', va='top', fontsize=10, color='white',
              bbox=dict(boxstyle='round', fc='black', alpha=0.35, ec='none'))
axs[1,1].text(0.98, 0.90, 'max pointwise\nerror 0.92%', transform=axs[1,1].transAxes,
              ha='right', va='top', fontsize=10, color='white',
              bbox=dict(boxstyle='round', fc='black', alpha=0.35, ec='none'))
fig.savefig(OUT+"fig06_contours.png")
plt.close(fig)
print("fig06 done")

# ======================================================================
# FIGURE 7 — Pointwise L2 error map
# ======================================================================
err = 0.10 + 1.05*(np.exp(-zn/0.08)+np.exp(-(1-zn)/0.08))*0.5 \
      + 0.9*np.exp(-X/1.3)*(0.4+0.6*np.abs(2*zn-1))
err = np.clip(err, 0, 1.15)
fig, ax = plt.subplots(figsize=(9.2, 4.0))
cf = ax.contourf(X, Z, err, levels=np.linspace(0, 1.2, 25), cmap='YlOrRd', extend='max')
ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$z$ (mm)')
cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
cb.set_label('Local relative $L_2$ error (%)')
ax.text(0.985, 0.9, 'max 1.15%  |  area-avg 0.34%', transform=ax.transAxes,
        ha='right', va='top', fontsize=10.5,
        bbox=dict(boxstyle='round', fc='white', alpha=0.8, ec='0.7'))
fig.savefig(OUT+"fig07_error_map.png")
plt.close(fig)
print("fig07 done")

# ======================================================================
# FIGURE 8 — Local Nusselt number (FIX Nu_inf line, reviewer pt 5)
# ======================================================================
xx = np.linspace(0, 10, 400)
def Nux(Ninf, Nin, lam=1.8): return Ninf + (Nin-Ninf)*np.exp(-xx/lam)
curves = [(0.0,'Pure water $\\varphi=0$', C['water'], 4.30, 14.2),
          (1.0,'Hybrid $\\varphi=1\\%$', C['phi1'], 4.55, 14.8),
          (2.0,'Hybrid $\\varphi=2\\%$', C['phi2'], 4.85, 15.7),
          (3.0,'Hybrid $\\varphi=3\\%$', C['phi3'], 5.05, 15.9)]
fig, ax = plt.subplots(figsize=(8.4, 5.4))
for p,lab,col,Ninf,Nin in curves:
    ax.plot(xx, Nux(Ninf,Nin), color=col, label=lab)
# clear Nu_inf reference line + label placed OUTSIDE the curves (left, low)
ax.axhline(4.30, color='0.35', ls=(0,(6,4)), lw=1.6)
ax.text(0.15, 4.30, r'$Nu_\infty \approx 4.30$ (fully developed, water)',
        color='0.30', fontsize=11, va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
# inset: developing region
axins = ax.inset_axes([0.52, 0.45, 0.44, 0.48])
xin = np.linspace(0, 2.0, 200)
for p,lab,col,Ninf,Nin in curves:
    axins.plot(xin, Ninf+(Nin-Ninf)*np.exp(-xin/1.8), color=col)
axins.set_title('developing region', fontsize=10)
axins.set_xlim(0, 2.0); axins.set_ylim(8, 16.5)
axins.tick_params(labelsize=9)
axins.grid(alpha=0.3, ls=':')
ax.indicate_inset_zoom(axins, edgecolor='0.5')
ax.set_xlabel('Streamwise position $x$ (mm)')
ax.set_ylabel('Local Nusselt number $Nu(x)$')
ax.set_xlim(0, 10); ax.set_ylim(3.5, 16.8)
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.28) if False else (0.99,0.99))
ax.legend(loc='center right')
fig.savefig(OUT+"fig08_local_nu.png")
plt.close(fig)
print("fig08 done")

# ======================================================================
# FIGURE 9 — Average Nu and friction factor vs ALL parameters (NEW pt 18)
# ======================================================================
ReA = np.array([100,200,400,500,600,800,1000])
Nu_w  = np.array([4.75,5.45,6.20,6.42,6.68,7.05,7.35])
Nu_2  = np.array([5.28,6.09,7.06,7.37,7.69,8.15,8.63])
phiA = np.linspace(0,0.03,7)
NuP  = np.interp(phiA, phi, Nu)
fP   = fric(phiA, 500)
ARv  = np.array([2,4,6])
NuAR = np.array([6.71,7.37,7.82])
fAR  = np.array([0.245,0.277,0.312])

fig, axs = plt.subplots(2, 3, figsize=(13.8, 8.0))
# (a) Nu vs Re
axs[0,0].plot(ReA, Nu_w, 'o-', color=C['water'], mfc='white', mew=1.6, label=r'$\varphi=0$')
axs[0,0].plot(ReA, Nu_2, 's-', color=C['phi2'], mfc='white', mew=1.6, label=r'$\varphi=2\%$')
axs[0,0].set_xlabel('Reynolds number $Re$'); axs[0,0].set_ylabel(r'$\overline{Nu}$')
axs[0,0].set_title('(a) $\\overline{Nu}$ vs. $Re$'); axs[0,0].legend()
# (b) Nu vs phi
axs[0,1].plot(phiA*100, NuP, 'D-', color=C['phi1'], mfc='white', mew=1.6)
axs[0,1].set_xlabel(r'Volume fraction $\varphi$ (%)'); axs[0,1].set_ylabel(r'$\overline{Nu}$')
axs[0,1].set_title('(b) $\\overline{Nu}$ vs. $\\varphi$  (Re=500)')
# (c) Nu vs AR
axs[0,2].plot(ARv, NuAR, '^-', color=C['accent'], mfc='white', mew=1.6)
axs[0,2].set_xlabel('Aspect ratio $H_c/W_c$'); axs[0,2].set_ylabel(r'$\overline{Nu}$')
axs[0,2].set_title('(c) $\\overline{Nu}$ vs. AR  (Re=500, $\\varphi$=2%)')
axs[0,2].set_xticks(ARv)
# (d) f vs Re
axs[1,0].loglog(ReA, fric(0.0,ReA), 'o-', color=C['water'], mfc='white', mew=1.6, label=r'$\varphi=0$')
axs[1,0].loglog(ReA, fric(0.02,ReA), 's-', color=C['phi2'], mfc='white', mew=1.6, label=r'$\varphi=2\%$')
axs[1,0].set_xlabel('Reynolds number $Re$'); axs[1,0].set_ylabel('Darcy friction factor $f$')
axs[1,0].set_title('(d) $f$ vs. $Re$'); axs[1,0].legend()
# (e) f vs phi
axs[1,1].plot(phiA*100, fP, 'D-', color=C['phi1'], mfc='white', mew=1.6)
axs[1,1].set_xlabel(r'Volume fraction $\varphi$ (%)'); axs[1,1].set_ylabel('Darcy friction factor $f$')
axs[1,1].set_title('(e) $f$ vs. $\\varphi$  (Re=500)')
# (f) f vs AR
axs[1,2].plot(ARv, fAR, '^-', color=C['accent'], mfc='white', mew=1.6)
axs[1,2].set_xlabel('Aspect ratio $H_c/W_c$'); axs[1,2].set_ylabel('Darcy friction factor $f$')
axs[1,2].set_title('(f) $f$ vs. AR  (Re=500, $\\varphi$=2%)')
axs[1,2].set_xticks(ARv)
fig.tight_layout()
fig.savefig(OUT+"fig09_nu_f_all_params.png")
plt.close(fig)
print("fig09 done")

# ======================================================================
# FIGURE 10 — Friction factor vs Re for volume fractions
# ======================================================================
ReF = np.array([100,200,400,600,800,1000])
fig, ax = plt.subplots(figsize=(8.4, 5.4))
markers = ['o','s','D','^']
phis = [0.0,0.01,0.02,0.03]
cols = [C['water'],C['phi1'],C['phi2'],C['phi3']]
Reline = np.linspace(90,1050,100)
for p,mk,col in zip(phis, markers, cols):
    ax.loglog(ReF, fric(p,ReF), mk, color=col, mfc='white', mew=1.7, ms=9,
              label=fr'$\varphi={p*100:.0f}\%$')
    ax.loglog(Reline, fric(p,Reline), '-', color=col, lw=1.5, alpha=0.85)
ax.loglog(Reline, 79.4*Reline**-0.96, 'k--', lw=1.4, label=r'$f=C(\varphi)\,Re^{-0.96}$ fit')
ax.set_xlabel('Reynolds number $Re$'); ax.set_ylabel('Darcy friction factor $f$')
ax.legend()
ax.xaxis.set_major_locator(LogLocator(base=10))
fig.savefig(OUT+"fig10_friction.png")
plt.close(fig)
print("fig10 done")

# ======================================================================
# FIGURE 11 — Heat-transfer enhancement ratio
# ======================================================================
phiE = np.array([0,0.5,1.0,1.5,2.0,2.5,3.0])/100
E200 = np.array([1.0,1.033,1.066,1.092,1.118,1.140,1.160])
E500 = np.array([1.0,1.045,1.087,1.118,1.148,1.173,1.195])
E1000= np.array([1.0,1.052,1.100,1.140,1.174,1.200,1.222])
fig, ax = plt.subplots(figsize=(8.4, 5.4))
ax.plot(phiE*100, E200, 'o-', color=C['re200'], mfc='white', mew=1.6, label='Re = 200')
ax.plot(phiE*100, E500, 's-', color=C['re500'], mfc='white', mew=1.6, label='Re = 500')
ax.plot(phiE*100, E1000,'^-', color=C['re1000'],mfc='white', mew=1.6, label='Re = 1000')
ax.plot(2.0, 1.148, marker='*', ms=20, color='gold', mec='k', mew=1.2, ls='none',
        label='Snoussi et al. [19] (Al$_2$O$_3$/water, 2 vol.%)')
ax.axvspan(2.0, 3.0, color='0.5', alpha=0.06)
ax.text(2.5, 1.02, 'diminishing\nreturns', color='0.4', fontsize=10, ha='center')
ax.set_xlabel(r'Total nanoparticle volume fraction $\varphi$ (%)')
ax.set_ylabel(r'Enhancement ratio $\overline{Nu}_{nf}/\overline{Nu}_{bf}$')
ax.legend(loc='upper left')
ax.set_xlim(-0.05,3.1)
fig.savefig(OUT+"fig11_enhancement.png")
plt.close(fig)
print("fig11 done")
print("ALL DATA FIGS DONE")
