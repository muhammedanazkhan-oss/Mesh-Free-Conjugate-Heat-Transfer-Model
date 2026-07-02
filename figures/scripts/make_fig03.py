"""Figure 3 rebuilt: (a) collocation SAMPLING points on the true channel-in-silicon
cross-section, drawn as +/x markers (clearly not nanoparticles); (b) interface T(n)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from fig_style import set_style, C
set_style()
OUT = "../"
rng = np.random.default_rng(7)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.9),
                               gridspec_kw={'width_ratios': [1.3, 1]})

# =================== (a) collocation points on cross-section ===================
axL.set_aspect('equal'); axL.set_xlim(0, 10); axL.set_ylim(0, 8)
# silicon block with an enclosed nanofluid channel (matches Fig 1b)
axL.add_patch(Rectangle((0, 0), 10, 8, fc='#e2e2e2', ec='0.3', lw=1.4))          # silicon
chx0, chx1, chy0, chy1 = 3.6, 6.4, 2.6, 6.4                                       # channel
axL.add_patch(Rectangle((chx0, chy0), chx1-chx0, chy1-chy0, fc='#bfe0f4', ec='0.3', lw=1.4))

def in_channel(x, y):
    return (chx0 < x) & (x < chx1) & (chy0 < y) & (y < chy1)

# solid-interior collocation points (grey +), only in silicon
sx, sy = [], []
while len(sx) < 190:
    x, y = rng.uniform(0.25, 9.75), rng.uniform(0.25, 7.75)
    if not in_channel(x, y) and not (chx0-0.3 < x < chx1+0.3 and chy0-0.3 < y < chy1+0.3 and in_channel(x, y)):
        if not in_channel(x, y):
            sx.append(x); sy.append(y)
axL.scatter(sx, sy, marker='+', s=22, color='0.45', linewidths=0.9, zorder=3,
            label='Solid-interior collocation (~30%)')
# fluid-interior collocation points (blue x), only in the channel
fx = rng.uniform(chx0+0.18, chx1-0.18, 150)
fy = rng.uniform(chy0+0.18, chy1-0.18, 150)
axL.scatter(fx, fy, marker='x', s=20, color=C['water'], linewidths=0.9, zorder=3,
            label='Fluid-interior collocation (~70%)')
# conjugate-interface points (red circles) on the 4 channel walls
tt = np.linspace(chy0, chy1, 16)
ss = np.linspace(chx0, chx1, 14)
ix = np.concatenate([np.full(16, chx0), np.full(16, chx1), ss, ss])
iy = np.concatenate([tt, tt, np.full(14, chy0), np.full(14, chy1)])
axL.scatter(ix, iy, marker='o', s=20, color=C['phi3'], edgecolor='k', lw=0.3, zorder=4,
            label='Conjugate-interface points (N=1200)')
# boundary points (green) on the outer edges
bx = np.concatenate([np.linspace(0, 10, 26), np.full(20, 0), np.full(20, 10), np.linspace(0, 10, 26)])
by = np.concatenate([np.full(26, 0), np.linspace(0, 8, 20), np.linspace(0, 8, 20), np.full(26, 8)])
axL.scatter(bx, by, marker='o', s=12, color=C['phi1'], zorder=4, label='Boundary points (N=1600)')

# region labels + normal
axL.text(5.0, 7.15, 'Cu-Al$_2$O$_3$/water nanofluid  ($T_f,\\,k_{nf}$)', ha='center', fontsize=9.6,
         color='#0d3b60', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
axL.text(1.7, 1.3, 'Silicon  ($T_s,\\,k_s$)', ha='center', fontsize=9.6, color='0.25',
         bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
axL.annotate('$\\mathbf{n}$', xy=(chx0-0.55, 4.5), xytext=(chx0-0.05, 4.5),
             arrowprops=dict(arrowstyle='->', color='k', lw=1.3), fontsize=12, va='center')
axL.annotate('interface $\\Gamma_{fs}$', xy=(chx1, 3.4), xytext=(7.7, 2.7),
             arrowprops=dict(arrowstyle='->', color=C['phi3'], lw=1.0), fontsize=9.5, color=C['phi3'])
axL.set_xticks([]); axL.set_yticks([])
axL.set_title('(a) Stratified collocation (sampling) points', fontsize=12.5)
axL.text(5.0, -0.55, 'markers denote PINN collocation/sampling locations — not nanoparticles',
         ha='center', fontsize=8.6, style='italic', color='0.45')
axL.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=9.2, framealpha=0.95)

# =================== (b) interface temperature / flux continuity ===================
axR.set_xlim(-1, 1); axR.set_ylim(0, 1)
n = np.linspace(-1, 1, 400)
T = np.where(n < 0, 0.55 + 0.16*n, 0.55 + 0.42*n)
axR.axvspan(-1, 0, color='#dedede', alpha=0.6)
axR.axvspan(0, 1, color='#bfe0f4', alpha=0.6)
axR.plot(n, T, color='k', lw=2.6, zorder=4)
axR.axvline(0, color='0.3', ls='--', lw=1.2)
axR.annotate('$T_s=T_f$  (Eq. 14)', xy=(0, 0.55), xytext=(-0.4, 0.85),
             arrowprops=dict(arrowstyle='->', color='k'), ha='center', fontsize=10.5)
axR.annotate('', xy=(0.02, 0.60), xytext=(0.35, 0.42), arrowprops=dict(arrowstyle='->', color='k'))
axR.text(0.42, 0.34, r'$k_s\dfrac{\partial T_s}{\partial n}=k_{nf}\dfrac{\partial T_f}{\partial n}$'+'\n(Eq. 15)',
         fontsize=10.5, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7'))
axR.set_xlabel('Wall-normal coordinate $n$'); axR.set_ylabel('Temperature $T(n)$ (scaled)')
axR.set_yticks([]); axR.set_xticks([-1, 0, 1]); axR.set_xticklabels(['solid', '$\\Gamma_{fs}$', 'fluid'])
axR.set_title('(b) Temperature and flux continuity at interface', fontsize=12.5)

fig.savefig(OUT+"fig03_interface.png")
plt.close(fig)
print("fig03 rebuilt")
