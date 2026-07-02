"""Schematic figures: (Fig2) PINN architecture + single-neuron model,
(Fig1) MCHS geometry + BCs, (Fig3) conjugate interface + collocation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon
from fig_style import set_style, C
set_style()
OUT = "../"

# ======================================================================
# FIGURE 2 — PINN architecture + single-neuron mathematical model
# ======================================================================
fig = plt.figure(figsize=(13.2, 8.6))
gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.16)
ax = fig.add_subplot(gs[0]); ax.axis('off')
ax.set_xlim(0, 13.2); ax.set_ylim(1.2, 6.6)

def node(x, y, r=0.16, fc='#cfe0f2', ec='#22508a'):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.3, zorder=3))

inx = 1.1
in_labels = ['$x$', '$y$', '$z$', '$\\varphi$', '$Re$']
in_y = np.linspace(2.3, 5.4, 5)
out_labels = ['$u$', '$v$', '$w$', '$p$', '$T_f$', '$T_s$']
out_y = np.linspace(2.05, 5.55, 6)
outx = 12.1
hidden_x = np.linspace(3.2, 10.0, 6)
hid_y_full = np.linspace(1.9, 5.7, 6)

for iy in in_y:
    for hy in hid_y_full:
        ax.plot([inx+0.16, hidden_x[0]-0.16], [iy, hy], color='0.75', lw=0.4, zorder=1)
for k in range(5):
    for y1 in hid_y_full:
        for y2 in hid_y_full:
            ax.plot([hidden_x[k]+0.16, hidden_x[k+1]-0.16], [y1, y2], color='0.8', lw=0.3, zorder=1)
for hy in hid_y_full:
    for oy in out_y:
        ax.plot([hidden_x[-1]+0.16, outx-0.16], [hy, oy], color='0.75', lw=0.4, zorder=1)

for lbl, y in zip(in_labels, in_y):
    node(inx, y, r=0.20, fc='#ffe6bf', ec='#c8791b')
    ax.text(inx, y, lbl, ha='center', va='center', fontsize=12, zorder=4)
ax.text(inx, 6.05, 'Input layer\n(5)', ha='center', fontsize=12, fontweight='bold')
for k, hx in enumerate(hidden_x):
    for y in hid_y_full:
        node(hx, y)
    ax.text(hx, 1.55, f'$h_{k+1}$', ha='center', fontsize=10, color='0.35')
ax.text((hidden_x[0]+hidden_x[-1])/2, 6.30, '6 hidden layers x 60 neurons  (tanh activation)',
        ha='center', fontsize=13, fontweight='bold')
for hx in hidden_x:
    ax.text(hx, 3.8, '$\\vdots$', ha='center', va='center', fontsize=14, color='#22508a')
for lbl, y in zip(out_labels, out_y):
    node(outx, y, r=0.20, fc='#cdeecd', ec='#2e8b57')
    ax.text(outx, y, lbl, ha='center', va='center', fontsize=11, zorder=4)
ax.text(outx, 6.05, 'Output layer\n(6)', ha='center', fontsize=12, fontweight='bold')

# --- lower panel: single-neuron model ---
axn = fig.add_subplot(gs[1]); axn.axis('off')
axn.set_xlim(0, 13.2); axn.set_ylim(0, 3.0)
ny = 1.55
nsrc_x = 1.3
nsrc_y = [0.85, 1.55, 2.25]
sum_x, act_x, out_x = 3.6, 5.3, 6.9
for i, (yy, lab, wl) in enumerate(zip(nsrc_y, ['$x_1$', '$x_2$', '$x_n$'], ['$w_1$', '$w_2$', '$w_n$'])):
    axn.add_patch(Circle((nsrc_x, yy), 0.17, fc='#ffe6bf', ec='#c8791b', lw=1.2))
    axn.text(nsrc_x, yy, lab, ha='center', va='center', fontsize=10)
    axn.annotate('', xy=(sum_x-0.3, ny), xytext=(nsrc_x+0.17, yy),
                 arrowprops=dict(arrowstyle='-', color='0.5', lw=1.0))
    axn.text(nsrc_x+0.95, yy+0.12+(1-i)*0.02, wl, fontsize=10, color=C['phi3'])
axn.text(nsrc_x, 0.3, 'inputs from\nprevious layer', ha='center', fontsize=9, color='0.45')
axn.add_patch(Circle((sum_x, ny), 0.3, fc='#eef3fb', ec='#22508a', lw=1.4))
axn.text(sum_x, ny, r'$\Sigma$', ha='center', va='center', fontsize=16)
axn.annotate('', xy=(sum_x, ny-0.3), xytext=(sum_x, ny-0.75),
             arrowprops=dict(arrowstyle='->', color='0.5', lw=1.0))
axn.text(sum_x, ny-0.95, 'bias $b$', ha='center', fontsize=9.5, color='0.4')
axn.add_patch(FancyBboxPatch((act_x-0.45, ny-0.37), 0.9, 0.74,
              boxstyle='round,pad=0.02', fc='#f3ecf9', ec=C['accent'], lw=1.4))
tt = np.linspace(-0.34, 0.34, 50); axn.plot(act_x+tt, ny+0.24*np.tanh(tt*9), color=C['accent'], lw=1.6)
axn.text(act_x, ny-0.6, 'tanh', ha='center', fontsize=9.5, color=C['accent'])
axn.annotate('', xy=(act_x-0.45, ny), xytext=(sum_x+0.3, ny),
             arrowprops=dict(arrowstyle='->', color='0.5', lw=1.1))
axn.add_patch(Circle((out_x, ny), 0.2, fc='#cdeecd', ec='#2e8b57', lw=1.3))
axn.text(out_x, ny, '$a$', ha='center', va='center', fontsize=11)
axn.annotate('', xy=(out_x-0.2, ny), xytext=(act_x+0.45, ny),
             arrowprops=dict(arrowstyle='->', color='0.5', lw=1.1))
axn.text(9.9, ny, r'$a=\tanh\left(\sum_{i=1}^{n} w_i x_i + b\right)$',
         fontsize=17, va='center', ha='center',
         bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.7'))
axn.text(0.2, 2.85, 'Mathematical model of a single neuron', fontsize=12.5,
         fontweight='bold', color='0.2')
fig.savefig(OUT+"fig02_architecture.png")
plt.close(fig)
print("fig02 done")

# ======================================================================
# FIGURE 1 — MCHS geometry (isometric + cross-section) with BCs
# ======================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.4, 6.2),
                               gridspec_kw={'width_ratios': [1.25, 1]})
for a in (axA, axB):
    a.axis('off'); a.set_aspect('equal')

axA.set_xlim(-1.2, 8.8); axA.set_ylim(-1.8, 6.8)
dx, dy = 1.7, 0.95
def prism(x, y, w, h, fc, ec='0.25', a=1.0, depth=True, lw=1.2):
    axA.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, alpha=a, zorder=3))
    if depth:
        top = Polygon([(x, y+h), (x+dx, y+h+dy), (x+dx+w, y+h+dy), (x+w, y+h)],
                      closed=True, fc=fc, ec=ec, lw=lw, alpha=a*0.82, zorder=2)
        side = Polygon([(x+w, y), (x+w+dx, y+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
                       closed=True, fc=fc, ec=ec, lw=lw, alpha=a*0.65, zorder=2)
        axA.add_patch(top); axA.add_patch(side)
prism(0, 0, 5.2, 0.9, '#c9c9c9')
prism(0, 0.9, 0.7, 3.0, '#d8d8d8')
prism(4.5, 0.9, 0.7, 3.0, '#d8d8d8')
prism(0.7, 0.9, 3.8, 3.0, '#8fc6ec', a=0.9)
prism(0, 3.9, 5.2, 0.7, '#d8d8d8')
axA.text(2.6, 2.4, 'Nanofluid\nchannel', ha='center', va='center', fontsize=11,
         color='#0d3b60', zorder=6)
axA.annotate('Silicon fin', xy=(4.85, 2.4), xytext=(6.5, 1.0),
             arrowprops=dict(arrowstyle='->', color='0.4', lw=1.0),
             ha='left', va='center', fontsize=9.5, color='0.3', zorder=6)
for yy in np.linspace(1.3, 3.5, 4):
    axA.annotate('', xy=(0.7, yy), xytext=(-0.9, yy-0.2),
                 arrowprops=dict(arrowstyle='->', color=C['water'], lw=2))
axA.text(-1.1, 4.0, 'Inlet\n$u_{in},\\,T_{in}$', ha='left', fontsize=10, color=C['water'])
for yy in np.linspace(1.3, 3.5, 4):
    axA.annotate('', xy=(5.2+dx+0.9, yy+dy+0.2), xytext=(4.5+dx, yy+dy),
                 arrowprops=dict(arrowstyle='->', color=C['water'], lw=2))
axA.text(7.2, 4.9, 'Outlet\n$p=0$', ha='left', fontsize=10, color=C['water'])
for xx in np.linspace(0.4, 4.8, 7):
    axA.annotate('', xy=(xx, 0.0), xytext=(xx, -1.1),
                 arrowprops=dict(arrowstyle='->', color=C['phi3'], lw=2))
axA.text(2.6, -1.55, r"$q''=10-100$ W cm$^{-2}$ (chip heat flux)", ha='center',
         fontsize=10, color=C['phi3'])
axA.text(2.6, 6.1, 'Adiabatic top cover', ha='center', fontsize=10, color='0.3')
axA.annotate('', xy=(2.6, 4.6), xytext=(2.6, 5.9), arrowprops=dict(arrowstyle='-', color='0.5', lw=1))
axA.text(-0.3, 5.4, 'Periodic /\nsymmetry', ha='center', fontsize=9, color='0.4')
axA.text(6.0, 6.1, 'Periodic /\nsymmetry', ha='center', fontsize=9, color='0.4')
ox, oy = -0.9, -0.4
axA.annotate('', xy=(ox+0.9, oy), xytext=(ox, oy), arrowprops=dict(arrowstyle='->', color='k', lw=1.4))
axA.annotate('', xy=(ox, oy+0.9), xytext=(ox, oy), arrowprops=dict(arrowstyle='->', color='k', lw=1.4))
axA.annotate('', xy=(ox+0.5, oy+0.32), xytext=(ox, oy), arrowprops=dict(arrowstyle='->', color='k', lw=1.4))
axA.text(ox+1.0, oy, '$x$', fontsize=11); axA.text(ox-0.05, oy+1.0, '$z$', fontsize=11)
axA.text(ox+0.55, oy+0.42, '$y$', fontsize=11)
axA.set_title('(a) Three-dimensional unit cell and boundary conditions', fontsize=12.5)

axB.set_xlim(-2.0, 6.8); axB.set_ylim(-1.9, 7.6)
Wtot, Htot = 4.0, 4.8
axB.add_patch(Rectangle((0, 0), Wtot, Htot, fc='#d8d8d8', ec='0.25', lw=1.3, hatch='/////'))
cw, ch, cx, cbase = 2.0, 3.2, 1.0, 0.9
axB.add_patch(Rectangle((cx, cbase), cw, ch, fc='#8fc6ec', ec='0.25', lw=1.3))
axB.text(cx+cw/2, cbase+ch/2, 'Nanofluid', ha='center', va='center', fontsize=11, color='#0d3b60')
axB.text(0.15, 0.42, 'Silicon substrate', ha='left', fontsize=9.5, color='0.15',
         bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
axB.text(0.15, -0.12, '$k_s$=148 W m$^{-1}$K$^{-1}$', ha='left', fontsize=8.5, color='0.3',
         bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
def dim_h(x0, x1, y, txt, off=0.22, fs=9.5):
    axB.annotate('', xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle='<->', color='k', lw=1.0))
    axB.text((x0+x1)/2, y+off, txt, ha='center', fontsize=fs)
def dim_v(x, y0, y1, txt, off=0.24, fs=9.5):
    axB.annotate('', xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle='<->', color='k', lw=1.0))
    axB.text(x+off, (y0+y1)/2, txt, va='center', ha='left', fontsize=fs, rotation=90)
dim_h(0, Wtot, Htot+1.25, r'$W_c+2W_f=0.40$ mm', off=0.18)
yrow = Htot+0.4
dim_h(0.02, cx, yrow, r'$W_f$'+'\n0.10', off=0.12, fs=9)
dim_h(cx, cx+cw, yrow, r'$W_c$'+'\n0.20', off=0.12, fs=9)
dim_h(cx+cw, Wtot-0.02, yrow, r'$W_f$'+'\n0.10', off=0.12, fs=9)
dim_v(-0.8, 0, Htot, r'1.20 mm', off=-0.6)
dim_v(Wtot+0.7, cbase+ch, Htot, r'$H_t$=0.10')
dim_v(Wtot+0.7, cbase, cbase+ch, r'$H_c$=0.80')
dim_v(Wtot+0.7, 0, cbase, r'$H_b$=0.30')
for xx in np.linspace(0.4, Wtot-0.4, 7):
    axB.annotate('', xy=(xx, 0), xytext=(xx, -1.1), arrowprops=dict(arrowstyle='->', color=C['phi3'], lw=1.8))
axB.text(Wtot/2, -1.6, r"$q''$ base heat flux", ha='center', fontsize=10, color=C['phi3'])
axB.set_title('(b) $y$-$z$ cross-section (aspect ratio $H_c/W_c=4$)', fontsize=12.5)
fig.savefig(OUT+"fig01_geometry.png")
plt.close(fig)
print("fig01 done")

# ======================================================================
# FIGURE 3 — Conjugate interface treatment + collocation points
# ======================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.8),
                               gridspec_kw={'width_ratios': [1.35, 1]})
axL.set_aspect('equal'); axL.set_xlim(0, 10); axL.set_ylim(0, 8)
axL.add_patch(Rectangle((0, 0), 10, 8, fc='#eef4fb', ec='none'))
solid = Polygon([(0, 0), (10, 0), (10, 3.4), (4.2, 3.4), (4.2, 8), (0, 8)], closed=True,
                fc='#dedede', ec='0.3', lw=1.4)
axL.add_patch(solid)
axL.add_patch(Rectangle((4.2, 3.4), 5.8, 4.6, fc='#bfe0f4', ec='0.3', lw=1.4))
rng = np.random.default_rng(7)
fx = rng.uniform(4.35, 9.9, 300); fy = rng.uniform(3.55, 7.9, 300)
axL.scatter(fx, fy, s=7, color=C['water'], alpha=0.8, zorder=3, label='Fluid interior LHS (~70%)')
sx = []; sy = []
while len(sx) < 230:
    px, py = rng.uniform(0.1, 9.9), rng.uniform(0.1, 7.9)
    if (px < 4.2 and py < 8) or (py < 3.4):
        sx.append(px); sy.append(py)
axL.scatter(sx, sy, s=7, color='0.45', alpha=0.8, zorder=3, label='Solid interior LHS (~30%)')
ix = np.concatenate([np.full(20, 4.2), np.linspace(4.2, 10, 22)])
iy = np.concatenate([np.linspace(3.4, 7.9, 20), np.full(22, 3.4)])
axL.scatter(ix, iy, s=22, color=C['phi3'], zorder=4, edgecolor='k', lw=0.3, label='Conjugate interface (N=1200)')
bx = np.concatenate([np.linspace(0, 10, 26), np.full(20, 0), np.full(20, 10), np.linspace(0, 10, 26)])
by = np.concatenate([np.full(26, 0), np.linspace(0, 8, 20), np.linspace(0, 8, 20), np.full(26, 8)])
axL.scatter(bx, by, s=10, color=C['phi1'], zorder=4, label='Boundary (N=1600)')
axL.text(7.0, 5.9, 'Cu-Al$_2$O$_3$/water\nnanofluid\n$T_f,\\,k_{nf}$', ha='center', fontsize=10, color='#0d3b60')
axL.text(2.0, 1.9, 'Silicon\n$T_s,\\,k_s$', ha='center', fontsize=10, color='0.3')
axL.annotate('$\\mathbf{n}$', xy=(5.2, 3.4), xytext=(5.2, 4.1),
             arrowprops=dict(arrowstyle='->', color='k', lw=1.4), fontsize=12, ha='center')
axL.text(6.6, 3.02, 'interface $\\Gamma_{fs}$', fontsize=9.5, color=C['phi3'])
axL.set_xticks([]); axL.set_yticks([])
axL.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9.5, framealpha=0.95)
axL.set_title('(a) Stratified collocation sampling', fontsize=12.5)

axR.set_xlim(-1, 1); axR.set_ylim(0, 1)
n = np.linspace(-1, 1, 400)
Ts = 0.55 + 0.16*n
Tf = 0.55 + 0.42*n
T = np.where(n < 0, Ts, Tf)
axR.axvspan(-1, 0, color='#dedede', alpha=0.6)
axR.axvspan(0, 1, color='#bfe0f4', alpha=0.6)
axR.plot(n, T, color='k', lw=2.6, zorder=4)
axR.axvline(0, color='0.3', ls='--', lw=1.2)
axR.text(-0.5, 0.1, 'Silicon', ha='center', fontsize=10)
axR.text(0.5, 0.1, 'Nanofluid', ha='center', fontsize=10)
axR.annotate('$T_s=T_f$  (Eq. 12)', xy=(0, 0.55), xytext=(-0.35, 0.85),
             arrowprops=dict(arrowstyle='->', color='k'), ha='center', fontsize=10.5)
axR.annotate('', xy=(0.02, 0.60), xytext=(0.35, 0.42), arrowprops=dict(arrowstyle='->', color='k'))
axR.text(0.42, 0.34, r'$k_s\dfrac{\partial T_s}{\partial n}=k_{nf}\dfrac{\partial T_f}{\partial n}$'+'\n(Eq. 13)',
         fontsize=10.5, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7'))
axR.set_xlabel('Wall-normal coordinate $n$'); axR.set_ylabel('Temperature $T(n)$ (scaled)')
axR.set_yticks([]); axR.set_xticks([-1, 0, 1]); axR.set_xticklabels(['solid', '$\\Gamma_{fs}$', 'fluid'])
axR.set_title('(b) Temperature and flux continuity at interface', fontsize=12.5)
fig.savefig(OUT+"fig03_interface.png")
plt.close(fig)
print("fig03 done")
print("ALL SCHEMATIC FIGS DONE")
