"""Graphical abstract v4 — isometric MCHS, flow left->right, clean labels."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch
from fig_style import set_style, C
set_style()
OUT = "../"

fig = plt.figure(figsize=(13.6, 5.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
ax.set_xlim(0, 13.6); ax.set_ylim(0, 5.4)

def panel(x, w, ttl, tcol):
    ax.add_patch(FancyBboxPatch((x, 0.55), w, 4.25, boxstyle='round,pad=0.05,rounding_size=0.12',
                 fc='#fbfcfe', ec=tcol, lw=1.6))
    ax.text(x+w/2, 4.52, ttl, ha='center', va='center', fontsize=12.5, fontweight='bold', color=tcol)

def arrow(x0, x1):
    ax.add_patch(FancyArrowPatch((x0, 2.6), (x1, 2.6), arrowstyle='-|>', mutation_scale=26,
                 lw=2.4, color='0.35'))

# ---------------- STAGE 1 ----------------
panel(0.2, 3.7, 'Conjugate heat transfer\nin a hybrid-nanofluid MCHS', C['grey'])
X0, Y0 = 0.92, 1.85
def P(X, Y, Z):
    return (X0 + X*0.82 + Y*0.34, Y0 + Z*0.74 + Y*0.24)
def quad(pts, fc, ec='0.30', a=1.0, z=3, lw=1.0):
    ax.add_patch(Polygon([P(*q) for q in pts], closed=True, fc=fc, ec=ec, alpha=a, zorder=z, lw=lw))

Lx, Wy = 2.5, 1.5
Hb0, Hbt = 0.0, 0.55
zt, zw = Hbt+0.66, Hbt+0.42
SIL_TOP, SIL_FR, SIL_RT = '#dcdcdc', '#c2c2c2', '#cfcfcf'
WAT_TOP, WAT_FR = '#a9d7f2', '#8fc6ec'

quad([(0,0,Hb0),(Lx,0,Hb0),(Lx,0,Hbt),(0,0,Hbt)], SIL_FR, z=2)
quad([(Lx,0,Hb0),(Lx,Wy,Hb0),(Lx,Wy,Hbt),(Lx,0,Hbt)], SIL_RT, z=2)
quad([(0,0,Hbt),(Lx,0,Hbt),(Lx,Wy,Hbt),(0,Wy,Hbt)], SIL_TOP, z=2)

strips = [('chan',0.00,0.30),('fin',0.30,0.50),('chan',0.50,0.80),
          ('fin',0.80,1.00),('chan',1.00,1.30),('fin',1.30,1.50)]
for kind, ya, yb in sorted(strips, key=lambda s: -s[1]):
    if kind == 'fin':
        quad([(0,ya,Hbt),(Lx,ya,Hbt),(Lx,ya,zt),(0,ya,zt)], SIL_FR, z=6)
        quad([(0,ya,zt),(Lx,ya,zt),(Lx,yb,zt),(0,yb,zt)], SIL_TOP, z=6)
        quad([(Lx,ya,Hbt),(Lx,yb,Hbt),(Lx,yb,zt),(Lx,ya,zt)], SIL_RT, z=6)
        quad([(0,ya,Hbt),(0,yb,Hbt),(0,yb,zt),(0,ya,zt)], '#b4b4b4', z=6)
    else:
        quad([(0,ya,zw),(Lx,ya,zw),(Lx,yb,zw),(0,yb,zw)], WAT_TOP, z=5)
        quad([(0,ya,Hbt),(0,yb,Hbt),(0,yb,zw),(0,ya,zw)], WAT_FR, z=5)

# flow arrows through each channel (left -> right)
for _, ya, yb in [s for s in strips if s[0] == 'chan']:
    yc = (ya+yb)/2
    ax.add_patch(FancyArrowPatch(P(-0.28, yc, zw), P(Lx*0.66, yc, zw), arrowstyle='-|>',
                 mutation_scale=13, lw=2.1, color=C['water'], zorder=8))
# nanoparticles in inlet stream
rng = np.random.default_rng(2)
for _ in range(8):
    _, ya, yb = rng.choice([s for s in strips if s[0] == 'chan'])
    yc = float(ya) + (float(yb)-float(ya))*rng.uniform(0.3, 0.7)
    px, py = P(rng.uniform(-0.18, 0.12), yc, zw)
    col = C['phi3'] if rng.random() < 0.5 else '#9a9a9a'
    ax.add_patch(Circle((px, py), 0.045, fc=col, ec='k', lw=0.4, zorder=9))
# clean top flow label (axes coords, above the block)
ax.add_patch(FancyArrowPatch((0.95, 3.5), (2.05, 3.5), arrowstyle='-|>', mutation_scale=16,
             lw=2.0, color=C['water'], zorder=8))
ax.text(1.5, 3.66, 'coolant flow', color=C['water'], fontsize=10, ha='center')
ax.text(0.78, 2.55, 'cold\nin', color=C['water'], fontsize=8.6, ha='center', va='center')
ax.text(3.18, 3.02, 'hot\nout', color='0.45', fontsize=8.6, ha='center', va='center')

# chip heat flux
for xx in np.linspace(0.35, Lx-0.35, 5):
    ax.add_patch(FancyArrowPatch(P(xx, 0.0, Hb0-0.72), P(xx, 0.0, Hb0-0.02), arrowstyle='-|>',
                 mutation_scale=13, lw=2.0, color=C['phi3'], zorder=4))
ax.add_patch(Circle((0.42, 4.05), 0.05, fc=C['phi3'], ec='k', lw=0.4))
ax.text(0.57, 4.05, 'Cu + Al$_2$O$_3$ nanoparticles', fontsize=9.3, color='0.3', ha='left', va='center')
ax.text(*P(Lx*0.5, 0.0, Hbt*0.5), 'Silicon,  $k_s/k_{nf}\\approx220$', fontsize=8.6, color='0.15',
        ha='center', va='center', zorder=3)
ax.text(2.0, 0.8, r"chip heat flux  $q''=10-100$ W cm$^{-2}$", fontsize=9.2, color=C['phi3'], ha='center')

# ---------------- STAGE 2 ----------------
panel(4.5, 4.4, 'Single-network mesh-free PINN\n(adaptive conjugate-interface loss)', C['water'])
lx = np.array([5.15, 6.05, 6.95, 7.85])
lys = [np.linspace(1.6, 3.5, k) for k in (3, 4, 4, 3)]
for a in range(3):
    for y1 in lys[a]:
        for y2 in lys[a+1]:
            ax.plot([lx[a], lx[a+1]], [y1, y2], color='0.72', lw=0.4, zorder=1)
for a, ys in enumerate(lys):
    for y in ys:
        ax.add_patch(Circle((lx[a], y), 0.11, fc='#cfe0f2', ec=C['water'], lw=1.0, zorder=2))
ax.text(6.5, 3.85, r'$(x,y,z,\varphi,Re)\!\rightarrow\!(u,v,w,p,T_f,T_s)$', ha='center', fontsize=10)
ax.text(6.5, 1.25, 'tanh · Adam+L-BFGS · 19,026 params', ha='center', fontsize=9.5, color='0.4')
ax.add_patch(FancyBboxPatch((8.15, 1.8), 0.6, 1.5, boxstyle='round,pad=0.04',
             fc='#f3ecf9', ec=C['accent'], lw=1.2))
xs = np.linspace(0, 1, 40)
ax.plot(8.2+0.5*xs, 3.15-1.15*xs**0.6, color=C['accent'], lw=1.8)
ax.text(8.45, 1.55, r'$\mathcal{L}\!\downarrow$', ha='center', fontsize=11, color=C['accent'])

# ---------------- STAGE 3 ----------------
panel(9.3, 4.1, 'Continuous thermal-hydraulic\ndesign map + fast surrogate', C['phi1'])
res = [('Validated vs. FVM benchmark:  $L_2<1.2\\%$', C['water']),
       ('One trained net spans\n$Re\\,[100,1000],\\ \\varphi\\,[0,3\\%],$ AR$\\,[2,6]$', '0.25'),
       (r'$\overline{Nu}\uparrow14.8\%,\ \Delta T_{base}\downarrow4.6$ K ($\varphi$=2%)', C['phi2']),
       (r'$\approx10^{5}\times$ faster/query (0.03 s vs. 45 min)', C['phi3'])]
yy = 3.6
for txt, col in res:
    ax.add_patch(Circle((9.6, yy), 0.06, fc=col, ec='none'))
    ax.text(9.78, yy, txt, ha='left', va='center', fontsize=9.6, color='0.15')
    yy -= 0.72
arrow(3.95, 4.45)
arrow(8.95, 9.25)
ax.text(6.8, 0.2, 'Physics-informed, mesh-free surrogate for electronics-cooling design, control and optimisation',
        ha='center', fontsize=10, style='italic', color='0.4')
fig.savefig(OUT+"fig00_graphical_abstract.png", dpi=300)
plt.close(fig)
print("GA v4 done")
