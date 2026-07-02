"""Figure 1 rebuilt v2 — same corrected geometry, labels distributed cleanly."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from fig_style import set_style, C
set_style()
OUT = "../"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.6, 6.4),
                               gridspec_kw={'width_ratios': [1.35, 1]})
for a in (axA, axB):
    a.axis('off'); a.set_aspect('equal')

# =================== (a) isometric: flow along x (left -> right) ===================
axA.set_xlim(-2.7, 9.2); axA.set_ylim(-2.4, 5.4)
def P(X, Y, Z):
    return (X*0.82 + Y*0.42, Z*0.80 + Y*0.30)
def quad(pts, fc, ec='0.30', a=1.0, z=3, lw=1.1):
    axA.add_patch(Polygon([P(*q) for q in pts], closed=True, fc=fc, ec=ec, alpha=a, zorder=z, lw=lw))

Lx = 5.5
yF0, yF1 = 0.00, 0.45
yC0, yC1 = 0.45, 1.05
yB0, yB1 = 1.05, 1.50
zb0, zb1 = 0.00, 0.55
zm0, zm1 = 0.55, 1.65
zt0, zt1 = 1.65, 2.05
GT, GF, GR = '#dcdcdc', '#c4c4c4', '#d0d0d0'
WT, WF, WR = '#a9d7f2', '#8fc6ec', '#7ab8e6'
LEAD = dict(arrowstyle='-', color='0.45', lw=0.9)   # clean leader lines

# base slab
quad([(0,yF0,zb1),(Lx,yF0,zb1),(Lx,yB1,zb1),(0,yB1,zb1)], GT, z=2)
quad([(0,yF0,zb0),(Lx,yF0,zb0),(Lx,yF0,zb1),(0,yF0,zb1)], GF, z=2)
quad([(Lx,yF0,zb0),(Lx,yB1,zb0),(Lx,yB1,zb1),(Lx,yF0,zb1)], GR, z=2)
# back fin
quad([(0,yB0,zb1),(Lx,yB0,zb1),(Lx,yB1,zb1),(0,yB1,zb1)], GT, z=2)
quad([(0,yB0,zm0),(Lx,yB0,zm0),(Lx,yB0,zm1),(0,yB0,zm1)], GF, z=3)
quad([(0,yB0,zm1),(Lx,yB0,zm1),(Lx,yB1,zm1),(0,yB1,zm1)], GT, z=3)
quad([(Lx,yB0,zm0),(Lx,yB1,zm0),(Lx,yB1,zm1),(Lx,yB0,zm1)], GR, z=3)
# channel fluid
quad([(0,yC0,zm1),(Lx,yC0,zm1),(Lx,yC1,zm1),(0,yC1,zm1)], WT, z=4)
quad([(0,yC0,zm0),(Lx,yC0,zm0),(Lx,yC0,zm1),(0,yC0,zm1)], WF, z=4)
quad([(Lx,yC0,zm0),(Lx,yC1,zm0),(Lx,yC1,zm1),(Lx,yC0,zm1)], WR, z=4)
# front fin (translucent)
quad([(0,yF0,zm0),(Lx,yF0,zm0),(Lx,yF0,zm1),(0,yF0,zm1)], GF, a=0.32, z=7)
quad([(0,yF0,zm1),(Lx,yF0,zm1),(Lx,yF1,zm1),(0,yF1,zm1)], GT, a=0.32, z=7)
quad([(Lx,yF0,zm0),(Lx,yF1,zm0),(Lx,yF1,zm1),(Lx,yF0,zm1)], GR, a=0.32, z=7)
# cover (translucent)
quad([(0,yF0,zt0),(Lx,yF0,zt0),(Lx,yF0,zt1),(0,yF0,zt1)], GF, a=0.32, z=7)
quad([(0,yF0,zt1),(Lx,yF0,zt1),(Lx,yB1,zt1),(0,yB1,zt1)], GT, a=0.32, z=7)
quad([(Lx,yF0,zt0),(Lx,yB1,zt0),(Lx,yB1,zt1),(Lx,yF0,zt1)], GR, a=0.32, z=7)

# ---- coolant flow along channel length ----
yc = 0.75
for zc in (0.85, 1.15, 1.45):
    axA.add_patch(FancyArrowPatch(P(-0.9, yc, zc), P(0.35, yc, zc), arrowstyle='-|>',
                  mutation_scale=13, lw=2.0, color=C['water'], zorder=9))
    axA.add_patch(FancyArrowPatch(P(Lx-0.2, yc, zc), P(Lx+1.0, yc, zc), arrowstyle='-|>',
                  mutation_scale=13, lw=2.0, color=C['water'], zorder=9))
axA.add_patch(FancyArrowPatch(P(0.5, yc, 1.15), P(Lx-0.4, yc, 1.15), arrowstyle='-|>',
              mutation_scale=16, lw=2.4, color='#0d5aa7', zorder=10))
axA.text(*P(Lx*0.5, yc, 1.62), 'coolant flow ($x$)', color='#0d5aa7', fontsize=10,
         ha='center', va='center', zorder=11)

# ---- chip heat flux ----
for xx in np.linspace(0.4, Lx-0.4, 7):
    axA.add_patch(FancyArrowPatch(P(xx, yF0, -0.72), P(xx, yF0, -0.02), arrowstyle='-|>',
                  mutation_scale=12, lw=1.9, color=C['phi3'], zorder=5))

# =================== labels: distributed, horizontal, non-crossing ===================
# left column
axA.annotate('Inlet:  $u_{in},\\,T_{in}$', xy=P(-0.85, yc, 1.15), xytext=(-2.6, 2.05),
             arrowprops=LEAD, fontsize=10, color=C['water'], ha='left', va='center')
axA.annotate('Nanofluid channel', xy=P(1.5, yC0, 1.15), xytext=(-2.6, 3.3),
             arrowprops=LEAD, fontsize=10, color='#0d3b60', ha='left', va='center')
# top
axA.annotate('Adiabatic top cover', xy=P(2.4, 0.2, zt1), xytext=(2.55, 4.95),
             arrowprops=LEAD, fontsize=10, color='0.3', ha='center', va='center')
# right column
axA.annotate('Outlet:  $p=0$', xy=P(Lx+0.9, yc, 1.15), xytext=(6.9, 2.05),
             arrowprops=LEAD, fontsize=10, color=C['water'], ha='left', va='center')
axA.annotate('Silicon fins\n(periodic / symmetry planes)', xy=P(4.7, yB0, 1.05), xytext=(6.4, 3.5),
             arrowprops=LEAD, fontsize=9.6, color='0.3', ha='left', va='center')
# base + heat flux
axA.text(*P(Lx*0.5, yF0, zb1*0.5), 'Silicon,  $k_s$ = 148 W m$^{-1}$K$^{-1}$', fontsize=8.8,
         color='0.12', ha='center', va='center', zorder=6)
axA.text(*P(Lx*0.5, yF0, -1.12), r"$q'' = 10-100$ W cm$^{-2}$  (chip heat flux)", color=C['phi3'],
         fontsize=10, ha='center')
# coordinate triad
ox, oy = -2.35, -1.75
axA.annotate('', xy=(ox+0.95, oy), xytext=(ox, oy), arrowprops=dict(arrowstyle='->', color='k', lw=1.4))
axA.annotate('', xy=(ox, oy+0.95), xytext=(ox, oy), arrowprops=dict(arrowstyle='->', color='k', lw=1.4))
axA.annotate('', xy=(ox+0.44, oy+0.32), xytext=(ox, oy), arrowprops=dict(arrowstyle='->', color='k', lw=1.4))
axA.text(ox+1.05, oy, '$x$', fontsize=10, va='center')
axA.text(ox-0.05, oy+1.05, '$z$', fontsize=10, ha='center')
axA.text(ox+0.52, oy+0.42, '$y$', fontsize=10)
axA.set_title('(a) Three-dimensional unit cell and boundary conditions', fontsize=12.5, y=1.0)

# =================== (b) y-z cross-section ===================
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
axB.set_title('(b) $y$-$z$ cross-section (aspect ratio $H_c/W_c=4$)', fontsize=12.5, y=1.0)

fig.savefig(OUT+"fig01_geometry.png")
plt.close(fig)
print("fig01 v2 done")
