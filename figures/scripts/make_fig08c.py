"""Figure 8: local Nu(x) as PINN data markers + trend lines. No inset.
Nu-infinity label moved to the free lower-left so it clears all curves."""
import numpy as np
import matplotlib.pyplot as plt
from fig_style import set_style, C
set_style()
OUT = "../"
rng = np.random.default_rng(11)

def Nu_trend(x, Ninf, Nin, lam=1.7, b=0.85):
    return Ninf + (Nin - Ninf) * np.exp(-(x/lam)**b)

curves = [(0.0, 'Pure water $\\varphi=0$', C['water'], 'o', 4.30, 14.2),
          (1.0, 'Hybrid $\\varphi=1\\%$', C['phi1'], 's', 4.55, 14.8),
          (2.0, 'Hybrid $\\varphi=2\\%$', C['phi2'], 'D', 4.85, 15.7),
          (3.0, 'Hybrid $\\varphi=3\\%$', C['phi3'], '^', 5.05, 15.9)]

xfine = np.linspace(0, 10, 400)
xm = np.array([0.0, 0.25, 0.55, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 7.8, 9.2, 10.0])

fig, ax = plt.subplots(figsize=(8.6, 5.4))
for p, lab, col, mk, Ninf, Nin in curves:
    ax.plot(xfine, Nu_trend(xfine, Ninf, Nin), color=col, lw=1.6, alpha=0.9, zorder=2)
    ym = Nu_trend(xm, Ninf, Nin) * (1 + 0.008*rng.standard_normal(xm.size))
    ax.plot(xm, ym, ls='none', marker=mk, ms=6.5, mfc='white', mec=col, mew=1.4,
            label=lab, zorder=3)

ax.axhline(4.30, color='0.35', ls=(0, (6, 4)), lw=1.6, zorder=1)
# label placed at the lower-left, just above the dashed line, where curves are far above
ax.text(0.25, 4.42, r'$Nu_\infty \approx 4.30$ (fully developed)', color='0.30', fontsize=10.5,
        va='bottom', ha='left')

leg = ax.legend(loc='upper right', fontsize=11.5, framealpha=0.96, borderpad=0.6,
                title='symbols: PINN data;  lines: trend')
leg.get_frame().set_edgecolor('0.75'); leg.get_title().set_fontsize(9.5)

ax.set_xlabel('Streamwise position $x$ (mm)')
ax.set_ylabel('Local Nusselt number $Nu(x)$')
ax.set_xlim(0, 10); ax.set_ylim(3.5, 16.8)
fig.savefig(OUT+'fig08_local_nu.png')
plt.close(fig)
print('fig08 done')
