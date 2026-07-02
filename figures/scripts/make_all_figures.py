"""Regenerate every figure. PNGs are written to the parent `figures/` folder.
Run:  python make_all_figures.py
"""
import os, sys, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ORDER = [
    "make_data_figs.py",        # Fig 4,5,6,7,9,10,11 (data plots)
    "make_schematic_figs.py",   # Fig 2 architecture (+ provisional 1,3)
    "make_ga4.py",              # graphical abstract (fig00)
    "make_fig01b.py",           # Fig 1 geometry (final)
    "make_fig03.py",            # Fig 3 interface/collocation (final)
    "make_fig08c.py",           # Fig 8 local Nusselt (final)
]
for s in ORDER:
    print("=> running", s)
    subprocess.run([sys.executable, s], check=True)
print("done - figures written to ../")
