"""Shared plotting style for all manuscript figures (professional, high-res)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def set_style():
    mpl.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'axes.linewidth': 1.1,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'legend.fontsize': 11.5,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '0.7',
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': ':',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

# consistent colour palette
C = {
    'water':  '#1f4e79',   # deep blue
    'phi1':   '#2e8b57',   # green
    'phi2':   '#e8820c',   # orange
    'phi3':   '#c0392b',   # red
    'accent': '#6a3d9a',   # purple
    'grey':   '#555555',
    're200':  '#1f77b4',
    're500':  '#e8820c',
    're1000': '#c0392b',
}
