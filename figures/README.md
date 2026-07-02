# Figures

High-resolution PNGs (300 dpi) for every figure in the paper, plus the scripts
that generate them from the CSV data in `../data/`.

Regenerate everything:

```bash
cd scripts
python make_all_figures.py     # writes the PNGs into this folder
```

| PNG | figure | script |
|-----|--------|--------|
| `fig00_graphical_abstract.png` | Graphical abstract | `make_ga4.py` |
| `fig01_geometry.png`           | Fig 1 unit cell + BCs | `make_fig01b.py` |
| `fig02_architecture.png`       | Fig 2 PINN + neuron model | `make_schematic_figs.py` |
| `fig03_interface.png`          | Fig 3 interface + collocation | `make_fig03.py` |
| `fig05_grid_independence.png`  | Fig 4 grid independence | `make_data_figs.py` |
| `fig06_contours.png`           | Fig 5 velocity/temperature contours | `make_data_figs.py` |
| `fig04_convergence.png`        | Fig 6 training convergence | `make_data_figs.py` |
| `fig07_error_map.png`          | Fig 7 pointwise error | `make_data_figs.py` |
| `fig08_local_nu.png`           | Fig 8 local Nusselt number | `make_fig08c.py` |
| `fig09_nu_f_all_params.png`    | Fig 9 Nu and f vs all parameters | `make_data_figs.py` |
| `fig10_friction.png`           | Fig 10 friction factor | `make_data_figs.py` |
| `fig11_enhancement.png`        | Fig 11 enhancement ratio | `make_data_figs.py` |

Note: the PNG file-name numbers are the original render order; the *figure*
numbers above match the manuscript.
