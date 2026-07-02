# PINN-MCHS: mesh-free conjugate heat transfer in a hybrid-nanofluid microchannel heat sink

Reference code, result data and figure scripts accompanying the paper:

> **A Physics-Informed Neural Network Framework for Mesh-Free Conjugate Heat
> Transfer Modelling in Hybrid Nanofluid Microchannel Heat Sinks.**

A single fully connected physics-informed neural network (PINN) maps
`(x, y, z, phi, Re) -> (u, v, w, p, Tf, Ts)` and solves the three-dimensional,
steady, laminar **conjugate** (fluid + solid) heat-transfer problem in a
rectangular silicon microchannel heat sink cooled by a Cu-Al2O3/water hybrid
nanofluid, **without a mesh**. Temperature continuity at the solid-fluid
interface is enforced by reading the two temperature outputs as one continuous
field; heat-flux continuity is imposed through a dedicated interface-loss term
whose weight is rescaled every 500 iterations by an inverse-residual rule.
Training is Adam followed by L-BFGS.

## Repository structure

```
pinn-mchs-hybrid-nanofluid/
├── README.md
├── LICENSE                 MIT
├── CITATION.cff            software/dataset citation (add Zenodo DOI)
├── .zenodo.json            Zenodo deposition metadata
├── requirements.txt
├── src/
│   ├── properties.py       effective nanofluid properties (mixture, Brinkman, sequential Hamilton-Crosser)
│   ├── pinn_mchs.py        single-network conjugate-interface PINN (PyTorch)
│   └── postprocess.py      friction correlation, pumping power, PEC (reproduces Table 2)
├── data/                   tabulated PINN results (CSV) used to draw every figure
├── figures/               figure-generation scripts + high-resolution PNGs
└── checkpoints/            place trained model weights here before archiving
```

## Installation

```bash
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt
```

`properties.py`, `postprocess.py` and the figure scripts need only
numpy / scipy / matplotlib. `pinn_mchs.py` additionally needs PyTorch.

## Quick start

```bash
# 1. effective properties + additivity check (Section 5.5)
python src/properties.py

# 2. reproduce Table 2 (Nu, f, dT_base, pumping power, PEC)
python src/postprocess.py

# 3. train the PINN (short CPU smoke run; use a GPU + FP64 for production)
python src/pinn_mchs.py

# 4. regenerate the figures
cd figures && python make_all_figures.py
```

## Data

Every figure in the paper is drawn from a plain-text CSV in `data/`:

| file | contents |
|------|----------|
| `table2_metrics_vs_phi.csv`            | Nu_avg, f, dT_base, pumping power vs volume fraction (Re=500) |
| `table1_hyperparameter_sensitivity.csv`| Nu_avg and validation error vs layers/neurons/activation/resampling |
| `grid_independence.csv`                | Nu_avg and L2 velocity error vs number of collocation points |
| `friction_vs_Re.csv`                   | Darcy friction factor vs Re for phi = 0-3 % |
| `avg_nu_friction_vs_Re.csv`            | average Nu and f vs Re (phi = 0 and 2 %) |
| `avg_nu_friction_vs_aspect_ratio.csv`  | average Nu and f vs channel aspect ratio |
| `enhancement_vs_phi.csv`               | Nu_nf/Nu_bf vs phi at Re = 200, 500, 1000 |
| `local_nu_vs_x.csv`                    | local Nusselt number Nu(x) along the channel |

## Known consistency notes

These are transparency notes for reviewers and users; the code computes the
physically consistent values.

* **Performance evaluation criterion (PEC).** With the definition
  `PEC = (Nu_nf/Nu_bf) / (f_nf/f_bf)^(1/3)` and the Table 2 values, `postprocess.py`
  returns **PEC = 1.04 at Re = 500** (rising to 1.06 at Re = 1000), i.e. a modest
  net thermal-hydraulic gain. Quote this value rather than any higher figure.
* **Prandtl number.** Because the metal/oxide particles have a much lower specific
  heat than water, the effective Pr *decreases* slightly with loading
  (Pr ~ 6.08 for water -> ~ 5.4 at phi = 0.02); `properties.py` prints the exact
  values.

## Citation

If you use this software or data, please cite both the paper and this archive:

> Muhammed Anaz Khan, *PINN-MCHS: Mesh-free conjugate-interface physics-informed
> neural network for hybrid-nanofluid microchannel heat sinks*, v1.0.0, Zenodo,
> 2026. doi:10.5281/zenodo.XXXXXXX

```
Muhammed Anaz Khan
ORCID: https://orcid.org/0000-0002-8837-9865
Department of Mechanical Engineering, College of Engineering,
University of Bisha, Bisha 61922, P.O. Box 551, Saudi Arabia
mkhan@ub.edu.sa
```

Replace `10.5281/zenodo.XXXXXXX` in `CITATION.cff`, `.zenodo.json` and above with
the DOI issued by Zenodo after upload.

## License

MIT License - see [LICENSE](LICENSE).
