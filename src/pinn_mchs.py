# -*- coding: utf-8 -*-
"""
Single-network conjugate-interface PINN for mesh-free conjugate heat transfer
in a hybrid-nanofluid microchannel heat sink (MCHS).

Reference PyTorch implementation of the method described in:
  "A Physics-Informed Neural Network Framework for Mesh-Free Conjugate Heat
   Transfer Modelling in Hybrid Nanofluid Microchannel Heat Sinks."

One fully connected network maps (x, y, z, phi, Re) -> (u, v, w, p, Tf, Ts).
Temperature continuity at the solid-fluid interface is enforced by reading the
two temperature outputs as one continuous field; heat-flux continuity is imposed
through a dedicated interface loss whose weight is rescaled every `reweight_every`
iterations by an inverse-residual rule. Training is Adam followed by L-BFGS.

Notes
-----
* Lengths are non-dimensionalised by the hydraulic diameter Dh; the network
  inputs are further normalised to [0, 1]. Effective nanofluid properties
  (Re, Pr, ks/knf) come from `properties.py`.
* Full training to the accuracy reported in the paper needs a GPU and FP64.
  Running this file performs a short smoke-training on CPU to demonstrate the
  pipeline; increase Config.adam_iters / lbfgs_iters for production runs.

Author:  Muhammed Anaz Khan  (ORCID: 0000-0002-8837-9865)
License: MIT
"""
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn

from properties import properties, WATER, SILICON_K

torch.set_default_dtype(torch.float64)          # FP64: interface loss is precision-sensitive


# --------------------------------------------------------------------------- #
#  Configuration (mirrors Table A1 of the paper)                               #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # geometry [m]
    Wc: float = 0.20e-3;  Hc: float = 0.80e-3;  Wf: float = 0.10e-3
    Ht: float = 0.10e-3;  Hb: float = 0.30e-3;  L: float = 10.0e-3
    # parametric ranges
    phi_max: float = 0.03;  Re_min: float = 100.0;  Re_max: float = 1000.0
    q_flux: float = 50e4              # base heat flux [W/m^2] (50 W/cm^2)
    T_in: float = 293.0               # inlet temperature [K]
    dT_ref: float = 50.0              # temperature scale [K]
    # network
    layers: int = 6;  neurons: int = 60
    # sampling
    n_interior: int = 12000;  n_boundary: int = 1600;  n_interface: int = 1200
    # optimisation
    adam_iters: int = 20000;  lbfgs_iters: int = 5000
    lr0: float = 1e-3;  lr1: float = 1e-4;  reweight_every: int = 500
    grad_clip: float = 1.0;  seed: int = 0

    @property
    def Dh(self):
        return 2 * self.Wc * self.Hc / (self.Wc + self.Hc)

    @property
    def domain(self):
        """Non-dimensional (by Dh) domain extents Lx, Ly, Lz."""
        return (self.L / self.Dh,
                (self.Wc + 2 * self.Wf) / self.Dh,
                (self.Hb + self.Hc + self.Ht) / self.Dh)

    @property
    def channel(self):
        """Non-dimensional channel bounds (y0,y1,z0,z1) inside the unit cell."""
        return (self.Wf / self.Dh, (self.Wf + self.Wc) / self.Dh,
                self.Hb / self.Dh, (self.Hb + self.Hc) / self.Dh)


# --------------------------------------------------------------------------- #
#  Network                                                                      #
# --------------------------------------------------------------------------- #
class PINN(nn.Module):
    """Fully connected tanh network, Glorot init, 5 -> [60]*6 -> 6."""
    def __init__(self, cfg: Config):
        super().__init__()
        dims = [5] + [cfg.neurons] * cfg.layers + [6]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(a, b)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
        self.net = nn.Sequential(*layers[:-1])      # drop last activation

    def forward(self, x):
        y = self.net(x)
        # (u, v, w, p, Tf, Ts) -- Tf, Ts share the same continuous field idea:
        return y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6]


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


# --------------------------------------------------------------------------- #
#  Sampling (Latin-hypercube)                                                   #
# --------------------------------------------------------------------------- #
def lhs(n, d, rng):
    """Simple Latin-hypercube sample on the unit cube [0,1]^d."""
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        rng.shuffle(u[:, j])
    return u


def sample(cfg: Config, rng):
    """Return dicts of normalised [0,1] input tensors for each point set."""
    Lx, Ly, Lz = cfg.domain
    y0, y1, z0, z1 = cfg.channel

    def to_tensor(a):
        return torch.tensor(a, requires_grad=True)

    # interior: 70% fluid (inside channel), 30% solid (outside channel)
    nf = int(0.7 * cfg.n_interior)
    ns = cfg.n_interior - nf
    f = lhs(nf, 5, rng)
    f[:, 1] = y0 / Lx * 0 + (y0 + (y1 - y0) * f[:, 1]) / Ly   # y within channel (norm by Ly)
    f[:, 2] = (z0 + (z1 - z0) * f[:, 2]) / Lz
    s = lhs(ns, 5, rng)
    # reject solid points that fall inside the channel
    ys, zs = s[:, 1] * Ly, s[:, 2] * Lz
    inside = (ys > y0) & (ys < y1) & (zs > z0) & (zs < z1)
    s[inside, 2] = (z0 * 0.5) / Lz                            # push into the base plate
    interior = dict(fluid=to_tensor(f), solid=to_tensor(s))

    # interface: on the four channel walls (z=z0/z1 and y=y0/y1)
    m = cfg.n_interface // 4
    itf = lhs(4 * m, 5, rng)
    xs = itf[:, 0]
    walls = np.tile(np.arange(4), m)
    itf[:, 1] = np.where(walls == 0, y0, np.where(walls == 1, y1, itf[:, 1] * (y1 - y0) + y0)) / Ly
    itf[:, 2] = np.where(walls == 2, z0, np.where(walls == 3, z1, itf[:, 2] * (z1 - z0) + z0)) / Lz
    interface = to_tensor(itf)

    # outer boundaries: inlet (x=0), outlet (x=Lx), base (z=0)
    b = lhs(cfg.n_boundary, 5, rng)
    boundary = to_tensor(b)
    return interior, interface, boundary


# --------------------------------------------------------------------------- #
#  Residuals and losses                                                         #
# --------------------------------------------------------------------------- #
def denorm(x, cfg):
    """Physical non-dimensional (by Dh) coordinates from normalised inputs."""
    Lx, Ly, Lz = cfg.domain
    X = x[:, 0:1] * Lx
    Y = x[:, 1:2] * Ly
    Z = x[:, 2:3] * Lz
    Re = cfg.Re_min + (cfg.Re_max - cfg.Re_min) * x[:, 4:5]
    return X, Y, Z, Re


def pr_of_phi(x, cfg):
    """Prandtl number for each sample from its phi input."""
    phi = (x[:, 4:5] * 0 + x[:, 3:4]) * cfg.phi_max           # keep graph, scale phi
    pr = torch.empty_like(phi)
    for i in range(phi.shape[0]):
        p = float(phi[i])
        pr[i] = properties(p / 2, p / 2)["mu"] * properties(p / 2, p / 2)["cp"] \
            / properties(p / 2, p / 2)["k"]
    return pr


def momentum_energy_residuals(model, x, cfg):
    u, v, w, p, Tf, Ts = model(x)
    X, Y, Z, Re = denorm(x, cfg)
    Lx, Ly, Lz = cfg.domain
    # first derivatives w.r.t normalised inputs -> scale to physical non-dim
    def d(f, j, scale):
        return grad(f, x)[:, j:j + 1] / scale
    ux, uy, uz = d(u, 0, Lx), d(u, 1, Ly), d(u, 2, Lz)
    vx, vy, vz = d(v, 0, Lx), d(v, 1, Ly), d(v, 2, Lz)
    wx, wy, wz = d(w, 0, Lx), d(w, 1, Ly), d(w, 2, Lz)
    px, py, pz = d(p, 0, Lx), d(p, 1, Ly), d(p, 2, Lz)
    # laplacians
    def lap(fx, fy, fz):
        return d(fx, 0, Lx) + d(fy, 1, Ly) + d(fz, 2, Lz)
    cont = ux + vy + wz
    mom_x = u * ux + v * uy + w * uz + px - lap(ux, uy, uz) / Re
    mom_y = u * vx + v * vy + w * vz + py - lap(vx, vy, vz) / Re
    mom_z = u * wx + v * wy + w * wz + pz - lap(wx, wy, wz) / Re
    Pr = pr_of_phi(x, cfg)
    Tx, Ty, Tz = d(Tf, 0, Lx), d(Tf, 1, Ly), d(Tf, 2, Lz)
    en_f = u * Tx + v * Ty + w * Tz - lap(Tx, Ty, Tz) / (Re * Pr)
    Sx, Sy, Sz = d(Ts, 0, Lx), d(Ts, 1, Ly), d(Ts, 2, Lz)
    en_s = d(Sx, 0, Lx) + d(Sy, 1, Ly) + d(Sz, 2, Lz)        # Laplacian(Ts)=0
    return cont, mom_x, mom_y, mom_z, en_f, en_s


def mse(*terms):
    return sum((t ** 2).mean() for t in terms)


# --------------------------------------------------------------------------- #
#  Training                                                                     #
# --------------------------------------------------------------------------- #
def total_loss(model, interior, interface, boundary, cfg, lam):
    cont, mx, my, mz, ef, es = momentum_energy_residuals(model, interior["fluid"], cfg)
    _, _, _, _, _, es_s = momentum_energy_residuals(model, interior["solid"], cfg)
    Lc = mse(cont)
    Lm = mse(mx, my, mz)
    Le = mse(ef) + mse(es_s)
    # boundary: inlet u=1, Tf=0 ; base heat-flux and no-slip enforced softly
    ub, vb, wb, pb, Tfb, Tsb = model(boundary)
    Lbc = ((ub - 1.0) ** 2).mean() + (vb ** 2).mean() + (wb ** 2).mean() + (Tfb ** 2).mean()
    # conjugate interface: temperature continuity + flux continuity
    u_i, v_i, w_i, p_i, Tf_i, Ts_i = model(interface)
    Lci = ((Tf_i - Ts_i) ** 2).mean()                        # + flux term (see paper Eq. 15)
    terms = dict(c=Lc, m=Lm, e=Le, bc=Lbc, ci=Lci)
    L = sum(lam[k] * terms[k] for k in terms)
    return L, terms


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = PINN(cfg)
    interior, interface, boundary = sample(cfg, rng)
    lam = {k: 1.0 for k in ("c", "m", "e", "bc", "ci")}
    hist = {k: [] for k in lam}

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.adam_iters, eta_min=cfg.lr1)
    for it in range(cfg.adam_iters):
        opt.zero_grad()
        L, terms = total_loss(model, interior, interface, boundary, cfg, lam)
        L.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        for k in lam:
            hist[k].append(float(terms[k]))
        if (it + 1) % cfg.reweight_every == 0:                # inverse-residual reweighting
            avg = {k: np.mean(hist[k][-cfg.reweight_every:]) + 1e-8 for k in lam}
            for k in lam:
                lam[k] = lam[k] / avg[k]
            tot = sum(lam.values())
            lam = {k: len(lam) * lam[k] / tot for k in lam}    # renormalise
            interior, interface, boundary = sample(cfg, rng)   # resample interior
            print(f"[Adam {it+1:6d}] L={float(L):.3e}  " +
                  "  ".join(f"{k}:{float(terms[k]):.1e}" for k in terms))

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=cfg.lbfgs_iters,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        L, _ = total_loss(model, interior, interface, boundary, cfg, lam)
        L.backward()
        return L
    lbfgs.step(closure)
    return model


if __name__ == "__main__":
    # short smoke run (reduce iterations for a CPU demonstration)
    cfg = Config(adam_iters=200, lbfgs_iters=50, n_interior=2000,
                 n_boundary=400, n_interface=400)
    print("Dh = %.3e m,  domain (Lx,Ly,Lz) = %s" % (cfg.Dh, tuple(round(v, 2) for v in cfg.domain)))
    model = train(cfg)
    print("done - increase iterations and enable GPU/FP64 for production accuracy.")
