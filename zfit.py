"""
zfit.py  -  Level 2 of "Find the Z boson yourself!"

Fits a dilepton invariant-mass histogram with
    signal     = Voigt profile  (Breit-Wigner  ⊗  Gaussian detector resolution)
    background = falling exponential
using an extended binned Poisson likelihood, the same kind of fit LHC analyses use.

Only needs numpy, scipy and matplotlib.

Typical use in your notebook:
    import zfit
    res = zfit.fit_z(mll_mumu, label="μμ")      # mll in GeV
    zfit.plot_fit(res)
    zfit.compare_channels([res_ee, res_mumu])
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import voigt_profile
import matplotlib.pyplot as plt

# PDG 2026 world averages (pdglive.lbl.gov, Z boson)
PDG_MZ, PDG_MZ_ERR = 91.1879, 0.0020          # GeV
PDG_GZ, PDG_GZ_ERR = 2.4955, 0.0023           # GeV


# ---------------------------------------------------------------- model pieces
def signal_pdf(x, M, sigma, Gamma):
    """Voigt profile: Breit-Wigner of full width Gamma, smeared by a Gaussian of
    width sigma. scipy's voigt_profile takes the BW *half* width."""
    return voigt_profile(x - M, sigma, Gamma / 2.0)


def background_pdf(x, lam, lo, hi):
    """exp(-lam*(x-lo)), normalised on [lo, hi]."""
    if abs(lam) < 1e-9:
        return np.full_like(x, 1.0 / (hi - lo))
    norm = (1.0 - np.exp(-lam * (hi - lo))) / lam
    return np.exp(-lam * (x - lo)) / norm


def _bin_integrals(pdf, edges, nsub=10):
    """Integrate a pdf over each bin (midpoint rule on nsub sub-bins).
    More accurate than value-at-centre x width when the peak is narrow."""
    lo, hi = edges[:-1], edges[1:]
    w = (hi - lo) / nsub
    t = (np.arange(nsub) + 0.5)[None, :]
    x = lo[:, None] + t * w[:, None]
    return (pdf(x) * w[:, None]).sum(axis=1)


# ---------------------------------------------------------------- the fit
def _expected(p, edges, lo, hi, Gamma_fixed):
    if Gamma_fixed is None:
        Ns, Nb, M, sigma, Gamma, lam = p
    else:
        Ns, Nb, M, sigma, lam = p
        Gamma = Gamma_fixed
    s = _bin_integrals(lambda x: signal_pdf(x, M, sigma, Gamma), edges)
    # renormalise signal to the fit window (the Voigt tails extend beyond it)
    s = s / s.sum() if s.sum() > 0 else s
    b = _bin_integrals(lambda x: background_pdf(x, lam, lo, hi), edges)
    return Ns * s, Nb * b


def _nll(p, n, edges, lo, hi, Gamma_fixed):
    s, b = _expected(p, edges, lo, hi, Gamma_fixed)
    mu = np.clip(s + b, 1e-12, None)
    # extended binned Poisson likelihood (constant log n! terms dropped)
    return np.sum(mu - n * np.log(mu))


def _hessian(f, p, rel=1e-4):
    """Numerical Hessian by central differences (for the covariance matrix)."""
    p = np.asarray(p, float)
    k = len(p)
    h = rel * np.maximum(np.abs(p), 1e-2)
    H = np.zeros((k, k))
    f0 = f(p)
    for i in range(k):
        for j in range(i, k):
            if i == j:
                pp, pm = p.copy(), p.copy()
                pp[i] += h[i]; pm[i] -= h[i]
                H[i, i] = (f(pp) - 2 * f0 + f(pm)) / h[i] ** 2
            else:
                ppp, ppm, pmp, pmm = (p.copy() for _ in range(4))
                ppp[i] += h[i]; ppp[j] += h[j]
                ppm[i] += h[i]; ppm[j] -= h[j]
                pmp[i] -= h[i]; pmp[j] += h[j]
                pmm[i] -= h[i]; pmm[j] -= h[j]
                H[i, j] = H[j, i] = (f(ppp) - f(ppm) - f(pmp) + f(pmm)) / (4 * h[i] * h[j])
    return H


def fit_z(mll, label="", lo=66.0, hi=116.0, nbins=100, weights=None,
          fix_width=True):
    """
    Fit the Z peak.

    mll        : array of dilepton invariant masses in GeV
    lo, hi     : fit window (66-116 GeV is the standard ATLAS/CMS Z window)
    nbins      : number of bins (0.5 GeV bins by default)
    weights    : optional per-event weights (leave None for real data)
    fix_width  : True  -> Gamma fixed to the PDG value; fit M, resolution sigma
                 False -> also fit Gamma (harder: Gamma and sigma are correlated)
    """
    mll = np.asarray(mll, float)
    edges = np.linspace(lo, hi, nbins + 1)
    n, _ = np.histogram(mll, bins=edges, weights=weights)
    ntot = n.sum()

    Gamma_fixed = PDG_GZ if fix_width else None
    # starting values: most events are signal, peak near the PDG mass
    if fix_width:
        names = ["N_sig", "N_bkg", "M", "sigma", "lambda"]
        p0 = [0.95 * ntot, 0.05 * ntot, 91.0, 2.0, 0.05]
        bounds = [(0, 2 * ntot), (0, 2 * ntot), (85, 97), (0.1, 10), (-0.5, 0.5)]
    else:
        names = ["N_sig", "N_bkg", "M", "sigma", "Gamma", "lambda"]
        p0 = [0.95 * ntot, 0.05 * ntot, 91.0, 2.0, 2.5, 0.05]
        bounds = [(0, 2 * ntot), (0, 2 * ntot), (85, 97), (0.1, 10), (0.1, 10), (-0.5, 0.5)]

    f = lambda p: _nll(p, n, edges, lo, hi, Gamma_fixed)
    opt = minimize(f, p0, method="L-BFGS-B", bounds=bounds)
    opt = minimize(f, opt.x, method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-6, maxiter=20000))  # polish

    # uncertainties from the curvature of -log L at the minimum
    H = _hessian(f, opt.x)
    try:
        cov = np.linalg.inv(H)
        errs = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov, errs = None, np.full(len(opt.x), np.nan)

    s, b = _expected(opt.x, edges, lo, hi, Gamma_fixed)
    mu = s + b
    ok = n > 0
    chi2 = np.sum((n[ok] - mu[ok]) ** 2 / mu[ok])
    ndf = ok.sum() - len(opt.x)

    params = dict(zip(names, opt.x))
    errors = dict(zip(names, errs))
    if fix_width:
        params["Gamma"], errors["Gamma"] = PDG_GZ, 0.0

    return dict(label=label, params=params, errors=errors, cov=cov, names=names,
                edges=edges, counts=n, signal=s, background=b, expected=mu,
                chi2=chi2, ndf=ndf, fix_width=fix_width, lo=lo, hi=hi,
                converged=opt.success)


# ---------------------------------------------------------------- reporting
def summary(res):
    p, e = res["params"], res["errors"]
    lines = [f"Fit result {res['label']}".strip(),
             f"  M_Z    = {p['M']:.3f} ± {e['M']:.3f} GeV   (PDG {PDG_MZ:.4f})",
             f"  sigma  = {p['sigma']:.3f} ± {e['sigma']:.3f} GeV   (detector resolution)",
             (f"  Gamma  = {p['Gamma']:.3f}  (fixed to PDG)" if res["fix_width"]
              else f"  Gamma  = {p['Gamma']:.3f} ± {e['Gamma']:.3f} GeV   (PDG {PDG_GZ:.4f})"),
             f"  N_sig  = {p['N_sig']:.0f} ± {e['N_sig']:.0f}",
             f"  N_bkg  = {p['N_bkg']:.0f} ± {e['N_bkg']:.0f}",
             f"  chi2/ndf = {res['chi2']:.1f}/{res['ndf']} = {res['chi2']/res['ndf']:.2f}"]
    out = "\n".join(lines)
    print(out)
    return out


def plot_fit(res, ax=None, logy=False):
    """Data with Poisson error bars, fitted signal/background, and a pull panel."""
    edges, n = res["edges"], res["counts"]
    c = 0.5 * (edges[1:] + edges[:-1])
    w = edges[1] - edges[0]

    # smooth curves: evaluate the model on a fine grid, scaled to counts per bin
    p = res["params"]
    xf = np.linspace(res["lo"], res["hi"], 1000)
    # same normalisation as the fit: signal pdf renormalised to the fit window
    win = _bin_integrals(lambda x: signal_pdf(x, p["M"], p["sigma"], p["Gamma"]), edges).sum()
    sf = p["N_sig"] * signal_pdf(xf, p["M"], p["sigma"], p["Gamma"]) * w / win
    bf = p["N_bkg"] * background_pdf(xf, p["lambda"], res["lo"], res["hi"]) * w

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
    a1.errorbar(c, n, yerr=np.sqrt(np.maximum(n, 1)), fmt="o", ms=3, color="black",
                label="Data")
    a1.plot(xf, sf + bf, color="#c0392b", lw=2, label="Total fit")
    a1.plot(xf, sf, color="#2e86c1", lw=1.5, ls="--", label="Signal (Voigt)")
    a1.fill_between(xf, bf, color="#f5b041", alpha=0.6, label="Background (exp)")
    a1.set_ylabel(f"Events / {w:.1f} GeV")
    if logy:
        a1.set_yscale("log")
    pr, er = res["params"], res["errors"]
    txt = (f"$M_Z$ = {pr['M']:.3f} ± {er['M']:.3f} GeV\n"
           f"$\\sigma$ = {pr['sigma']:.2f} ± {er['sigma']:.2f} GeV\n"
           f"$\\chi^2$/ndf = {res['chi2']:.0f}/{res['ndf']}")
    a1.text(0.03, 0.95, txt, transform=a1.transAxes, va="top", fontsize=10)
    a1.set_title(f"Z → {res['label']}" if res["label"] else "Z boson fit")
    a1.legend(loc="upper right", frameon=False)

    pulls = np.where(n > 0, (n - res["expected"]) / np.sqrt(np.maximum(res["expected"], 1e-9)), 0)
    a2.bar(c, pulls, width=w, color="#7f8c8d")
    a2.axhline(0, color="black", lw=0.8)
    for y in (-2, 2):
        a2.axhline(y, color="black", lw=0.5, ls=":")
    a2.set_ylim(-5, 5)
    a2.set_ylabel("Pull")
    a2.set_xlabel(r"$m_{\ell\ell}$ [GeV]")
    return fig


def compare_channels(results):
    """Side-by-side table of fit results (e.g. ee vs μμ)."""
    hdr = f"{'channel':<10}{'M_Z [GeV]':>20}{'sigma [GeV]':>18}{'N_sig':>12}{'chi2/ndf':>12}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        p, e = r["params"], r["errors"]
        print(f"{r['label']:<10}{p['M']:>11.3f} ± {e['M']:<6.3f}{p['sigma']:>10.3f} ± {e['sigma']:<5.3f}"
              f"{p['N_sig']:>12.0f}{r['chi2']/r['ndf']:>12.2f}")
    print(f"{'PDG':<10}{PDG_MZ:>11.4f} ± {PDG_MZ_ERR:<6.4f}")
    if len(results) == 2:
        a, b = results
        d = a["params"]["M"] - b["params"]["M"]
        de = np.hypot(a["errors"]["M"], b["errors"]["M"])
        print(f"\nΔM ({a['label']} − {b['label']}) = {d:.3f} ± {de:.3f} GeV  →  {abs(d)/de:.1f}σ apart")
