# HeatTail: heat-structured pole reachability

Reproducible research accompanying **Which Complex Poles Are Compatible with Unresolved Data? Sharp Sobolev-Tail Bounds for Periodic Burgers Flow**, by Shivam Gupta (2026).

[Read the manuscript](paper/Heat_Structured_Pole_Reachability_Shivam_Gupta.pdf).

This repository studies complex poles of periodic, zero-mean viscous Burgers solutions when the normalized initial Cole–Hopf potential has fixed low Fourier modes and a bounded **real** Sobolev tail. The result is conditional on that uncertainty set; a finite spectrum does not itself establish a tail norm bound.

## What is implemented

- Exact one-point minimum-norm interpolation and explicit pole-producing witnesses.
- Shared-tail interpolation for several zero/derivative constraints.
- Outward-rounded infinite H1-tail bounds using FLINT/Arb.
- Continuous rectangular-region and contour certificates. An unresolved result is never treated as a certificate.
- Sharp single-cosine analyticity-width brackets, resolution sweeps, asymmetric centers, and an independent dealiased Fourier/RK4 Burgers evolution.

The foundational ideas are classical real pseudozeros and Hilbert-space minimum-norm interpolation. The paper specializes them to heat-propagated unresolved modes, with a uniform positivity condition, a sharp small-time law, and a reproducible validated computation. It does not claim a general-purpose PDE singularity solver or a new general interpolation theorem.

## Reproduce

Use Python **3.9–3.12**. A separate environment avoids changing existing installations.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiments/run_study.py
.venv/bin/python experiments/resolution.py
.venv/bin/python experiments/general_centers.py
.venv/bin/python experiments/witness_certificate.py
.venv/bin/python experiments/forward_check.py
.venv/bin/python experiments/make_figures.py
```

Rebuild the manuscript with [Tectonic](https://tectonic-typesetting.github.io/), from the repository root:

```bash
tectonic --outdir paper paper/main.tex
```

This generates `paper/main.pdf`. The named manuscript is the released PDF.

## Key result

For initial center `1 + 0.8 cos(x)`, fixed modes through `N=1`, scaled time `tau=nu*t=0.02`, and H1 tail budget `eta=0.01`, the sharp robust analyticity half-width lies in `(0.5816, 0.5817)`. The nominal width is about `0.725915`. An exact finite witness puts a simple pole at `pi + 0.59i`, has norm about `0.00855225244`, and retains an initial potential bounded below by `0.1960731252`.

For requested poles at `pi+0.58i` and `pi+0.60i`, individual minimum norms are about `0.0102964` and `0.00704220`, whereas the shared-tail minimum is `0.04162996`. They are individually possible at budget `0.02` but jointly impossible.

## Files and verification contract

- `src/heatpoles.py`: ordinary floating-point exploration. `witness` uses an SVD cutoff of `1e-13`; inspect its residual and rank, particularly for ill-conditioned constraints. It is not a rigorous certificate.
- `src/verified.py`: Arb computations at 160 bits, rigorous geometric tail bounds, continuous box subdivision, and contour winding.
- `results/study.json`: 24 width brackets, five joint cases, four symmetric contours, and six asymptotic evaluations.
- `results/resolution.json`: seven resolution brackets.
- `results/general.json`: six asymmetric contour certificates.
- `results/witness_certificate.json`: outward-rounded norm, positivity, and nonzero derivative for the exact finite witness formula.
- `results/witness.npz`: floating-point coefficients for plotting and forward-solver checking. **Use the formula in the certificate for the mathematical witness**, rather than treating rounded coefficients as exact interpolation.
- `results/forward.json`: independent forward evolution errors, including the underresolved 64-point case.
- `tests/test_core.py`: 240 comparisons against independently assembled real interpolation matrices, 60 high-precision checks of infinite-sum enclosures, conjugacy/derivative tests, and certificate refusal/orientation checks.
- `figures/`: four generated vector PDFs and PNG previews.
- `paper/`: LaTeX source and released manuscript.

Norm convention: `||h||_s^2 = sum_{k>N} k^(2s)*(a_k^2+b_k^2)`. In particular `||h||_1^2 = (1/pi)*integral_0^(2pi) |h'(x)|^2 dx`. Positivity is guaranteed when `eta * sqrt(sum_{k>N} k^(-2s)) < min(p0)`. Outside this regime, unconstrained interpolation need not produce positive initial data.

A successful contour output is a signed winding number. Positive orientation gives the number of zeros with multiplicity. A negative orientation has the opposite sign. These are potential-zero counts, not necessarily counts of distinct poles if multiplicities occur.

The certified implementation covers H1 tails. The paper's general-s results are mathematical; arbitrary-s interval certification is not implemented. Primary examples use analytic heat evolution, with the independent real-domain integrator as an additional check, not as the source of the pole certificates. Timings and floating rounding depend on the environment; all interval decisions should be rerun.

## Literature and overlap

The closest foundations include Graillat and Langlois (2007), real versus complex polynomial pseudozeros; Aronszajn (1950), reproducing kernels; VandenHeuvel et al. (2023), Burgers complex singularities; Trefethen (2023), numerical analytic continuation; and Gentner, Dallaston, and McCue (2026), piecewise-continuous Burgers initial data. Full references and contribution boundaries are in the paper.

The author's earlier manuscripts in the supplied research collection were reviewed for topic overlap. The nearby computational-mathematics manuscript on finite-field cellular automata studies a different dynamical system, observation map, and algebra. No earlier text, data, or figures from that collection are reused here. A literature search cannot certify universal priority, and this repository does not assert it.

## Citation

```bibtex
@misc{gupta2026heattail,
  author = {Shivam Gupta},
  title = {Which Complex Poles Are Compatible with Unresolved Data? Sharp Sobolev-Tail Bounds for Periodic Burgers Flow},
  year = {2026},
  howpublished = {Research manuscript and reproducible software},
  url = {https://github.com/shi1720/heat-pole-reachability}
}
```

License: MIT for original code. References remain the work of their respective authors.
