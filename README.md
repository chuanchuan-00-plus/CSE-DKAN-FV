# CSE-DKAN-FV

Research code, frozen protocols, processed results, and manuscript sources for **Discontinuity-Aware KANs for Conservative Subcell Reconstruction of One-Dimensional Shocks**.

The method does not learn the finite-volume time integrator. A Godunov/HLLC backbone advances parent-cell averages, while a discontinuity-aware Kolmogorov--Arnold network proposes only zero-mean subcell corrections. Positivity, stencil, total-variation, and trust-region checks can contract unsafe proposals toward the fixed MC reconstruction without changing the evolved parent mean.

## Authors

- Jiakang Cao — first author and repository owner

School of Physics, Changchun University of Science and Technology, Changchun, Jilin, China  
Correspondence: [waasledainbu776@gmail.com](mailto:waasledainbu776@gmail.com)

The GitHub repository is owned solely by **Jiakang Cao** (`chuanchuan-00-plus`). The manuscript authorship above is separate from repository ownership.

## Repository layout

- `src/cse_dkan/`: DKAN models, finite-volume solvers, losses, metrics, and training utilities.
- `scripts/`: training, benchmark, summarization, figure-generation, and audit entry points.
- `tests/`: automated tests for Burgers, Euler, subcell reconstruction, FNO, and training code.
- `protocols/`: frozen JSON evaluation protocols and checksums.
- `results/`: processed experiment outputs, checkpoints, and machine-readable summaries used in the study.
- `report/` and `reports/`: research logs, protocol notes, and evidence summaries.
- `paper/`: Overleaf-ready LaTeX source, figures, source data, bibliography, and compiled PDF.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[analysis]"
```

For CUDA acceleration, install the PyTorch build appropriate for the local driver before installing this package.

## Validation

Run the unit tests from the repository root:

```bash
python -m pytest -q
```

Frozen experiment definitions are under `protocols/`. The scripts beginning with `run_`, `train_`, and `summarize_` expose their options through `--help`. Processed evidence for the principal claims is stored in `results/final_float64_v3/`, `results/canonical_pinn_baselines_v1/`, and `results/mechanism_generalization_v3/`.

## Manuscript

Upload the contents of `paper/` directly to Overleaf and set `main.tex` as the main document. The compiled manuscript is available as `paper/main.pdf`.

The current evidence is limited to one-dimensional Burgers and ideal-gas Euler problems. The repository does not claim multidimensional validation, universal neural-solver superiority, or replacement of fine-grid finite-volume reference methods.

## Licence and archival status

No open-source licence or archival DOI has yet been assigned. The repository is private; reuse or redistribution therefore requires permission from the repository owner.
