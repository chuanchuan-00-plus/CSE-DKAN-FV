# Overleaf package

Upload this directory to a new Overleaf project and set `main.tex` as the main document. The compiled preview is `main.pdf`.

This revision reframes the method around one scientific principle: DKAN learns only zero-mean subcell corrections in the null space of the parent-cell averaging operator. The Godunov or HLLC update remains fixed. The manuscript does not claim independent novelty for the classical flux, jump profile, sensor, positivity limiter, or TV projection.

Included:

- `main.tex`: 27-page English double-column manuscript.
- `references.bib`: compile-ready BibTeX, including the closest learned-TVD-limiter comparison added during positioning review.
- `figures/`: the 17 PDF figures referenced by `main.tex`.
- `source_data/`: CSV and JSON records underlying the reported figures, tables, frozen protocols, sensor ablation, and reference-convergence audit.

Before submission, replace every red `[AUTHOR ACTION: ...]` field. Authors, affiliations, repository and archive identifiers, contributions, funding, computing details, and competing interests cannot be inferred from the research files. The journal class and bibliography style should be changed only after a target journal is selected.

Local compile command:

```powershell
Set-Location paper_build/overleaf_upload_top_journal_revision
..\..\.venv-manuscript\Scripts\tectonic.exe main.tex --keep-logs --keep-intermediates
```
