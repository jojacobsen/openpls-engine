---
title: "openpls-engine: An open-source Python implementation of Partial Least Squares Structural Equation Modeling"
tags:
  - Python
  - PLS-SEM
  - structural equation modeling
  - variance-based SEM
  - reproducibility
authors:
  - name: Johannes Jacob
    orcid: 0009-0007-5488-8446
    affiliation: 1
affiliations:
  - index: 1
    name: Independent Researcher, Hamburg, Germany
date: 8 June 2026
bibliography: paper.bib
---

# Summary

`openpls-engine` is a Python 3 library for Partial Least Squares Structural
Equation Modeling (PLS-SEM), a variance-based technique widely used in
marketing, information-systems, organisational-behavior, and tourism research
[@wold1982; @lohmoeller1989; @hair2017primer]. The library estimates inner
and outer models for reflective (Mode A) and formative (Mode B) latent
constructs, computes the quality criteria expected of modern PLS-SEM reporting
(HTMT, SRMR, $d_{ULS}$, $f^{2}$, Stone-Geisser $Q^{2}$, Cronbach $\alpha$,
Dijkstra-Henseler $\rho_{A}$, BIC for endogenous latent variables), and ships
the analyses that go beyond a basic primer fit: multi-group analysis,
Importance-Performance Map Analysis, two-stage moderation, FIMIX-PLS
finite-mixture segmentation, and PLSpredict out-of-sample assessment
[@henseler2015htmt; @ringle2016ipma; @henseler2010moderation; @hahn2002fimix;
@shmueli2019plspredict]. The library is published on PyPI under the GPL-3.0
license and is the computational core of the hosted application OpenPLS
(<https://openpls.app>).

# Statement of need

PLS-SEM is the variance-based counterpart to covariance-based SEM and is
preferred when sample sizes are moderate, when the structural model contains
formative constructs, or when prediction rather than parameter recovery is the
research goal. In applied research the dominant implementation is a
closed-source commercial desktop application [@smartpls4]. Two practical
problems follow from this status quo. First, license cost: an institutional
seat is several hundred Euros per year, a student license approximately
EUR 90 per semester, which excludes researchers in low-budget institutions
and complicates teaching at scale. Second, reproducibility: a peer-reviewer
cannot rerun a published analysis without acquiring the same licensed tool,
and the algorithmic implementation is not publicly inspectable, which makes
auditing and methodological discussion of edge cases difficult.

`openpls-engine` is targeted at applied PLS-SEM researchers and graduate
students who need a Python-native, reproducible, freely-licensed tool that
covers the same quality criteria the field expects, and at methodologists
who need an inspectable reference implementation against which alternative
formulations can be tested.

# State of the field

Two open-source PLS-SEM implementations exist. The R package `seminr` offers
a comprehensive domain-specific language for model specification and is the
most actively maintained open-source option, but it lives outside the Python
data-science ecosystem [@ray2022seminr]. The Python package `plspm-python`
[@humble2017plspm], originally released by Jez Humble (Google Cloud Platform)
under GPL-3.0, implements the algorithmic core (centroid, factorial, and path
inner-weighting schemes; reflective and formative outer models; bootstrap
inference) but has not seen a release since 2020 and lacks the modern quality
criteria and advanced analyses that PLS-SEM reporting guidelines now expect
(HTMT, SRMR, $d_{ULS}$, $f^{2}$, $Q^{2}$, IPMA, FIMIX, MGA, PLSpredict,
two-stage moderation).

`openpls-engine` is a fork of `plspm-python` 0.5.7 that preserves the
algorithm core and its GPL-3.0 license (the upstream attribution is kept in
`ATTRIBUTION.md` and `NOTICE`) and adds the missing metrics and analyses,
plus a deterministic latent-variable sign convention, column-wise
indicator standardisation, and a stable public API (`Plspm`, `Config`,
`Mode`, `Scheme`, `HTMT`, `ModelFit`, `QSquared`, `IPMA`, `PLSPredict`,
`Moderation`, `FIMIX`, `MGA`, `LongBootstrap`).

# Software design

The library is pure Python and depends only on `numpy`, `pandas`, `scipy`,
`statsmodels`, and `scikit-learn`. It supports Python 3.10 through 3.13.
Five inner-weighting schemes are exposed:

- **CENTROID**, **FACTORIAL**, **PATH** — the classical schemes of Wold and
  Lohmöller [@wold1982; @lohmoeller1989], inherited from `plspm-python`;
- **NEWTON** — a quasi-Newton (BFGS) joint optimisation over the full
  latent-variable neighbourhood;
- **PCA** — Lohmöller's first-principal-component scheme [@lohmoeller1989].

The estimation pipeline (initialisation → outer estimation → inner estimation
→ convergence check) is exposed as a single `Plspm(...)` call returning a
result object whose methods are the typed accessors expected by PLS-SEM
reporting (`path_coefficients`, `inner_summary`, `outer_loadings`,
`outer_weights`, `model_fit`, `htmt`, `effect_sizes`, `q_squared`, ...).
Advanced analyses (`IPMA`, `PLSPredict`, `Moderation`, `FIMIX`, `MGA`,
`LongBootstrap`) are separate top-level classes that consume a fitted
`Plspm` object, so they can be added to existing pipelines without
re-estimating the base model.

The library follows Semantic Versioning from 1.0.0 onwards. Numerical
changes in any post-1.0 release are pinned by a regression test and
documented in `CHANGELOG.md`. Documentation, installation, quickstart,
worked examples, and the full API reference are published at
<https://openpls.app/engine>.

# Research impact statement

The library has been validated against the dominant proprietary
implementation on 14 reference cases drawn from the canonical PLS-SEM
literature, including the ECSI Complete Data benchmark, the ACSI model, the
Technology Acceptance Model, UTAUT, the organisational-identification model
of Mael and Ashforth, an Employee Retention model, and the Corporate
Reputation primer of Hair et al. in its simple, extended, advanced, and four
redundancy-analysis variants. For every comparable cell (path coefficients,
$R^{2}$, indicator loadings, outer weights, HTMT pairs) the engine matches
the reference within an absolute tolerance of $|\Delta| < 0.001$ across
all 14 cases under the path inner-weighting scheme. The full numerical
matrix, the residual SRMR offset, and the FACTORIAL-scheme edge case are
documented in the accompanying validation pre-print [@jacob2026validation],
with reproducibility supplementary material archived on Zenodo
[@jacob2026supplement]. The engine release used for those results is
archived under DOI 10.5281/zenodo.20509385 [@jacob2026engine].

# AI usage disclosure

Generative AI assistants (Anthropic Claude) were used during development for
exploratory code review, scaffolding of validation scripts and CI workflows,
prose editing of documentation and this manuscript, and as a discussion
partner for debugging numerical deviations against the reference
implementation. The algorithmic core inherited from `plspm-python` is
unchanged in its mathematical structure. The added quality criteria
(HTMT, SRMR, $d_{ULS}$, $f^{2}$, $Q^{2}$, $\rho_{A}$, BIC) and advanced
analyses (IPMA, FIMIX-PLS, two-stage moderation, MGA, PLSpredict,
NEWTON/PCA inner schemes) were implemented by the author with explicit
references to the methodological literature cited above, and every
implementation choice was independently validated against the reference
on the 14-case matrix before release. The author takes full responsibility
for the correctness of the code and for the content of this paper.

# Acknowledgements

`openpls-engine` is a fork of `plspm-python` by Jez Humble (Google Cloud
Platform), released under GPL-3.0. The fork preserves the original copyright
and inherits the GPL-3.0-or-later license. The author acknowledges the
original authors of the public PLS-SEM teaching datasets used as reference
inputs in the accompanying validation work (full citations in the validation
pre-print). No external funding was received for this work.

# References
