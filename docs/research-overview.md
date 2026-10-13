# Neptune Prediction from Orbital Perturbations

**Anton Thiriaux · Don van Beusichem**  
École Polytechnique

[Read the full report](neptune-report.pdf) — December 2025.

## Scientific question

How can an unseen perturbing planet be inferred from discrepancies between observed and predicted planetary motion? This study explores that inverse problem in the context of the prediction of Neptune from Uranus's orbital deviations.

## Computational formulation

The implementation contains orbital mechanics utilities, a perturbative representation of gravitational effects, and a Fourier decomposition of longitude deviations. Harmonic coefficients provide initial parameter estimates, followed by nonlinear refinement through Gauss–Newton or SciPy-based optimization.

Supporting analyses examine the response to observational noise, sensitivity to model parameters, and three-body numerical dynamics. These components are available in the `leverrier/` and `simulations/` directories.

## Data and interpretation

The repository includes a reconstructed historical observation table and data-loading utilities. Its historical provenance and uncertainty estimates should be checked against the cited original sources before drawing archival conclusions. Reconstructed data and synthetic experiments must be distinguished from independent observations.

The numerical implementation is a modern computational treatment inspired by the historical inverse problem. It should not be read as evidence that every historical calculation or modelling assumption has been reproduced exactly.

## Reproducibility

The [repository README](../README.md) gives installation and execution instructions. The existing test suite is in `tests/test_leverrier.py`; examples are in `examples/`.

This overview describes the supplied implementation. No new numerical results or independently verified accuracy estimates are introduced by this documentation revision. A quantitative report should state the dataset, units, reference epoch, initialization, convergence criteria, and validation procedure associated with each result.

## Attribution

This work is credited jointly to **Anton Thiriaux and Don van Beusichem**. The independent repository preserves the shared implementation and its commit history. See [authorship and provenance](../AUTHORS.md) and [citation metadata](../CITATION.cff).
