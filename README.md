# Neptune Prediction from Orbital Perturbations

**Anton Thiriaux · Don van Beusichem**  
École Polytechnique

A computational study of the inverse problem underlying Neptune's discovery: inferring an unseen planet from deviations in Uranus's orbit. The implementation combines perturbation theory, Fourier analysis, least-squares estimation, and nonlinear parameter refinement.

## Research approach

The code represents longitude deviations through harmonic components of the synodic phase, estimates orbital parameters, and refines them against the supplied observations. Supporting modules explore parameter sensitivity, Monte Carlo noise experiments, and numerical three-body dynamics.

The supplied data are described by the original implementation as reconstructed historical observations. They should be distinguished from a directly transcribed and independently verified archival dataset. Numerical accuracy claims require reproducible runs with the stated data, model assumptions, and parameter settings.

## Documentation

- [Full report (PDF)](docs/neptune-report.pdf): *The Mathematical Discovery of Neptune: Le Verrier’s Predictive Method, A Complete Reconstruction with Original Analysis* — Anton Thiriaux and Don van Beusichem, December 2025.
- [Research overview](docs/research-overview.md): scope, methods, data, and limitations.
- [Authorship and provenance](AUTHORS.md): joint attribution and source history.
- [Citation metadata](CITATION.cff): both authors and this repository's address.

## Repository contents

| Directory | Contents |
|---|---|
| `leverrier/` | Orbital mechanics, perturbation model, Fourier analysis, inverse solver, and data utilities |
| `simulations/` | Monte Carlo, sensitivity, and three-body calculations |
| `examples/` | Basic prediction, extended analysis, and visualization scripts |
| `data/` | Reconstructed observation table |
| `tests/` | Existing numerical tests in `test_leverrier.py` |

## Installation and execution

Python 3.10 or later is specified by the package metadata. From the repository root:

```bash
git clone https://github.com/beusy/neptune-prediction.git
cd neptune-prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/basic_prediction.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

To run the existing test suite:

```bash
python -m pytest tests/ -v
```

The computational implementation is preserved from the shared source. This presentation update does not independently validate its numerical results or establish complete test coverage.

## Citation

```bibtex
@software{thiriaux_vanbeusichem_neptune,
  author = {Thiriaux, Anton and {van Beusichem}, Don},
  title = {Neptune Prediction from Orbital Perturbations},
  url = {https://github.com/beusy/neptune-prediction}
}
```

## Authors and provenance

**Anton Thiriaux and Don van Beusichem** are credited jointly for this work.
