"""
Le Verrier's Neptune Prediction Algorithm
=========================================

A modern Python implementation of Urbain Le Verrier's 1846 method
for predicting Neptune's position from Uranus's orbital deviations.

Quick Start
-----------
>>> from leverrier import LeVerrierSolver, BouvardData
>>> 
>>> # Load historical data
>>> data = BouvardData()
>>> obs = data.get_observations()
>>> 
>>> # Solve the inverse problem
>>> solver = LeVerrierSolver(obs.times, obs.deviations, obs.uncertainties)
>>> result = solver.solve()
>>> 
>>> print(f"Predicted semi-major axis: {result.semi_major_axis:.2f} AU")
>>> print(f"Predicted mass: {result.mass:.3e} kg")
>>> print(f"Predicted longitude: {result.longitude:.1f}°")

Modules
-------
constants : Physical and astronomical constants
orbital : Orbital mechanics (Kepler's laws, anomaly conversions)
perturbation : Perturbation theory and Legendre expansion
fourier : Fourier decomposition of longitude deviations
inverse : Inverse problem solver (Gauss-Newton method)
data : Bouvard observation data (1781-1845)

References
----------
- Le Verrier, U.J.J. (1846). Comptes rendus, 22, 907-918
- Murray & Dermott (1999). Solar System Dynamics
- Tarantola (2005). Inverse Problem Theory
"""

__version__ = "1.0.0"
__author__ = "Anton Thiriaux and Don van Beusichem"


# Core imports
from .constants import (
    G, M_SUN, AU,
    URANUS, NEPTUNE, PLANETS,
    DEG_TO_RAD, RAD_TO_DEG,
    ARCMIN_TO_RAD, RAD_TO_ARCMIN,
    period_from_semi_major_axis,
    mean_motion_from_semi_major_axis
)

from .orbital import (
    OrbitalElements,
    solve_kepler_equation,
    eccentric_to_true_anomaly,
    true_to_eccentric_anomaly,
    orbital_radius,
    position_in_orbital_plane,
    synodic_period,
    synodic_phase
)

from .perturbation import (
    evaluate_legendre,
    inverse_distance_legendre,
    disturbing_function_total,
    longitude_deviation_first_order,
    amplitude_scaling,
    semi_major_axis_from_amplitude_ratio,
    mass_from_amplitude
)

from .fourier import (
    FourierCoefficients,
    build_design_matrix,
    fit_fourier_coefficients,
    evaluate_fourier_model,
    harmonic_amplitude,
    harmonic_phase,
    condition_number,
    assess_phase_coverage
)

from .inverse import (
    InversionResult,
    LeVerrierSolver,
    estimate_initial_parameters,
    solve_gauss_newton,
    solve_scipy,
    forward_model_full
)

from .data import (
    Observation,
    ObservationDataset,
    BouvardData,
    get_bouvard_data,
    get_historical_comparison,
    generate_synthetic_data
)

# Convenient aliases
Solver = LeVerrierSolver
Data = BouvardData

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Constants
    'G', 'M_SUN', 'AU',
    'URANUS', 'NEPTUNE', 'PLANETS',
    'DEG_TO_RAD', 'RAD_TO_DEG',
    'ARCMIN_TO_RAD', 'RAD_TO_ARCMIN',
    'period_from_semi_major_axis',
    'mean_motion_from_semi_major_axis',
    
    # Orbital mechanics
    'OrbitalElements',
    'solve_kepler_equation',
    'eccentric_to_true_anomaly',
    'true_to_eccentric_anomaly',
    'orbital_radius',
    'position_in_orbital_plane',
    'synodic_period',
    'synodic_phase',
    
    # Perturbation theory
    'evaluate_legendre',
    'inverse_distance_legendre',
    'disturbing_function_total',
    'longitude_deviation_first_order',
    'amplitude_scaling',
    'semi_major_axis_from_amplitude_ratio',
    'mass_from_amplitude',
    
    # Fourier analysis
    'FourierCoefficients',
    'build_design_matrix',
    'fit_fourier_coefficients',
    'evaluate_fourier_model',
    'harmonic_amplitude',
    'harmonic_phase',
    'condition_number',
    'assess_phase_coverage',
    
    # Inverse problem
    'InversionResult',
    'LeVerrierSolver',
    'estimate_initial_parameters',
    'solve_gauss_newton',
    'solve_scipy',
    'forward_model_full',
    
    # Data
    'Observation',
    'ObservationDataset',
    'BouvardData',
    'get_bouvard_data',
    'get_historical_comparison',
    'generate_synthetic_data',
    
    # Aliases
    'Solver',
    'Data'
]
