#!/usr/bin/env python3
"""
Sensitivity Analysis Module
===========================

Analyze sensitivity of Neptune prediction to various parameters.

Usage:
    python -m simulations.sensitivity
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverrier import LeVerrierSolver, URANUS, forward_model_full


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""
    parameter_name: str
    parameter_values: np.ndarray
    mass_errors: np.ndarray
    a_errors: np.ndarray
    lambda_errors: np.ndarray
    convergence_rates: np.ndarray


def sensitivity_to_noise(noise_levels: List[float] = [0.5, 1.0, 2.0, 5.0],
                          n_trials: int = 30,
                          seed: Optional[int] = 42) -> SensitivityResult:
    """Analyze sensitivity to observation noise level."""
    if seed is not None:
        np.random.seed(seed)
    
    true_mass, true_a, true_lambda = 1.024e26, 30.07, 325.0
    true_params = np.array([true_mass/1e26, true_a, true_lambda])
    
    n_obs = 60
    times = np.linspace(0, 50, n_obs)
    true_signal = forward_model_full(true_params, times, URANUS.mean_motion, URANUS.semi_major_axis)
    
    mass_errors, a_errors, lambda_errors, convergence_rates = [], [], [], []
    
    for noise in noise_levels:
        m_err, a_err, l_err, converged = [], [], [], 0
        
        for _ in range(n_trials):
            observations = true_signal + np.random.normal(0, noise, n_obs)
            try:
                solver = LeVerrierSolver(times, observations)
                result = solver.solve(method='scipy')
                if result.converged:
                    m_err.append(abs(result.mass - true_mass) / true_mass)
                    a_err.append(abs(result.semi_major_axis - true_a) / true_a)
                    l_err.append(abs(result.longitude - true_lambda))
                    converged += 1
            except:
                pass
        
        mass_errors.append(np.mean(m_err) if m_err else 1.0)
        a_errors.append(np.mean(a_err) if a_err else 1.0)
        lambda_errors.append(np.mean(l_err) if l_err else 180.0)
        convergence_rates.append(converged / n_trials)
    
    return SensitivityResult('noise_level', np.array(noise_levels),
                              np.array(mass_errors), np.array(a_errors),
                              np.array(lambda_errors), np.array(convergence_rates))


def sensitivity_to_n_observations(n_obs_values: List[int] = [10, 20, 30, 50, 60],
                                   noise_level: float = 2.0,
                                   n_trials: int = 30,
                                   seed: Optional[int] = 42) -> SensitivityResult:
    """Analyze sensitivity to number of observations."""
    if seed is not None:
        np.random.seed(seed)
    
    true_mass, true_a, true_lambda = 1.024e26, 30.07, 325.0
    mass_errors, a_errors, lambda_errors, convergence_rates = [], [], [], []
    
    for n_obs in n_obs_values:
        times = np.linspace(0, 50, n_obs)
        true_params = np.array([true_mass/1e26, true_a, true_lambda])
        true_signal = forward_model_full(true_params, times, URANUS.mean_motion, URANUS.semi_major_axis)
        
        m_err, a_err, l_err, converged = [], [], [], 0
        
        for _ in range(n_trials):
            observations = true_signal + np.random.normal(0, noise_level, n_obs)
            try:
                solver = LeVerrierSolver(times, observations)
                result = solver.solve(method='scipy')
                if result.converged:
                    m_err.append(abs(result.mass - true_mass) / true_mass)
                    a_err.append(abs(result.semi_major_axis - true_a) / true_a)
                    l_err.append(abs(result.longitude - true_lambda))
                    converged += 1
            except:
                pass
        
        mass_errors.append(np.mean(m_err) if m_err else 1.0)
        a_errors.append(np.mean(a_err) if a_err else 1.0)
        lambda_errors.append(np.mean(l_err) if l_err else 180.0)
        convergence_rates.append(converged / n_trials)
    
    return SensitivityResult('n_observations', np.array(n_obs_values),
                              np.array(mass_errors), np.array(a_errors),
                              np.array(lambda_errors), np.array(convergence_rates))


def print_sensitivity_table(result: SensitivityResult):
    """Print sensitivity results as a formatted table."""
    print(f"\nSensitivity to {result.parameter_name}:")
    print("-" * 65)
    print(f"{'Value':>10} {'Mass Err':>12} {'a Err':>12} {'λ Err':>12} {'Conv':>10}")
    print("-" * 65)
    
    for i in range(len(result.parameter_values)):
        print(f"{result.parameter_values[i]:>10.2f} "
              f"{result.mass_errors[i]*100:>11.1f}% "
              f"{result.a_errors[i]*100:>11.1f}% "
              f"{result.lambda_errors[i]:>11.1f}° "
              f"{result.convergence_rates[i]*100:>9.1f}%")
    print("-" * 65)


def main():
    """Run sensitivity analysis."""
    print("=" * 65)
    print("NEPTUNE PREDICTION: SENSITIVITY ANALYSIS")
    print("=" * 65)
    
    print("\n1. Sensitivity to noise level...")
    noise_result = sensitivity_to_noise(n_trials=20)
    print_sensitivity_table(noise_result)
    
    print("\n2. Sensitivity to number of observations...")
    nobs_result = sensitivity_to_n_observations(n_trials=20)
    print_sensitivity_table(nobs_result)
    
    print("\n" + "=" * 65)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
