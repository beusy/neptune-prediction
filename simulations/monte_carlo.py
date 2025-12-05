"""
Monte Carlo Simulations
=======================

Robustness analysis and sensitivity studies for Le Verrier's method.

Features
--------
- Noise robustness testing
- Parameter sensitivity analysis
- Detection threshold estimation
- Observation requirements study

Usage
-----
>>> from simulations.monte_carlo import run_robustness_study
>>> results = run_robustness_study(n_trials=100, noise_levels=[0.5, 1.0, 2.0])
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverrier import (
    LeVerrierSolver, 
    generate_synthetic_data,
    URANUS, NEPTUNE, M_SUN,
    forward_model_full
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MonteCarloResult:
    """
    Results from Monte Carlo simulation.
    
    Attributes
    ----------
    n_trials : int
        Number of simulation trials
    noise_level : float
        Noise standard deviation (arcmin)
    mass_mean : float
        Mean recovered mass (kg)
    mass_std : float
        Standard deviation of recovered mass (kg)
    mass_bias : float
        Bias in mass recovery (recovered - true) / true
    a_mean : float
        Mean recovered semi-major axis (AU)
    a_std : float
        Standard deviation of semi-major axis (AU)
    a_bias : float
        Bias in semi-major axis recovery
    lambda_mean : float
        Mean recovered longitude (degrees)
    lambda_std : float
        Standard deviation of longitude (degrees)
    convergence_rate : float
        Fraction of trials that converged
    rms_residuals : np.ndarray
        RMS residuals for each trial
    """
    n_trials: int
    noise_level: float
    mass_mean: float
    mass_std: float
    mass_bias: float
    a_mean: float
    a_std: float
    a_bias: float
    lambda_mean: float
    lambda_std: float
    convergence_rate: float
    rms_residuals: np.ndarray
    
    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Monte Carlo Results (n={self.n_trials}, noise={self.noise_level} arcmin)\n"
            f"  Mass: {self.mass_mean:.3e} ± {self.mass_std:.3e} kg (bias: {self.mass_bias*100:.1f}%)\n"
            f"  a: {self.a_mean:.2f} ± {self.a_std:.2f} AU (bias: {self.a_bias*100:.1f}%)\n"
            f"  λ: {self.lambda_mean:.1f} ± {self.lambda_std:.1f}° \n"
            f"  Convergence: {self.convergence_rate*100:.1f}%"
        )


# =============================================================================
# ROBUSTNESS STUDY
# =============================================================================

def run_robustness_study(n_trials: int = 100,
                          noise_levels: List[float] = [0.5, 1.0, 2.0, 5.0],
                          n_obs: int = 60,
                          time_span: float = 50.0,
                          true_mass: float = 1.024e26,
                          true_a: float = 30.07,
                          true_lambda: float = 325.0,
                          seed: Optional[int] = 42,
                          verbose: bool = True) -> List[MonteCarloResult]:
    """
    Run robustness study across multiple noise levels.
    
    Parameters
    ----------
    n_trials : int
        Number of Monte Carlo trials per noise level
    noise_levels : list
        Noise levels to test (arcmin)
    n_obs : int
        Number of observations per trial
    time_span : float
        Observation time span (years)
    true_mass : float
        True mass of perturber (kg)
    true_a : float
        True semi-major axis (AU)
    true_lambda : float
        True longitude (degrees)
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress information
        
    Returns
    -------
    list
        MonteCarloResult for each noise level
    """
    if seed is not None:
        np.random.seed(seed)
    
    results = []
    
    # True parameters
    true_params = np.array([true_mass / 1e26, true_a, true_lambda])
    
    # Generate base times (same for all trials)
    times = np.linspace(0, time_span, n_obs)
    
    # Compute true signal
    true_signal = forward_model_full(true_params, times, URANUS.mean_motion, URANUS.semi_major_axis)
    
    for noise_level in noise_levels:
        if verbose:
            print(f"\nRunning {n_trials} trials with noise = {noise_level} arcmin...")
        
        masses = []
        semi_major_axes = []
        longitudes = []
        converged_count = 0
        rms_residuals = []
        
        for trial in range(n_trials):
            # Add noise to observations
            noise = np.random.normal(0, noise_level, n_obs)
            observations = true_signal + noise
            uncertainties = np.ones(n_obs) * noise_level
            
            try:
                # Solve inverse problem
                solver = LeVerrierSolver(times, observations, uncertainties)
                result = solver.solve(method='scipy')
                
                if result.converged:
                    masses.append(result.mass)
                    semi_major_axes.append(result.semi_major_axis)
                    longitudes.append(result.longitude)
                    rms_residuals.append(result.rms_residual)
                    converged_count += 1
            except Exception as e:
                if verbose and trial == 0:
                    print(f"  Trial {trial} failed: {e}")
        
        # Compute statistics
        masses = np.array(masses)
        semi_major_axes = np.array(semi_major_axes)
        longitudes = np.array(longitudes)
        rms_residuals = np.array(rms_residuals)
        
        if len(masses) > 0:
            mc_result = MonteCarloResult(
                n_trials=n_trials,
                noise_level=noise_level,
                mass_mean=np.mean(masses),
                mass_std=np.std(masses),
                mass_bias=(np.mean(masses) - true_mass) / true_mass,
                a_mean=np.mean(semi_major_axes),
                a_std=np.std(semi_major_axes),
                a_bias=(np.mean(semi_major_axes) - true_a) / true_a,
                lambda_mean=np.mean(longitudes),
                lambda_std=np.std(longitudes),
                convergence_rate=converged_count / n_trials,
                rms_residuals=rms_residuals
            )
        else:
            mc_result = MonteCarloResult(
                n_trials=n_trials,
                noise_level=noise_level,
                mass_mean=0, mass_std=0, mass_bias=1.0,
                a_mean=0, a_std=0, a_bias=1.0,
                lambda_mean=0, lambda_std=0,
                convergence_rate=0,
                rms_residuals=np.array([])
            )
        
        results.append(mc_result)
        
        if verbose:
            print(mc_result.summary())
    
    return results


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_to_parameter(parameter: str,
                              parameter_range: Tuple[float, float],
                              n_points: int = 10,
                              n_trials: int = 50,
                              noise_level: float = 2.0,
                              seed: Optional[int] = 42) -> Dict:
    """
    Analyze sensitivity to a specific parameter.
    
    Parameters
    ----------
    parameter : str
        Parameter to vary: 'mass', 'semi_major_axis', or 'longitude'
    parameter_range : tuple
        (min_value, max_value) for the parameter
    n_points : int
        Number of parameter values to test
    n_trials : int
        Trials per parameter value
    noise_level : float
        Observation noise (arcmin)
    seed : int, optional
        Random seed
        
    Returns
    -------
    dict
        Sensitivity analysis results
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default true values
    true_mass = 1.024e26
    true_a = 30.07
    true_lambda = 325.0
    
    # Parameter values to test
    param_values = np.linspace(parameter_range[0], parameter_range[1], n_points)
    
    # Results storage
    recovery_errors = []
    recovery_stds = []
    
    n_obs = 60
    times = np.linspace(0, 50, n_obs)
    
    for param_val in param_values:
        # Set true parameters
        if parameter == 'mass':
            true_params = np.array([param_val / 1e26, true_a, true_lambda])
            true_value = param_val
        elif parameter == 'semi_major_axis':
            true_params = np.array([true_mass / 1e26, param_val, true_lambda])
            true_value = param_val
        else:  # longitude
            true_params = np.array([true_mass / 1e26, true_a, param_val])
            true_value = param_val
        
        # Generate true signal
        true_signal = forward_model_full(true_params, times, URANUS.mean_motion, URANUS.semi_major_axis)
        
        # Run trials
        recovered = []
        for _ in range(n_trials):
            noise = np.random.normal(0, noise_level, n_obs)
            observations = true_signal + noise
            
            try:
                solver = LeVerrierSolver(times, observations)
                result = solver.solve(method='scipy')
                
                if result.converged:
                    if parameter == 'mass':
                        recovered.append(result.mass)
                    elif parameter == 'semi_major_axis':
                        recovered.append(result.semi_major_axis)
                    else:
                        recovered.append(result.longitude)
            except:
                pass
        
        if len(recovered) > 0:
            recovered = np.array(recovered)
            error = np.abs(np.mean(recovered) - true_value) / true_value
            std = np.std(recovered) / true_value
        else:
            error = 1.0
            std = 1.0
        
        recovery_errors.append(error)
        recovery_stds.append(std)
    
    return {
        'parameter': parameter,
        'values': param_values,
        'recovery_errors': np.array(recovery_errors),
        'recovery_stds': np.array(recovery_stds),
        'noise_level': noise_level,
        'n_trials': n_trials
    }


# =============================================================================
# DETECTION LIMITS
# =============================================================================

def find_detection_limit(noise_level: float = 2.0,
                          a_perturber: float = 30.0,
                          n_trials: int = 50,
                          detection_threshold: float = 3.0,
                          seed: Optional[int] = 42) -> float:
    """
    Find minimum detectable mass at given noise level.
    
    A perturber is considered "detected" if the recovered mass is
    within detection_threshold * uncertainty of the true mass.
    
    Parameters
    ----------
    noise_level : float
        Observation noise (arcmin)
    a_perturber : float
        Semi-major axis of perturber (AU)
    n_trials : int
        Number of trials per mass value
    detection_threshold : float
        Detection significance (sigma)
    seed : int, optional
        Random seed
        
    Returns
    -------
    float
        Minimum detectable mass (kg)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Mass values to test (in kg)
    mass_values = np.logspace(24, 27, 20)  # 10^24 to 10^27 kg
    
    n_obs = 60
    times = np.linspace(0, 50, n_obs)
    
    for mass in mass_values:
        true_params = np.array([mass / 1e26, a_perturber, 325.0])
        true_signal = forward_model_full(true_params, times, URANUS.mean_motion, URANUS.semi_major_axis)
        
        detections = 0
        for _ in range(n_trials):
            noise = np.random.normal(0, noise_level, n_obs)
            observations = true_signal + noise
            
            try:
                solver = LeVerrierSolver(times, observations)
                result = solver.solve(method='scipy')
                
                if result.converged:
                    # Check if detection is significant
                    snr = abs(result.mass) / (result.mass_uncertainty + 1e-30)
                    if snr >= detection_threshold:
                        detections += 1
            except:
                pass
        
        detection_rate = detections / n_trials
        if detection_rate >= 0.5:  # 50% detection rate threshold
            return mass
    
    return mass_values[-1]  # Return largest mass if no detection


# =============================================================================
# OBSERVATION REQUIREMENTS
# =============================================================================

def study_observation_requirements(n_obs_values: List[int] = [10, 20, 30, 50, 60, 100],
                                    noise_level: float = 2.0,
                                    n_trials: int = 50,
                                    seed: Optional[int] = 42) -> Dict:
    """
    Study how number of observations affects recovery accuracy.
    
    Parameters
    ----------
    n_obs_values : list
        Number of observations to test
    noise_level : float
        Observation noise (arcmin)
    n_trials : int
        Number of trials per n_obs value
    seed : int, optional
        Random seed
        
    Returns
    -------
    dict
        Results for each n_obs value
    """
    if seed is not None:
        np.random.seed(seed)
    
    true_mass = 1.024e26
    true_a = 30.07
    true_lambda = 325.0
    true_params = np.array([true_mass / 1e26, true_a, true_lambda])
    
    results = {
        'n_obs': [],
        'mass_error': [],
        'a_error': [],
        'lambda_error': [],
        'convergence_rate': []
    }
    
    for n_obs in n_obs_values:
        times = np.linspace(0, 50, n_obs)
        true_signal = forward_model_full(true_params, times, URANUS.mean_motion, URANUS.semi_major_axis)
        
        mass_errors = []
        a_errors = []
        lambda_errors = []
        converged = 0
        
        for _ in range(n_trials):
            noise = np.random.normal(0, noise_level, n_obs)
            observations = true_signal + noise
            
            try:
                solver = LeVerrierSolver(times, observations)
                result = solver.solve(method='scipy')
                
                if result.converged:
                    mass_errors.append(abs(result.mass - true_mass) / true_mass)
                    a_errors.append(abs(result.semi_major_axis - true_a) / true_a)
                    lambda_errors.append(abs(result.longitude - true_lambda))
                    converged += 1
            except:
                pass
        
        results['n_obs'].append(n_obs)
        results['mass_error'].append(np.mean(mass_errors) if mass_errors else 1.0)
        results['a_error'].append(np.mean(a_errors) if a_errors else 1.0)
        results['lambda_error'].append(np.mean(lambda_errors) if lambda_errors else 180.0)
        results['convergence_rate'].append(converged / n_trials)
    
    return results


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run comprehensive Monte Carlo analysis."""
    print("=" * 60)
    print("Neptune Prediction: Monte Carlo Analysis")
    print("=" * 60)
    
    # Robustness study
    print("\n1. ROBUSTNESS TO NOISE")
    print("-" * 40)
    robustness_results = run_robustness_study(
        n_trials=50,
        noise_levels=[0.5, 1.0, 2.0, 5.0],
        verbose=True
    )
    
    # Observation requirements
    print("\n2. OBSERVATION REQUIREMENTS")
    print("-" * 40)
    obs_results = study_observation_requirements(
        n_obs_values=[10, 20, 30, 50, 60],
        n_trials=30
    )
    
    print("\nObservations vs Recovery Error:")
    for n, err in zip(obs_results['n_obs'], obs_results['a_error']):
        print(f"  N={n:3d}: a error = {err*100:.1f}%")
    
    # Detection limits
    print("\n3. DETECTION LIMITS")
    print("-" * 40)
    for noise in [2.0, 1.0, 0.5]:
        mass_limit = find_detection_limit(noise_level=noise, n_trials=30)
        print(f"  Noise = {noise} arcmin: min detectable mass = {mass_limit:.2e} kg")
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
