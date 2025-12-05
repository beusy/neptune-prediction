#!/usr/bin/env python3
"""
Full Neptune Analysis Pipeline
==============================

Complete analysis including:
- Data loading and preprocessing
- Fourier decomposition
- Inverse problem solution
- Error analysis
- Results visualization

Usage:
    python examples/full_analysis.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverrier import (
    LeVerrierSolver, BouvardData, 
    get_historical_comparison,
    URANUS, NEPTUNE, M_SUN,
    forward_model_full,
    fit_fourier_coefficients,
    assess_phase_coverage,
    synodic_period
)


def analyze_data_quality(observations):
    """Analyze quality of observation data."""
    print("\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    print(f"\nNumber of observations: {len(observations)}")
    print(f"Time baseline: {observations.time_baseline:.1f} years")
    print(f"Year range: {observations.years.min():.1f} - {observations.years.max():.1f}")
    
    print(f"\nDeviation statistics:")
    print(f"  Min: {observations.deviations.min():.2f} arcmin")
    print(f"  Max: {observations.deviations.max():.2f} arcmin")
    print(f"  Mean: {np.mean(observations.deviations):.2f} arcmin")
    print(f"  Std: {np.std(observations.deviations):.2f} arcmin")
    
    print(f"\nUncertainty statistics:")
    print(f"  Min: {observations.uncertainties.min():.2f} arcmin")
    print(f"  Max: {observations.uncertainties.max():.2f} arcmin")
    print(f"  Mean: {np.mean(observations.uncertainties):.2f} arcmin")
    
    n_N_guess = 2 * np.pi / np.sqrt(30.0**3)
    coverage = assess_phase_coverage(
        observations.times, URANUS.mean_motion, n_N_guess
    )
    
    print(f"\nPhase coverage:")
    print(f"  Synodic cycles covered: {coverage['cycles_covered']:.2f}")
    print(f"  Phase uniformity: {coverage['phase_uniformity']:.2f}")
    print(f"  Synodic period: {coverage['synodic_period']:.1f} years")
    
    return coverage


def analyze_fourier_decomposition(solver, observations):
    """Detailed Fourier analysis."""
    print("\n" + "=" * 60)
    print("FOURIER DECOMPOSITION ANALYSIS")
    print("=" * 60)
    
    for K in [1, 2, 3, 4]:
        coeffs = fit_fourier_coefficients(
            observations.times, observations.deviations,
            URANUS.mean_motion, 2*np.pi/np.sqrt(30**3),
            K=K
        )
        print(f"\nK = {K} harmonics:")
        print(f"  RMS residual: {coeffs.residual_rms:.3f} arcmin")
        print(f"  DC component: {coeffs.C0:.3f} arcmin")
        for i, (amp, phase) in enumerate(zip(coeffs.amplitudes, coeffs.phases)):
            print(f"  Harmonic {i+1}: amplitude = {amp:.3f} arcmin, "
                  f"phase = {np.degrees(phase):.1f}°")
    
    fourier = solver.fit_fourier(K=2)
    
    if len(fourier.amplitudes) >= 2:
        ratio = fourier.amplitudes[0] / fourier.amplitudes[1]
        print(f"\nAmplitude ratio (A1/A2): {ratio:.2f}")
        print(f"  Implies a_N/a_U ≈ {ratio:.2f}")
        print(f"  Estimated a_N ≈ {URANUS.semi_major_axis * ratio:.1f} AU")
    
    return fourier


def analyze_inverse_solution(solver):
    """Detailed inverse problem analysis."""
    print("\n" + "=" * 60)
    print("INVERSE PROBLEM SOLUTION")
    print("=" * 60)
    
    print("\n--- Scipy Solver ---")
    result_scipy = solver.solve(method='scipy')
    print(f"Converged: {result_scipy.converged}")
    print(f"Iterations: {result_scipy.iterations}")
    print(f"RMS residual: {result_scipy.rms_residual:.3f} arcmin")
    
    result = result_scipy
    
    print("\n=== FINAL SOLUTION ===")
    print(f"\nMass:")
    print(f"  {result.mass:.4e} kg")
    print(f"  {result.mass_earth:.2f} Earth masses")
    print(f"  {result.mass / NEPTUNE.mass * 100:.1f}% of true Neptune mass")
    print(f"  Uncertainty: ±{result.mass_uncertainty:.2e} kg")
    
    print(f"\nSemi-major axis:")
    print(f"  {result.semi_major_axis:.3f} AU")
    print(f"  True Neptune: {NEPTUNE.semi_major_axis:.3f} AU")
    print(f"  Error: {abs(result.semi_major_axis - NEPTUNE.semi_major_axis):.3f} AU "
          f"({abs(result.semi_major_axis - NEPTUNE.semi_major_axis)/NEPTUNE.semi_major_axis*100:.1f}%)")
    print(f"  Uncertainty: ±{result.semi_major_axis_uncertainty:.3f} AU")
    
    print(f"\nLongitude:")
    print(f"  {result.longitude:.2f}°")
    print(f"  Uncertainty: ±{result.longitude_uncertainty:.2f}°")
    
    return result


def compare_with_history(result):
    """Compare results with historical predictions."""
    print("\n" + "=" * 60)
    print("HISTORICAL COMPARISON")
    print("=" * 60)
    
    hist = get_historical_comparison()
    
    print("\n--- Le Verrier (1846) ---")
    lv = hist['le_verrier']
    print(f"Longitude: {lv['longitude_predicted']}°")
    print(f"Distance: {lv['distance_predicted']} AU")
    print(f"Mass: {lv['mass_predicted']} Earth masses")
    
    print("\n--- Adams (1845) ---")
    adams = hist['adams']
    print(f"Longitude: {adams['longitude_predicted']}°")
    print(f"Distance: {adams['distance_predicted']} AU")
    print(f"Mass: {adams['mass_predicted']} Earth masses")
    
    print("\n--- Actual Neptune ---")
    actual = hist['actual']
    print(f"Longitude at discovery: {actual['longitude_discovery']}°")
    print(f"Semi-major axis: {actual['distance']} AU")
    print(f"Mass: {actual['mass']} Earth masses")
    
    print("\n--- Our Prediction Errors ---")
    print(f"vs Le Verrier longitude: {abs(result.longitude - lv['longitude_predicted']):.2f}°")
    print(f"vs Actual longitude: {abs(result.longitude - actual['longitude_discovery']):.2f}°")
    print(f"vs Actual distance: {abs(result.semi_major_axis - actual['distance']):.2f} AU")


def main():
    print("=" * 60)
    print("NEPTUNE PREDICTION: FULL ANALYSIS PIPELINE")
    print("=" * 60)
    
    print("\nLoading data...")
    data = BouvardData()
    observations = data.get_observations()
    
    coverage = analyze_data_quality(observations)
    
    solver = LeVerrierSolver(
        observations.times,
        observations.deviations,
        observations.uncertainties
    )
    
    fourier = analyze_fourier_decomposition(solver, observations)
    solver._fourier_result = None
    solver._inversion_result = None
    
    result = analyze_inverse_solution(solver)
    compare_with_history(result)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    result = main()
