#!/usr/bin/env python3
"""
Basic Neptune Prediction Example
================================

This script demonstrates the simplest usage of Le Verrier's algorithm
to predict Neptune's position from Uranus observation data.

Usage:
    python examples/basic_prediction.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverrier import LeVerrierSolver, BouvardData, get_historical_comparison


def main():
    print("=" * 60)
    print("Neptune Prediction: Basic Example")
    print("=" * 60)
    
    # Step 1: Load historical observation data
    print("\n1. Loading Bouvard observation data...")
    data = BouvardData()
    observations = data.get_observations()
    
    print(f"   Loaded {len(observations)} observations")
    print(f"   Time span: {observations.years.min():.0f} - {observations.years.max():.0f}")
    print(f"   Deviation range: {observations.deviations.min():.1f} to "
          f"{observations.deviations.max():.1f} arcmin")
    
    # Step 2: Create solver
    print("\n2. Creating Le Verrier solver...")
    solver = LeVerrierSolver(
        times=observations.times,
        deviations=observations.deviations,
        uncertainties=observations.uncertainties
    )
    
    # Step 3: Fit Fourier coefficients
    print("\n3. Fitting Fourier coefficients...")
    fourier = solver.fit_fourier(K=2)
    
    print(f"   DC component (C0): {fourier.C0:.2f} arcmin")
    print(f"   First harmonic amplitude: {fourier.amplitudes[0]:.2f} arcmin")
    print(f"   Second harmonic amplitude: {fourier.amplitudes[1]:.2f} arcmin")
    print(f"   RMS residual: {fourier.residual_rms:.2f} arcmin")
    
    # Step 4: Get initial parameter estimates
    print("\n4. Initial parameter estimates from Fourier analysis...")
    mass_init, a_init, lambda_init = solver.estimate_parameters()
    
    print(f"   Mass: {mass_init:.3e} kg ({mass_init/5.972e24:.1f} Earth masses)")
    print(f"   Semi-major axis: {a_init:.2f} AU")
    print(f"   Longitude: {lambda_init:.1f}°")
    
    # Step 5: Solve the full inverse problem
    print("\n5. Solving inverse problem (Gauss-Newton refinement)...")
    result = solver.solve(method='scipy')
    
    print(f"\n   === SOLUTION ===")
    print(f"   Converged: {result.converged}")
    print(f"   Iterations: {result.iterations}")
    print(f"   RMS residual: {result.rms_residual:.2f} arcmin")
    print()
    print(f"   Mass: {result.mass:.3e} kg")
    print(f"         ({result.mass_earth:.1f} Earth masses)")
    print(f"         Uncertainty: ±{result.mass_uncertainty:.2e} kg")
    print()
    print(f"   Semi-major axis: {result.semi_major_axis:.2f} AU")
    print(f"         Uncertainty: ±{result.semi_major_axis_uncertainty:.2f} AU")
    print()
    print(f"   Longitude: {result.longitude:.1f}°")
    print(f"         Uncertainty: ±{result.longitude_uncertainty:.1f}°")
    
    # Step 6: Compare with historical values
    print("\n6. Comparison with historical values...")
    hist = get_historical_comparison()
    
    print("\n   Le Verrier's 1846 prediction:")
    print(f"      Longitude: {hist['le_verrier']['longitude_predicted']}°")
    print(f"      Distance: {hist['le_verrier']['distance_predicted']} AU")
    
    print("\n   Actual Neptune values:")
    print(f"      Longitude at discovery: {hist['actual']['longitude_discovery']}°")
    print(f"      Semi-major axis: {hist['actual']['distance']} AU")
    print(f"      Mass: {hist['actual']['mass']} Earth masses")
    
    print("\n   Our prediction errors:")
    a_error = abs(result.semi_major_axis - hist['actual']['distance']) / hist['actual']['distance'] * 100
    lambda_error = abs(result.longitude - hist['actual']['longitude_discovery'])
    print(f"      Semi-major axis error: {a_error:.1f}%")
    print(f"      Longitude error: {lambda_error:.1f}°")
    
    print("\n" + "=" * 60)
    print("Prediction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
