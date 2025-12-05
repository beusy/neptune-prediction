#!/usr/bin/env python3
"""
Neptune Prediction Visualizations
=================================

Generate publication-quality plots for:
- Observation data and fit
- Residual analysis
- Parameter recovery
- Historical comparison

Usage:
    python examples/visualizations.py
    
Output:
    Saves figures to examples/figures/
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Ellipse
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

from leverrier import (
    LeVerrierSolver, BouvardData,
    get_historical_comparison,
    URANUS, NEPTUNE,
    forward_model_full,
    evaluate_fourier_model
)


def setup_style():
    """Set up matplotlib style for publication quality."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_observations(observations, solver, save_path=None):
    """Plot raw observations and fitted model."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot_observations (matplotlib not available)")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Get fit
    fourier = solver.fit_fourier(K=2)
    result = solver.solve(method='scipy')
    
    # Fine time grid for model
    t_fine = np.linspace(observations.times.min(), observations.times.max(), 500)
    years_fine = observations.epoch + t_fine
    
    # Evaluate model
    params = np.array([result.mass/1e26, result.semi_major_axis, result.longitude])
    model_fine = forward_model_full(params, t_fine, URANUS.mean_motion, URANUS.semi_major_axis)
    
    # Top panel: Data and fit
    ax1 = axes[0]
    ax1.errorbar(observations.years, observations.deviations, 
                 yerr=observations.uncertainties,
                 fmt='o', markersize=4, capsize=2, alpha=0.7,
                 label='Bouvard observations')
    ax1.plot(years_fine, model_fine, 'r-', linewidth=2, label='Best fit model')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Longitude Deviation (arcmin)')
    ax1.set_title("Uranus Longitude Deviations (1781-1846)")
    ax1.legend(loc='upper left')
    ax1.set_ylim(-5, 30)
    
    # Bottom panel: Residuals
    ax2 = axes[1]
    model_at_obs = forward_model_full(params, observations.times, 
                                       URANUS.mean_motion, URANUS.semi_major_axis)
    residuals = observations.deviations - model_at_obs
    
    ax2.errorbar(observations.years, residuals,
                 yerr=observations.uncertainties,
                 fmt='o', markersize=4, capsize=2, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='-', linewidth=1)
    ax2.fill_between([observations.years.min(), observations.years.max()],
                     -2, 2, alpha=0.2, color='gray', label='±2 arcmin')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Residual (arcmin)')
    ax2.set_title(f'Fit Residuals (RMS = {result.rms_residual:.2f} arcmin)')
    ax2.set_ylim(-8, 8)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_fourier_components(observations, solver, save_path=None):
    """Plot Fourier decomposition of the signal."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot_fourier_components (matplotlib not available)")
        return
    
    fourier = solver.fit_fourier(K=3)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    t_fine = np.linspace(observations.times.min(), observations.times.max(), 500)
    n_N_guess = 2 * np.pi / np.sqrt(30**3)
    phi = (URANUS.mean_motion - n_N_guess) * t_fine
    years_fine = observations.epoch + t_fine
    
    # Top left: Original data
    ax = axes[0, 0]
    ax.plot(observations.years, observations.deviations, 'o', markersize=4, alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Deviation (arcmin)')
    ax.set_title('Original Data')
    
    # Top right: DC + secular
    ax = axes[0, 1]
    dc_component = np.ones_like(t_fine) * fourier.C0
    ax.plot(years_fine, dc_component, 'b-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Deviation (arcmin)')
    ax.set_title(f'DC Component: C₀ = {fourier.C0:.2f} arcmin')
    
    # Bottom left: First harmonic
    ax = axes[1, 0]
    h1 = fourier.C[0] * np.cos(phi) + fourier.S[0] * np.sin(phi)
    ax.plot(years_fine, h1, 'g-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Deviation (arcmin)')
    ax.set_title(f'First Harmonic: A₁ = {fourier.amplitudes[0]:.2f} arcmin')
    
    # Bottom right: Second harmonic
    ax = axes[1, 1]
    h2 = fourier.C[1] * np.cos(2*phi) + fourier.S[1] * np.sin(2*phi)
    ax.plot(years_fine, h2, 'm-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Deviation (arcmin)')
    ax.set_title(f'Second Harmonic: A₂ = {fourier.amplitudes[1]:.2f} arcmin')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_historical_comparison(result, save_path=None):
    """Plot comparison with historical predictions."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot_historical_comparison (matplotlib not available)")
        return
    
    hist = get_historical_comparison()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Semi-major axis comparison
    ax = axes[0]
    methods = ['Le Verrier\n(1846)', 'Adams\n(1845)', 'This Work', 'Actual\nNeptune']
    values = [
        hist['le_verrier']['distance_predicted'],
        hist['adams']['distance_predicted'],
        result.semi_major_axis,
        hist['actual']['distance']
    ]
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
    
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=hist['actual']['distance'], color='crimson', linestyle='--', 
               linewidth=2, alpha=0.7, label='True value')
    ax.set_ylabel('Semi-major Axis (AU)')
    ax.set_title('Semi-major Axis Predictions')
    ax.set_ylim(25, 40)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Right: Longitude comparison
    ax = axes[1]
    methods = ['Le Verrier', 'Adams', 'This Work', 'Actual']
    values = [
        hist['le_verrier']['longitude_predicted'],
        hist['adams']['longitude_predicted'],
        result.longitude,
        hist['actual']['longitude_discovery']
    ]
    
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=hist['actual']['longitude_discovery'], color='crimson', 
               linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel('Longitude (degrees)')
    ax.set_title('Longitude Predictions at Discovery')
    ax.set_ylim(320, 335)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}°', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_orbital_diagram(result, save_path=None):
    """Plot orbital diagram showing Uranus, Neptune, and prediction."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot_orbital_diagram (matplotlib not available)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw orbits
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Sun
    ax.plot(0, 0, 'yo', markersize=20, label='Sun')
    
    # Uranus orbit
    r_U = URANUS.semi_major_axis
    x_U = r_U * np.cos(theta)
    y_U = r_U * np.sin(theta)
    ax.plot(x_U, y_U, 'c-', linewidth=2, label=f'Uranus (a = {r_U:.1f} AU)')
    
    # Neptune orbit (actual)
    r_N = NEPTUNE.semi_major_axis
    x_N = r_N * np.cos(theta)
    y_N = r_N * np.sin(theta)
    ax.plot(x_N, y_N, 'b-', linewidth=2, label=f'Neptune actual (a = {r_N:.1f} AU)')
    
    # Neptune orbit (predicted)
    r_pred = result.semi_major_axis
    x_pred = r_pred * np.cos(theta)
    y_pred = r_pred * np.sin(theta)
    ax.plot(x_pred, y_pred, 'g--', linewidth=2, 
            label=f'Neptune predicted (a = {r_pred:.1f} AU)')
    
    # Mark predicted position
    lambda_pred = np.radians(result.longitude)
    x_mark = r_pred * np.cos(lambda_pred)
    y_mark = r_pred * np.sin(lambda_pred)
    ax.plot(x_mark, y_mark, 'g*', markersize=20, label='Predicted position')
    
    # Mark actual position at discovery
    lambda_actual = np.radians(326.2)
    x_actual = r_N * np.cos(lambda_actual)
    y_actual = r_N * np.sin(lambda_actual)
    ax.plot(x_actual, y_actual, 'b*', markersize=20, label='Actual position')
    
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_title('Solar System Orbits: Neptune Prediction')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def main():
    """Generate all visualizations."""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib required for visualizations")
        print("Install with: pip install matplotlib")
        return
    
    print("=" * 60)
    print("Neptune Prediction Visualizations")
    print("=" * 60)
    
    # Setup
    setup_style()
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load data and solve
    print("\nLoading data and solving...")
    data = BouvardData()
    observations = data.get_observations()
    
    solver = LeVerrierSolver(
        observations.times,
        observations.deviations,
        observations.uncertainties
    )
    result = solver.solve(method='scipy')
    
    print(f"\nSolution:")
    print(f"  Semi-major axis: {result.semi_major_axis:.2f} AU")
    print(f"  Longitude: {result.longitude:.1f}°")
    
    # Generate plots
    print("\nGenerating plots...")
    
    print("  1. Observations and fit...")
    plot_observations(observations, solver, 
                      os.path.join(fig_dir, 'observations_fit.png'))
    
    print("  2. Fourier components...")
    solver._fourier_result = None
    plot_fourier_components(observations, solver,
                            os.path.join(fig_dir, 'fourier_components.png'))
    
    print("  3. Historical comparison...")
    plot_historical_comparison(result,
                               os.path.join(fig_dir, 'historical_comparison.png'))
    
    print("  4. Orbital diagram...")
    plot_orbital_diagram(result,
                         os.path.join(fig_dir, 'orbital_diagram.png'))
    
    print(f"\nAll figures saved to: {fig_dir}")
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    
    # Show plots if running interactively
    plt.show()


if __name__ == "__main__":
    main()
