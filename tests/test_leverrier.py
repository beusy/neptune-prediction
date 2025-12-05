"""
Unit Tests for Neptune Prediction Package
=========================================

Comprehensive test suite covering:
- Constants and physical values
- Orbital mechanics calculations
- Perturbation theory
- Fourier decomposition
- Inverse problem solver
- Data handling

Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverrier import (
    # Constants
    G, M_SUN, AU, URANUS, NEPTUNE,
    DEG_TO_RAD, RAD_TO_DEG,
    period_from_semi_major_axis,
    mean_motion_from_semi_major_axis,
    
    # Orbital mechanics
    OrbitalElements,
    solve_kepler_equation,
    eccentric_to_true_anomaly,
    orbital_radius,
    synodic_period,
    
    # Perturbation
    evaluate_legendre,
    inverse_distance_legendre,
    
    # Fourier
    FourierCoefficients,
    build_design_matrix,
    fit_fourier_coefficients,
    harmonic_amplitude,
    
    # Inverse
    LeVerrierSolver,
    forward_model_full,
    
    # Data
    BouvardData,
    generate_synthetic_data,
    get_historical_comparison
)


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestConstants:
    """Test physical constants and planetary data."""
    
    def test_gravitational_constant(self):
        """G should be approximately 6.67e-11."""
        assert 6.67e-11 < G < 6.68e-11
    
    def test_solar_mass(self):
        """Solar mass should be approximately 2e30 kg."""
        assert 1.98e30 < M_SUN < 2.0e30
    
    def test_au_value(self):
        """AU should be approximately 1.5e11 meters."""
        assert 1.49e11 < AU < 1.50e11
    
    def test_uranus_data(self):
        """Uranus orbital data should be reasonable."""
        assert 19.0 < URANUS.semi_major_axis < 19.5
        assert 83 < URANUS.orbital_period < 85
        assert 0.04 < URANUS.eccentricity < 0.05
    
    def test_neptune_data(self):
        """Neptune orbital data should be reasonable."""
        assert 30.0 < NEPTUNE.semi_major_axis < 30.2
        assert 164 < NEPTUNE.orbital_period < 166
        assert 0.008 < NEPTUNE.eccentricity < 0.01
    
    def test_period_from_semi_major_axis(self):
        """Test Kepler's third law."""
        # Earth: a = 1 AU, T = 1 year
        assert abs(period_from_semi_major_axis(1.0) - 1.0) < 0.001
        
        # Uranus
        T_uranus = period_from_semi_major_axis(URANUS.semi_major_axis)
        assert abs(T_uranus - URANUS.orbital_period) < 0.1
    
    def test_mean_motion_consistency(self):
        """Mean motion should be 2π/T."""
        n = mean_motion_from_semi_major_axis(URANUS.semi_major_axis)
        T = period_from_semi_major_axis(URANUS.semi_major_axis)
        assert abs(n - 2 * np.pi / T) < 1e-6


# =============================================================================
# ORBITAL MECHANICS TESTS
# =============================================================================

class TestOrbitalMechanics:
    """Test orbital mechanics calculations."""
    
    def test_kepler_equation_circular(self):
        """For e=0, E should equal M."""
        M = np.pi / 4
        E = solve_kepler_equation(M, e=0.0)
        assert abs(E - M) < 1e-10
    
    def test_kepler_equation_small_e(self):
        """Test Kepler equation for small eccentricity."""
        M = np.pi / 3
        e = 0.05
        E = solve_kepler_equation(M, e)
        
        # Verify: M = E - e*sin(E)
        residual = abs(E - e * np.sin(E) - M)
        assert residual < 1e-10
    
    def test_kepler_equation_large_e(self):
        """Test Kepler equation for larger eccentricity."""
        M = np.pi / 2
        e = 0.5
        E = solve_kepler_equation(M, e)
        
        residual = abs(E - e * np.sin(E) - M)
        assert residual < 1e-10
    
    def test_anomaly_conversion_circular(self):
        """For e=0, true anomaly equals eccentric anomaly."""
        E = np.pi / 3
        theta = eccentric_to_true_anomaly(E, e=0.0)
        assert abs(theta - E) < 1e-10
    
    def test_orbital_radius_perihelion(self):
        """At perihelion (θ=0), r = a(1-e)."""
        a, e = 30.0, 0.1
        r = orbital_radius(a, e, theta=0.0)
        expected = a * (1 - e)
        assert abs(r - expected) < 1e-10
    
    def test_orbital_radius_aphelion(self):
        """At aphelion (θ=π), r = a(1+e)."""
        a, e = 30.0, 0.1
        r = orbital_radius(a, e, theta=np.pi)
        expected = a * (1 + e)
        assert abs(r - expected) < 1e-10
    
    def test_synodic_period(self):
        """Test synodic period calculation."""
        n1 = 2 * np.pi  # 1 cycle/year
        n2 = np.pi      # 0.5 cycles/year
        P_syn = synodic_period(n1, n2)
        expected = 2 * np.pi / np.pi  # = 2 years
        assert abs(P_syn - expected) < 1e-10
    
    def test_orbital_elements(self):
        """Test OrbitalElements class."""
        elem = OrbitalElements(a=30.0, e=0.01, M0=0.0)
        
        assert abs(elem.period - np.sqrt(30.0**3)) < 0.1
        assert abs(elem.mean_motion - 2*np.pi/elem.period) < 1e-6


# =============================================================================
# PERTURBATION THEORY TESTS
# =============================================================================

class TestPerturbation:
    """Test perturbation theory calculations."""
    
    def test_legendre_p0(self):
        """P_0(x) = 1 for all x."""
        for x in [-1, 0, 0.5, 1]:
            assert abs(evaluate_legendre(0, x) - 1.0) < 1e-10
    
    def test_legendre_p1(self):
        """P_1(x) = x."""
        for x in [-1, 0, 0.5, 1]:
            assert abs(evaluate_legendre(1, x) - x) < 1e-10
    
    def test_legendre_p2(self):
        """P_2(x) = (3x² - 1)/2."""
        for x in [-1, 0, 0.5, 1]:
            expected = (3 * x**2 - 1) / 2
            assert abs(evaluate_legendre(2, x) - expected) < 1e-10
    
    def test_legendre_at_one(self):
        """P_n(1) = 1 for all n."""
        for n in range(10):
            assert abs(evaluate_legendre(n, 1.0) - 1.0) < 1e-10
    
    def test_inverse_distance_expansion(self):
        """Test Legendre expansion of 1/Δ at conjunction."""
        r_inner = 19.19
        r_outer = 30.07
        x = r_inner / r_outer
        
        # At conjunction (cos θ = 1), exact value is 1/(r_outer - r_inner)
        exact = 1.0 / (r_outer - r_inner)
        
        # Legendre expansion
        approx = inverse_distance_legendre(r_inner, r_outer, cos_theta=1.0, n_terms=20)
        
        # Should converge to exact value
        rel_error = abs(approx - exact) / exact
        assert rel_error < 0.05  # Within 5%


# =============================================================================
# FOURIER ANALYSIS TESTS
# =============================================================================

class TestFourier:
    """Test Fourier decomposition."""
    
    def test_design_matrix_shape(self):
        """Design matrix should have correct shape."""
        times = np.linspace(0, 50, 60)
        K = 2
        A = build_design_matrix(times, n_U=0.075, n_N=0.038, K=K)
        
        assert A.shape == (60, 2*K + 1)
    
    def test_design_matrix_dc_column(self):
        """First column should be all ones."""
        times = np.linspace(0, 50, 60)
        A = build_design_matrix(times, n_U=0.075, n_N=0.038, K=2)
        
        assert np.allclose(A[:, 0], 1.0)
    
    def test_fourier_fit_noiseless(self):
        """Fourier fit should recover exact coefficients without noise."""
        np.random.seed(42)
        
        times = np.linspace(0, 50, 100)
        n_U, n_N = 0.075, 0.038
        phi = (n_U - n_N) * times
        
        # True coefficients
        C0, C1, S1, C2, S2 = 1.0, 2.0, 1.5, 0.8, 0.4
        
        # Generate exact signal
        signal = C0 + C1*np.cos(phi) + S1*np.sin(phi) + C2*np.cos(2*phi) + S2*np.sin(2*phi)
        
        # Fit
        result = fit_fourier_coefficients(times, signal, n_U, n_N, K=2)
        
        assert abs(result.C0 - C0) < 0.01
        assert abs(result.C[0] - C1) < 0.01
        assert abs(result.S[0] - S1) < 0.01
    
    def test_harmonic_amplitude(self):
        """Test amplitude calculation."""
        C, S = 3.0, 4.0
        amp = harmonic_amplitude(C, S)
        assert abs(amp - 5.0) < 1e-10  # 3-4-5 triangle
    
    def test_fourier_fit_with_noise(self):
        """Fourier fit should approximately recover coefficients with noise."""
        np.random.seed(42)
        
        times = np.linspace(0, 50, 100)
        n_U, n_N = 0.075, 0.038
        phi = (n_U - n_N) * times
        
        C0, C1, S1 = 5.0, 3.0, 2.0
        signal = C0 + C1*np.cos(phi) + S1*np.sin(phi)
        signal += np.random.normal(0, 0.5, len(times))  # Small noise
        
        result = fit_fourier_coefficients(times, signal, n_U, n_N, K=2)
        
        # Should recover first harmonic amplitude reasonably well
        true_amp1 = np.sqrt(C1**2 + S1**2)
        assert abs(result.amplitudes[0] - true_amp1) / true_amp1 < 0.3


# =============================================================================
# INVERSE PROBLEM TESTS
# =============================================================================

class TestInverse:
    """Test inverse problem solver."""
    
    def test_forward_model_zero_mass(self):
        """Zero mass should produce zero deviations."""
        times = np.linspace(0, 50, 60)
        params = np.array([0.0, 30.0, 325.0])  # Zero mass
        
        deviations = forward_model_full(params, times, URANUS.mean_motion, URANUS.semi_major_axis)
        
        # Should be very small (numerical noise only)
        assert np.max(np.abs(deviations)) < 1e-10
    
    def test_solver_convergence(self):
        """Solver should converge on synthetic data."""
        np.random.seed(42)
        
        # Generate synthetic data
        data = generate_synthetic_data(n_obs=60, noise_level=1.0, seed=42)
        
        # Solve
        solver = LeVerrierSolver(data.times, data.deviations)
        result = solver.solve(method='scipy')
        
        # Should converge
        assert result.converged
    
    def test_solver_recovery_accuracy(self):
        """Solver should recover parameters accurately on noiseless data."""
        np.random.seed(42)
        
        # True parameters
        true_mass = 1.024e26
        true_a = 30.07
        true_lambda = 325.0
        
        times = np.linspace(0, 50, 100)
        true_params = np.array([true_mass / 1e26, true_a, true_lambda])
        
        # Generate noiseless observations
        observations = forward_model_full(true_params, times, URANUS.mean_motion, URANUS.semi_major_axis)
        
        # Solve
        solver = LeVerrierSolver(times, observations)
        result = solver.solve(method='scipy')
        
        # Should recover within 10%
        assert abs(result.semi_major_axis - true_a) / true_a < 0.1


# =============================================================================
# DATA HANDLING TESTS
# =============================================================================

class TestData:
    """Test data handling."""
    
    def test_bouvard_data_exists(self):
        """Bouvard data should be loadable."""
        data = BouvardData()
        obs = data.get_observations()
        
        assert len(obs) > 0
    
    def test_bouvard_data_range(self):
        """Bouvard data should span expected time range."""
        data = BouvardData()
        obs = data.get_observations()
        
        assert obs.years.min() < 1790
        assert obs.years.max() > 1845
    
    def test_bouvard_data_values(self):
        """Bouvard deviations should be in reasonable range."""
        data = BouvardData()
        obs = data.get_observations()
        
        # Deviations should grow from ~0 to ~25 arcmin
        assert obs.deviations.min() < 0
        assert obs.deviations.max() > 20
    
    def test_synthetic_data_generation(self):
        """Synthetic data generation should work."""
        data = generate_synthetic_data(n_obs=50, noise_level=2.0, seed=42)
        
        assert len(data) == 50
        assert data.time_baseline > 0
    
    def test_historical_comparison(self):
        """Historical comparison data should exist."""
        hist = get_historical_comparison()
        
        assert 'le_verrier' in hist
        assert 'adams' in hist
        assert 'actual' in hist
        
        # Le Verrier's prediction should be close to actual
        error = abs(hist['le_verrier']['longitude_predicted'] - 
                   hist['actual']['longitude_discovery'])
        assert error < 5  # Within 5 degrees


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test complete analysis pipeline."""
        # Load data
        data = BouvardData()
        obs = data.get_observations()
        
        # Create solver
        solver = LeVerrierSolver(obs.times, obs.deviations, obs.uncertainties)
        
        # Fit Fourier
        fourier = solver.fit_fourier(K=2)
        assert fourier.n_harmonics == 2
        
        # Get initial estimates
        m0, a0, lam0 = solver.estimate_parameters()
        assert m0 > 0
        assert 20 < a0 < 50
        
        # Full solution
        result = solver.solve()
        assert result.semi_major_axis > 20
        assert result.mass > 0
    
    def test_synthetic_recovery(self):
        """Test recovery on synthetic data with known parameters."""
        np.random.seed(123)
        
        # Known parameters
        true_mass = 1.0e26
        true_a = 30.0
        true_lambda = 320.0
        
        times = np.linspace(0, 50, 80)
        true_params = np.array([true_mass / 1e26, true_a, true_lambda])
        
        # Generate observations with moderate noise
        true_signal = forward_model_full(true_params, times, URANUS.mean_motion, URANUS.semi_major_axis)
        noise = np.random.normal(0, 1.5, len(times))
        observations = true_signal + noise
        
        # Solve
        solver = LeVerrierSolver(times, observations)
        result = solver.solve(method='scipy')
        
        # Check convergence
        assert result.converged
        
        # Check approximate recovery (within 20% for moderate noise)
        a_error = abs(result.semi_major_axis - true_a) / true_a
        assert a_error < 0.2


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
