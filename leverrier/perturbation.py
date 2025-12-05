"""
Perturbation Theory Module
==========================

Implements gravitational perturbation theory including:
- Legendre polynomial expansion of the disturbing function
- First and higher-order perturbation calculations
- Longitude deviation models

References
----------
- Murray & Dermott (1999): Solar System Dynamics, Ch. 6
- Brouwer & Clemence (1961): Methods of Celestial Mechanics
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from scipy.special import legendre
from functools import lru_cache

from .constants import M_SUN, ARCMIN_TO_RAD, RAD_TO_ARCMIN


# =============================================================================
# LEGENDRE POLYNOMIALS
# =============================================================================

@lru_cache(maxsize=64)
def legendre_polynomial(n: int) -> Callable[[float], float]:
    """
    Get Legendre polynomial P_n(x) as a callable function.
    
    Parameters
    ----------
    n : int
        Degree of the polynomial
        
    Returns
    -------
    callable
        Function that evaluates P_n(x)
    """
    return legendre(n)


def evaluate_legendre(n: int, x: float) -> float:
    """
    Evaluate Legendre polynomial P_n(x).
    
    Parameters
    ----------
    n : int
        Degree of the polynomial
    x : float
        Argument (typically cos(θ) where θ is angle between position vectors)
        
    Returns
    -------
    float
        Value of P_n(x)
    """
    P = legendre_polynomial(n)
    return float(P(x))


def legendre_series_coefficients(n_max: int) -> np.ndarray:
    """
    Compute Legendre polynomial coefficients for expansion.
    
    This returns the coefficients c_k such that:
    P_n(x) = sum_k c_k * x^k
    
    Parameters
    ----------
    n_max : int
        Maximum degree
        
    Returns
    -------
    np.ndarray
        Array of coefficient arrays, shape (n_max+1, n_max+1)
    """
    coeffs = np.zeros((n_max + 1, n_max + 1))
    for n in range(n_max + 1):
        P = legendre(n)
        coeffs[n, :n+1] = P.coefficients[::-1]
    return coeffs


# =============================================================================
# INVERSE DISTANCE EXPANSION
# =============================================================================

def inverse_distance_legendre(r_inner: float, r_outer: float, 
                               cos_theta: float, n_terms: int = 10) -> float:
    """
    Compute 1/Δ using Legendre polynomial expansion.
    
    For two bodies at distances r_inner and r_outer from the central body,
    with angle θ between their position vectors:
    
    1/Δ = (1/r_outer) * Σ (r_inner/r_outer)^n * P_n(cos θ)
    
    Parameters
    ----------
    r_inner : float
        Distance of inner body from central body
    r_outer : float
        Distance of outer body from central body
    cos_theta : float
        Cosine of angle between position vectors
    n_terms : int
        Number of terms in the expansion
        
    Returns
    -------
    float
        Approximation of 1/Δ
    """
    x = r_inner / r_outer
    result = 0.0
    x_power = 1.0
    
    for n in range(n_terms):
        P_n = evaluate_legendre(n, cos_theta)
        result += P_n * x_power
        x_power *= x
    
    return result / r_outer


def inverse_distance_exact(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    Compute exact 1/|r1 - r2|.
    
    Parameters
    ----------
    r1 : np.ndarray
        Position vector 1
    r2 : np.ndarray
        Position vector 2
        
    Returns
    -------
    float
        1/|r1 - r2|
    """
    delta = np.linalg.norm(r1 - r2)
    return 1.0 / delta


def convergence_analysis(x: float, n_terms: int = 20) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Analyze convergence of Legendre series at conjunction (cos θ = 1).
    
    At conjunction, P_n(1) = 1 for all n, so the series becomes:
    1/Δ = (1/r_outer) * Σ x^n = (1/r_outer) * 1/(1-x)
    
    Parameters
    ----------
    x : float
        Ratio r_inner/r_outer
    n_terms : int
        Number of terms to compute
        
    Returns
    -------
    tuple
        (partial_sums, truncation_errors, exact_value)
    """
    # Exact value at conjunction
    exact = 1.0 / (1.0 - x)
    
    # Compute partial sums
    partial_sums = np.zeros(n_terms)
    cumsum = 0.0
    x_power = 1.0
    
    for n in range(n_terms):
        cumsum += x_power  # P_n(1) = 1
        partial_sums[n] = cumsum
        x_power *= x
    
    # Truncation errors
    truncation_errors = (exact - partial_sums) / exact
    
    return partial_sums, truncation_errors, exact


# =============================================================================
# DISTURBING FUNCTION
# =============================================================================

def disturbing_function_direct(r_U: float, r_N: float, cos_theta: float,
                                m_N: float, n_terms: int = 10) -> float:
    """
    Direct part of the disturbing function.
    
    R_direct = G * m_N / Δ
    
    Parameters
    ----------
    r_U : float
        Distance of Uranus from Sun (AU)
    r_N : float
        Distance of Neptune from Sun (AU)
    cos_theta : float
        Cosine of angle between position vectors
    m_N : float
        Mass of Neptune (kg)
    n_terms : int
        Number of terms in Legendre expansion
        
    Returns
    -------
    float
        Direct disturbing function (in appropriate units)
    """
    inv_delta = inverse_distance_legendre(r_U, r_N, cos_theta, n_terms)
    return m_N * inv_delta


def disturbing_function_indirect(r_U: float, r_N: float, cos_theta: float,
                                  m_N: float) -> float:
    """
    Indirect part of the disturbing function (reference frame correction).
    
    R_indirect = -G * m_N * r_U * cos(θ) / r_N²
    
    This term accounts for the acceleration of the Sun due to Neptune.
    
    Parameters
    ----------
    r_U : float
        Distance of Uranus from Sun
    r_N : float  
        Distance of Neptune from Sun
    cos_theta : float
        Cosine of angle between position vectors
    m_N : float
        Mass of Neptune
        
    Returns
    -------
    float
        Indirect disturbing function
    """
    return -m_N * r_U * cos_theta / (r_N**2)


def disturbing_function_total(r_U: float, r_N: float, cos_theta: float,
                               m_N: float, n_terms: int = 10) -> float:
    """
    Total disturbing function (direct + indirect).
    
    Note: The dipole terms (n=1) cancel between direct and indirect!
    This is a fundamental result of perturbation theory.
    
    Parameters
    ----------
    r_U : float
        Distance of Uranus from Sun
    r_N : float
        Distance of Neptune from Sun
    cos_theta : float
        Cosine of angle between position vectors
    m_N : float
        Mass of Neptune
    n_terms : int
        Number of terms in Legendre expansion
        
    Returns
    -------
    float
        Total disturbing function
    """
    R_direct = disturbing_function_direct(r_U, r_N, cos_theta, m_N, n_terms)
    R_indirect = disturbing_function_indirect(r_U, r_N, cos_theta, m_N)
    return R_direct + R_indirect


# =============================================================================
# LONGITUDE DEVIATION MODEL
# =============================================================================

def longitude_deviation_first_order(t: float, m_N: float, a_U: float, a_N: float,
                                     n_U: float, n_N: float,
                                     lambda_U0: float = 0.0, 
                                     lambda_N0: float = 0.0) -> float:
    """
    First-order estimate of longitude deviation.
    
    This is a simplified estimate:
    dλ/dt ~ (m_N/M_sun) * (a_U/a_N)³
    
    Parameters
    ----------
    t : float
        Time in years
    m_N : float
        Mass of Neptune (kg)
    a_U : float
        Semi-major axis of Uranus (AU)
    a_N : float
        Semi-major axis of Neptune (AU)
    n_U : float
        Mean motion of Uranus (rad/year)
    n_N : float
        Mean motion of Neptune (rad/year)
    lambda_U0 : float
        Initial longitude of Uranus (rad)
    lambda_N0 : float
        Initial longitude of Neptune (rad)
        
    Returns
    -------
    float
        Longitude deviation in radians
    """
    mass_ratio = m_N / M_SUN
    distance_ratio = (a_U / a_N)**3
    rate = mass_ratio * distance_ratio  # rad/year
    return rate * t


def longitude_deviation_model(t: float, params: np.ndarray, 
                               n_U: float, n_N: float,
                               lambda_U0: float = 0.0) -> float:
    """
    Fourier model of longitude deviation.
    
    Δλ(t) = C_0 + Σ [C_k cos(k*φ) + S_k sin(k*φ)]
    
    where φ = (n_U - n_N)*t + (λ_U0 - λ_N0)
    
    Parameters
    ----------
    t : float or np.ndarray
        Time(s) in years
    params : np.ndarray
        Fourier coefficients [C_0, C_1, S_1, C_2, S_2, ...]
    n_U : float
        Mean motion of Uranus (rad/year)
    n_N : float
        Mean motion of Neptune (rad/year)
    lambda_U0 : float
        Initial longitude of Uranus (rad)
        
    Returns
    -------
    float or np.ndarray
        Longitude deviation(s) in same units as params (typically arcmin)
    """
    t = np.atleast_1d(t)
    
    # Synodic phase
    phi = (n_U - n_N) * t
    
    # Start with DC component
    result = params[0] * np.ones_like(t)
    
    # Add harmonic terms
    k = 1
    idx = 1
    while idx + 1 < len(params):
        C_k = params[idx]
        S_k = params[idx + 1]
        result += C_k * np.cos(k * phi) + S_k * np.sin(k * phi)
        k += 1
        idx += 2
    
    return result.squeeze()


# =============================================================================
# PERTURBATION AMPLITUDE SCALING
# =============================================================================

def amplitude_scaling(k: int, m_N: float, a_U: float, a_N: float,
                      alpha: float = 1.0) -> float:
    """
    Theoretical scaling of k-th Fourier harmonic amplitude.
    
    Amplitude_k ∝ (m_N/M_sun) * (a_U/a_N)^(k+1)
    
    Parameters
    ----------
    k : int
        Harmonic number (1, 2, 3, ...)
    m_N : float
        Mass of perturbing body (kg)
    a_U : float
        Semi-major axis of perturbed body (AU)
    a_N : float
        Semi-major axis of perturbing body (AU)
    alpha : float
        Proportionality constant (depends on orbital geometry)
        
    Returns
    -------
    float
        Expected amplitude scaling
    """
    mass_ratio = m_N / M_SUN
    distance_ratio = a_U / a_N
    return alpha * mass_ratio * (distance_ratio ** (k + 1))


def semi_major_axis_from_amplitude_ratio(amp1: float, amp2: float, 
                                          a_U: float) -> float:
    """
    Estimate semi-major axis of perturber from amplitude ratio.
    
    From: Amp_1/Amp_2 = (a_U/a_N)^(-1)
    
    Therefore: a_N = a_U * (Amp_1/Amp_2)
    
    Parameters
    ----------
    amp1 : float
        First harmonic amplitude
    amp2 : float
        Second harmonic amplitude
    a_U : float
        Semi-major axis of perturbed body (AU)
        
    Returns
    -------
    float
        Estimated semi-major axis of perturber (AU)
    """
    if amp2 == 0:
        raise ValueError("Second harmonic amplitude cannot be zero")
    ratio = amp1 / amp2
    return a_U * ratio


def mass_from_amplitude(amp1: float, a_U: float, a_N: float, 
                        alpha: float = 1.0) -> float:
    """
    Estimate mass of perturber from first harmonic amplitude.
    
    From: Amp_1 = α * (m_N/M_sun) * (a_U/a_N)²
    
    Therefore: m_N = M_sun * Amp_1 / (α * (a_U/a_N)²)
    
    Parameters
    ----------
    amp1 : float
        First harmonic amplitude (arcmin)
    a_U : float
        Semi-major axis of perturbed body (AU)
    a_N : float
        Semi-major axis of perturber (AU)
    alpha : float
        Proportionality constant
        
    Returns
    -------
    float
        Estimated mass of perturber (kg)
    """
    distance_ratio_sq = (a_U / a_N)**2
    mass_ratio = amp1 / (alpha * distance_ratio_sq)
    return M_SUN * mass_ratio


# =============================================================================
# PERTURBATION FORCE CALCULATIONS
# =============================================================================

def perturbation_force_radial(r_U: float, r_N: float, theta: float,
                               m_N: float, n_terms: int = 10) -> float:
    """
    Radial component of perturbation force on Uranus.
    
    F_r = -∂R/∂r_U
    
    Parameters
    ----------
    r_U : float
        Distance of Uranus from Sun (AU)
    r_N : float
        Distance of Neptune from Sun (AU)
    theta : float
        Angle between position vectors (rad)
    m_N : float
        Mass of Neptune (kg)
    n_terms : int
        Number of terms in expansion
        
    Returns
    -------
    float
        Radial force component
    """
    # Numerical derivative
    dr = 1e-6 * r_U
    R_plus = disturbing_function_total(r_U + dr, r_N, np.cos(theta), m_N, n_terms)
    R_minus = disturbing_function_total(r_U - dr, r_N, np.cos(theta), m_N, n_terms)
    return -(R_plus - R_minus) / (2 * dr)


def perturbation_force_tangential(r_U: float, r_N: float, theta: float,
                                   m_N: float, n_terms: int = 10) -> float:
    """
    Tangential component of perturbation force on Uranus.
    
    F_theta = -(1/r_U) * ∂R/∂θ
    
    Parameters
    ----------
    r_U : float
        Distance of Uranus from Sun (AU)
    r_N : float
        Distance of Neptune from Sun (AU)
    theta : float
        Angle between position vectors (rad)
    m_N : float
        Mass of Neptune (kg)
    n_terms : int
        Number of terms in expansion
        
    Returns
    -------
    float
        Tangential force component
    """
    # Numerical derivative
    dtheta = 1e-6
    R_plus = disturbing_function_total(r_U, r_N, np.cos(theta + dtheta), m_N, n_terms)
    R_minus = disturbing_function_total(r_U, r_N, np.cos(theta - dtheta), m_N, n_terms)
    return -(R_plus - R_minus) / (2 * dtheta * r_U)


if __name__ == "__main__":
    from .constants import URANUS, NEPTUNE
    
    print("Perturbation Theory Tests")
    print("=" * 50)
    
    # Test Legendre expansion convergence
    x = URANUS.semi_major_axis / NEPTUNE.semi_major_axis
    print(f"\nRatio a_U/a_N = {x:.4f}")
    
    partial_sums, errors, exact = convergence_analysis(x, n_terms=10)
    print(f"Exact value at conjunction: {exact:.4f}")
    print("\nConvergence:")
    for n in range(6):
        print(f"  n={n}: partial sum = {partial_sums[n]:.4f}, error = {errors[n]*100:.1f}%")
    
    # Test first-order deviation estimate
    t = 50  # years
    deviation_rad = longitude_deviation_first_order(
        t, NEPTUNE.mass, 
        URANUS.semi_major_axis, NEPTUNE.semi_major_axis,
        URANUS.mean_motion, NEPTUNE.mean_motion
    )
    deviation_arcmin = deviation_rad * RAD_TO_ARCMIN
    print(f"\nFirst-order deviation over {t} years: {deviation_arcmin:.2f} arcmin")
    print("(Observed: ~20-30 arcmin - factor of 10 larger due to higher-order effects)")
