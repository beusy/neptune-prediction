"""
Fourier Decomposition Module
============================

Implements Fourier analysis of longitude deviations:
- Design matrix construction
- Least-squares coefficient estimation
- Amplitude and phase extraction
- Harmonic analysis

References
----------
- Press et al. (2007): Numerical Recipes, Ch. 13
- Aster et al. (2013): Parameter Estimation and Inverse Problems
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .constants import RAD_TO_DEG, DEG_TO_RAD


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FourierCoefficients:
    """
    Container for Fourier decomposition results.
    
    Attributes
    ----------
    C0 : float
        DC (constant) component
    C : np.ndarray
        Cosine coefficients [C_1, C_2, ..., C_K]
    S : np.ndarray
        Sine coefficients [S_1, S_2, ..., S_K]
    amplitudes : np.ndarray
        Harmonic amplitudes sqrt(C_k² + S_k²)
    phases : np.ndarray
        Harmonic phases atan2(S_k, C_k)
    residual_rms : float
        RMS of fit residuals
    covariance : np.ndarray
        Covariance matrix of coefficients
    """
    C0: float
    C: np.ndarray
    S: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    residual_rms: float
    covariance: Optional[np.ndarray] = None
    
    @property
    def n_harmonics(self) -> int:
        """Number of harmonic modes."""
        return len(self.C)
    
    @property
    def all_coefficients(self) -> np.ndarray:
        """All coefficients as single array [C0, C1, S1, C2, S2, ...]."""
        result = [self.C0]
        for c, s in zip(self.C, self.S):
            result.extend([c, s])
        return np.array(result)
    
    def __repr__(self) -> str:
        return (f"FourierCoefficients(C0={self.C0:.4f}, "
                f"K={self.n_harmonics}, RMS={self.residual_rms:.4f})")


# =============================================================================
# DESIGN MATRIX CONSTRUCTION
# =============================================================================

def build_design_matrix(times: np.ndarray, n_U: float, n_N: float,
                        K: int = 2, lambda_U0: float = 0.0,
                        lambda_N0: float = 0.0) -> np.ndarray:
    """
    Build the design matrix for Fourier decomposition.
    
    The model is:
    Δλ(t) = C_0 + Σ_{k=1}^{K} [C_k cos(kφ) + S_k sin(kφ)]
    
    where φ(t) = (n_U - n_N)*t + (λ_U0 - λ_N0)
    
    Parameters
    ----------
    times : np.ndarray
        Observation times (years)
    n_U : float
        Mean motion of Uranus (rad/year)
    n_N : float
        Mean motion of Neptune (rad/year)
    K : int
        Number of harmonic modes
    lambda_U0 : float
        Initial longitude of Uranus (rad)
    lambda_N0 : float
        Initial longitude of Neptune (rad)
        
    Returns
    -------
    np.ndarray
        Design matrix A with shape (N, 2K+1)
        Columns: [1, cos(φ), sin(φ), cos(2φ), sin(2φ), ...]
    """
    N = len(times)
    n_params = 2 * K + 1
    
    A = np.zeros((N, n_params))
    
    # Synodic phase for each observation
    phi = (n_U - n_N) * times + (lambda_U0 - lambda_N0)
    
    # DC component
    A[:, 0] = 1.0
    
    # Harmonic components
    for k in range(1, K + 1):
        A[:, 2*k - 1] = np.cos(k * phi)  # C_k column
        A[:, 2*k] = np.sin(k * phi)      # S_k column
    
    return A


def build_design_matrix_general(times: np.ndarray, 
                                 frequencies: np.ndarray) -> np.ndarray:
    """
    Build design matrix for arbitrary frequencies.
    
    Parameters
    ----------
    times : np.ndarray
        Observation times
    frequencies : np.ndarray
        Frequencies to include (rad/time unit)
        
    Returns
    -------
    np.ndarray
        Design matrix with columns [1, cos(ω₁t), sin(ω₁t), ...]
    """
    N = len(times)
    n_freq = len(frequencies)
    n_params = 2 * n_freq + 1
    
    A = np.zeros((N, n_params))
    A[:, 0] = 1.0  # DC component
    
    for i, omega in enumerate(frequencies):
        A[:, 2*i + 1] = np.cos(omega * times)
        A[:, 2*i + 2] = np.sin(omega * times)
    
    return A


# =============================================================================
# LEAST SQUARES FITTING
# =============================================================================

def fit_fourier_coefficients(times: np.ndarray, deviations: np.ndarray,
                              n_U: float, n_N: float, K: int = 2,
                              weights: Optional[np.ndarray] = None,
                              lambda_U0: float = 0.0,
                              lambda_N0: float = 0.0) -> FourierCoefficients:
    """
    Fit Fourier coefficients to longitude deviation data.
    
    Solves the weighted least-squares problem:
    min ||W^(1/2) (A*c - y)||²
    
    Parameters
    ----------
    times : np.ndarray
        Observation times (years)
    deviations : np.ndarray
        Observed longitude deviations (arcmin)
    n_U : float
        Mean motion of Uranus (rad/year)
    n_N : float
        Mean motion of Neptune (rad/year)
    K : int
        Number of harmonic modes
    weights : np.ndarray, optional
        Observation weights (1/σ²). If None, uniform weights.
    lambda_U0 : float
        Initial longitude of Uranus (rad)
    lambda_N0 : float
        Initial longitude of Neptune (rad)
        
    Returns
    -------
    FourierCoefficients
        Fitted coefficients with uncertainties
    """
    # Build design matrix
    A = build_design_matrix(times, n_U, n_N, K, lambda_U0, lambda_N0)
    y = deviations
    
    # Apply weights if provided
    if weights is not None:
        W_sqrt = np.sqrt(np.diag(weights))
        A = W_sqrt @ A
        y = W_sqrt @ y
    
    # Solve normal equations: (A^T A) c = A^T y
    ATA = A.T @ A
    ATy = A.T @ y
    
    # Use numpy's least squares for numerical stability
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    
    # Extract coefficients
    C0 = coeffs[0]
    C = coeffs[1::2]  # Odd indices (1, 3, 5, ...)
    S = coeffs[2::2]  # Even indices (2, 4, 6, ...)
    
    # Compute amplitudes and phases
    amplitudes = np.sqrt(C**2 + S**2)
    phases = np.arctan2(S, C)
    
    # Compute residuals and RMS
    A_original = build_design_matrix(times, n_U, n_N, K, lambda_U0, lambda_N0)
    fitted = A_original @ coeffs
    residual_values = deviations - fitted
    residual_rms = np.sqrt(np.mean(residual_values**2))
    
    # Compute covariance matrix
    try:
        # σ² estimate from residuals
        n_obs = len(deviations)
        n_params = len(coeffs)
        sigma_sq = np.sum(residual_values**2) / (n_obs - n_params)
        
        # Covariance = σ² * (A^T A)^(-1)
        ATA_inv = np.linalg.inv(ATA)
        covariance = sigma_sq * ATA_inv
    except np.linalg.LinAlgError:
        covariance = None
    
    return FourierCoefficients(
        C0=C0,
        C=C,
        S=S,
        amplitudes=amplitudes,
        phases=phases,
        residual_rms=residual_rms,
        covariance=covariance
    )


def evaluate_fourier_model(times: np.ndarray, coeffs: FourierCoefficients,
                            n_U: float, n_N: float,
                            lambda_U0: float = 0.0,
                            lambda_N0: float = 0.0) -> np.ndarray:
    """
    Evaluate Fourier model at given times.
    
    Parameters
    ----------
    times : np.ndarray
        Times to evaluate (years)
    coeffs : FourierCoefficients
        Fourier coefficients
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
    np.ndarray
        Model predictions
    """
    phi = (n_U - n_N) * times + (lambda_U0 - lambda_N0)
    
    result = coeffs.C0 * np.ones_like(times)
    
    for k, (c, s) in enumerate(zip(coeffs.C, coeffs.S), start=1):
        result += c * np.cos(k * phi) + s * np.sin(k * phi)
    
    return result


# =============================================================================
# HARMONIC ANALYSIS
# =============================================================================

def harmonic_amplitude(C: float, S: float) -> float:
    """
    Compute harmonic amplitude from sine and cosine coefficients.
    
    A = sqrt(C² + S²)
    """
    return np.sqrt(C**2 + S**2)


def harmonic_phase(C: float, S: float) -> float:
    """
    Compute harmonic phase from sine and cosine coefficients.
    
    φ = atan2(S, C)
    
    Returns phase in radians.
    """
    return np.arctan2(S, C)


def amplitude_uncertainty(C: float, S: float, 
                          sigma_C: float, sigma_S: float,
                          cov_CS: float = 0.0) -> float:
    """
    Compute uncertainty in amplitude via error propagation.
    
    σ_A² = (∂A/∂C)² σ_C² + (∂A/∂S)² σ_S² + 2(∂A/∂C)(∂A/∂S) cov(C,S)
    
    Parameters
    ----------
    C : float
        Cosine coefficient
    S : float
        Sine coefficient
    sigma_C : float
        Uncertainty in C
    sigma_S : float
        Uncertainty in S
    cov_CS : float
        Covariance between C and S
        
    Returns
    -------
    float
        Uncertainty in amplitude
    """
    A = np.sqrt(C**2 + S**2)
    if A == 0:
        return np.sqrt(sigma_C**2 + sigma_S**2)
    
    dA_dC = C / A
    dA_dS = S / A
    
    variance = (dA_dC**2 * sigma_C**2 + 
                dA_dS**2 * sigma_S**2 + 
                2 * dA_dC * dA_dS * cov_CS)
    
    return np.sqrt(max(0, variance))


def phase_uncertainty(C: float, S: float,
                      sigma_C: float, sigma_S: float,
                      cov_CS: float = 0.0) -> float:
    """
    Compute uncertainty in phase via error propagation.
    
    Parameters
    ----------
    C : float
        Cosine coefficient
    S : float
        Sine coefficient
    sigma_C : float
        Uncertainty in C
    sigma_S : float
        Uncertainty in S
    cov_CS : float
        Covariance between C and S
        
    Returns
    -------
    float
        Uncertainty in phase (radians)
    """
    A_sq = C**2 + S**2
    if A_sq == 0:
        return np.pi  # Maximum uncertainty
    
    dPhi_dC = -S / A_sq
    dPhi_dS = C / A_sq
    
    variance = (dPhi_dC**2 * sigma_C**2 + 
                dPhi_dS**2 * sigma_S**2 + 
                2 * dPhi_dC * dPhi_dS * cov_CS)
    
    return np.sqrt(max(0, variance))


# =============================================================================
# CONDITION NUMBER AND NUMERICAL STABILITY
# =============================================================================

def condition_number(A: np.ndarray) -> float:
    """
    Compute condition number of design matrix.
    
    A well-conditioned problem has condition number close to 1.
    Condition number > 100 indicates potential numerical issues.
    
    Parameters
    ----------
    A : np.ndarray
        Design matrix
        
    Returns
    -------
    float
        Condition number (ratio of largest to smallest singular value)
    """
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] == 0:
        return np.inf
    return s[0] / s[-1]


def assess_phase_coverage(times: np.ndarray, n_U: float, n_N: float,
                          lambda_U0: float = 0.0, 
                          lambda_N0: float = 0.0) -> dict:
    """
    Assess quality of phase coverage in observations.
    
    Parameters
    ----------
    times : np.ndarray
        Observation times (years)
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
    dict
        Assessment including:
        - phase_range: total phase range covered (radians)
        - cycles_covered: number of synodic cycles
        - phase_uniformity: measure of uniform distribution (0-1)
    """
    phi = (n_U - n_N) * times + (lambda_U0 - lambda_N0)
    phi_wrapped = phi % (2 * np.pi)
    
    # Phase range
    phase_range = np.max(phi) - np.min(phi)
    
    # Synodic period
    synodic_period = 2 * np.pi / abs(n_U - n_N)
    cycles_covered = phase_range / (2 * np.pi)
    
    # Phase uniformity (via histogram)
    n_bins = 12
    hist, _ = np.histogram(phi_wrapped, bins=n_bins, range=(0, 2*np.pi))
    expected = len(times) / n_bins
    uniformity = 1.0 - np.std(hist) / expected if expected > 0 else 0.0
    
    return {
        'phase_range': phase_range,
        'cycles_covered': cycles_covered,
        'phase_uniformity': max(0, min(1, uniformity)),
        'synodic_period': synodic_period,
        'time_baseline': np.max(times) - np.min(times)
    }


if __name__ == "__main__":
    from .constants import URANUS, NEPTUNE
    
    print("Fourier Decomposition Tests")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_obs = 60
    times = np.linspace(0, 50, n_obs)  # 50 years
    
    # True coefficients
    true_C0 = 1.0
    true_C1, true_S1 = 1.5, 0.8
    true_C2, true_S2 = 0.9, 0.4
    
    # Generate observations
    n_U = URANUS.mean_motion
    n_N = NEPTUNE.mean_motion
    phi = (n_U - n_N) * times
    
    true_signal = (true_C0 + 
                   true_C1 * np.cos(phi) + true_S1 * np.sin(phi) +
                   true_C2 * np.cos(2*phi) + true_S2 * np.sin(2*phi))
    
    noise = np.random.normal(0, 0.5, n_obs)
    observations = true_signal + noise
    
    # Fit Fourier coefficients
    result = fit_fourier_coefficients(times, observations, n_U, n_N, K=2)
    
    print(f"\nTrue coefficients: C0={true_C0}, C1={true_C1}, S1={true_S1}")
    print(f"Fitted coefficients: C0={result.C0:.3f}, C1={result.C[0]:.3f}, S1={result.S[0]:.3f}")
    print(f"\nFirst harmonic amplitude: {result.amplitudes[0]:.3f}")
    print(f"True amplitude: {np.sqrt(true_C1**2 + true_S1**2):.3f}")
    print(f"\nRMS residual: {result.residual_rms:.4f}")
    
    # Phase coverage
    coverage = assess_phase_coverage(times, n_U, n_N)
    print(f"\nPhase coverage: {coverage['cycles_covered']:.2f} cycles")
    print(f"Uniformity score: {coverage['phase_uniformity']:.2f}")
