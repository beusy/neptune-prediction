"""
Orbital Mechanics Module
========================

Core orbital mechanics calculations including:
- Kepler's laws
- Mean and true anomaly conversions
- Orbital element transformations
- Position and velocity calculations

References
----------
- Murray & Dermott (1999): Solar System Dynamics
- Danby (1992): Fundamentals of Celestial Mechanics
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .constants import (
    GM_SUN_AU, DEG_TO_RAD, RAD_TO_DEG,
    period_from_semi_major_axis, mean_motion_from_semi_major_axis
)


# =============================================================================
# ORBITAL ELEMENTS
# =============================================================================

@dataclass
class OrbitalElements:
    """
    Classical Keplerian orbital elements.
    
    Attributes
    ----------
    a : float
        Semi-major axis (AU)
    e : float
        Eccentricity (dimensionless, 0 ≤ e < 1 for ellipse)
    i : float
        Inclination (radians)
    Omega : float
        Longitude of ascending node (radians)
    omega : float
        Argument of perihelion (radians)
    M0 : float
        Mean anomaly at epoch (radians)
    epoch : float
        Reference epoch (years from J2000 or arbitrary reference)
    """
    a: float
    e: float
    i: float = 0.0
    Omega: float = 0.0
    omega: float = 0.0
    M0: float = 0.0
    epoch: float = 0.0
    
    @property
    def period(self) -> float:
        """Orbital period in years."""
        return period_from_semi_major_axis(self.a)
    
    @property
    def mean_motion(self) -> float:
        """Mean motion in radians per year."""
        return mean_motion_from_semi_major_axis(self.a)
    
    def mean_anomaly(self, t: float) -> float:
        """
        Mean anomaly at time t.
        
        Parameters
        ----------
        t : float
            Time in years from epoch
            
        Returns
        -------
        float
            Mean anomaly in radians (wrapped to [0, 2π))
        """
        M = self.M0 + self.mean_motion * (t - self.epoch)
        return M % (2 * np.pi)
    
    def true_anomaly(self, t: float, tol: float = 1e-10) -> float:
        """
        True anomaly at time t.
        
        Parameters
        ----------
        t : float
            Time in years from epoch
        tol : float
            Tolerance for Kepler equation solver
            
        Returns
        -------
        float
            True anomaly in radians
        """
        M = self.mean_anomaly(t)
        E = solve_kepler_equation(M, self.e, tol)
        return eccentric_to_true_anomaly(E, self.e)
    
    def longitude(self, t: float) -> float:
        """
        True longitude at time t (for coplanar orbits).
        
        Parameters
        ----------
        t : float
            Time in years from epoch
            
        Returns
        -------
        float
            True longitude in radians
        """
        return (self.omega + self.true_anomaly(t)) % (2 * np.pi)
    
    def mean_longitude(self, t: float) -> float:
        """
        Mean longitude at time t.
        
        Parameters
        ----------
        t : float
            Time in years from epoch
            
        Returns
        -------
        float
            Mean longitude in radians
        """
        return (self.omega + self.mean_anomaly(t)) % (2 * np.pi)


# =============================================================================
# KEPLER'S EQUATION
# =============================================================================

def solve_kepler_equation(M: float, e: float, tol: float = 1e-10, 
                          max_iter: int = 50) -> float:
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.
    
    Uses Newton-Raphson iteration with guaranteed convergence for e < 1.
    
    Parameters
    ----------
    M : float
        Mean anomaly in radians
    e : float
        Orbital eccentricity (0 ≤ e < 1)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    float
        Eccentric anomaly in radians
        
    Raises
    ------
    ValueError
        If eccentricity is invalid or iteration fails to converge
    """
    if e < 0 or e >= 1:
        raise ValueError(f"Eccentricity must be in [0, 1), got {e}")
    
    # Initial guess (Danby's starting value)
    if e < 0.8:
        E = M
    else:
        E = np.pi
    
    # Newton-Raphson iteration
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime
        E = E - delta
        
        if abs(delta) < tol:
            return E
    
    raise ValueError(f"Kepler equation failed to converge after {max_iter} iterations")


def solve_kepler_equation_vectorized(M: np.ndarray, e: float, 
                                      tol: float = 1e-10) -> np.ndarray:
    """
    Vectorized Kepler equation solver for array of mean anomalies.
    
    Parameters
    ----------
    M : np.ndarray
        Array of mean anomalies in radians
    e : float
        Orbital eccentricity
    tol : float
        Convergence tolerance
        
    Returns
    -------
    np.ndarray
        Array of eccentric anomalies in radians
    """
    M = np.asarray(M)
    E = M.copy()
    
    for _ in range(50):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime
        E = E - delta
        
        if np.max(np.abs(delta)) < tol:
            break
    
    return E


# =============================================================================
# ANOMALY CONVERSIONS
# =============================================================================

def eccentric_to_true_anomaly(E: float, e: float) -> float:
    """
    Convert eccentric anomaly to true anomaly.
    
    Parameters
    ----------
    E : float
        Eccentric anomaly in radians
    e : float
        Orbital eccentricity
        
    Returns
    -------
    float
        True anomaly in radians
    """
    # tan(theta/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    beta = e / (1 + np.sqrt(1 - e**2))
    theta = E + 2 * np.arctan2(beta * np.sin(E), 1 - beta * np.cos(E))
    return theta


def true_to_eccentric_anomaly(theta: float, e: float) -> float:
    """
    Convert true anomaly to eccentric anomaly.
    
    Parameters
    ----------
    theta : float
        True anomaly in radians
    e : float
        Orbital eccentricity
        
    Returns
    -------
    float
        Eccentric anomaly in radians
    """
    # tan(E/2) = sqrt((1-e)/(1+e)) * tan(theta/2)
    E = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(theta / 2),
        np.sqrt(1 + e) * np.cos(theta / 2)
    )
    return E


# =============================================================================
# POSITION AND VELOCITY
# =============================================================================

def orbital_radius(a: float, e: float, theta: float) -> float:
    """
    Calculate orbital radius from true anomaly.
    
    r = a(1 - e²) / (1 + e*cos(θ))
    
    Parameters
    ----------
    a : float
        Semi-major axis
    e : float
        Eccentricity
    theta : float
        True anomaly in radians
        
    Returns
    -------
    float
        Orbital radius in same units as a
    """
    return a * (1 - e**2) / (1 + e * np.cos(theta))


def position_in_orbital_plane(a: float, e: float, theta: float) -> Tuple[float, float]:
    """
    Calculate position in the orbital plane (2D).
    
    Parameters
    ----------
    a : float
        Semi-major axis (AU)
    e : float
        Eccentricity
    theta : float
        True anomaly in radians
        
    Returns
    -------
    tuple
        (x, y) position in AU, with x toward perihelion
    """
    r = orbital_radius(a, e, theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def position_3d(elements: OrbitalElements, t: float) -> Tuple[float, float, float]:
    """
    Calculate 3D heliocentric position.
    
    Parameters
    ----------
    elements : OrbitalElements
        Orbital elements
    t : float
        Time in years from epoch
        
    Returns
    -------
    tuple
        (x, y, z) heliocentric position in AU
    """
    theta = elements.true_anomaly(t)
    r = orbital_radius(elements.a, elements.e, theta)
    
    # Position in orbital plane
    x_orb = r * np.cos(theta)
    y_orb = r * np.sin(theta)
    
    # Rotation matrices
    cos_O = np.cos(elements.Omega)
    sin_O = np.sin(elements.Omega)
    cos_i = np.cos(elements.i)
    sin_i = np.sin(elements.i)
    cos_w = np.cos(elements.omega)
    sin_w = np.sin(elements.omega)
    
    # Combined rotation (orbital plane to ecliptic)
    x = (cos_O * cos_w - sin_O * sin_w * cos_i) * x_orb + \
        (-cos_O * sin_w - sin_O * cos_w * cos_i) * y_orb
    y = (sin_O * cos_w + cos_O * sin_w * cos_i) * x_orb + \
        (-sin_O * sin_w + cos_O * cos_w * cos_i) * y_orb
    z = sin_w * sin_i * x_orb + cos_w * sin_i * y_orb
    
    return x, y, z


def velocity_in_orbital_plane(a: float, e: float, theta: float) -> Tuple[float, float]:
    """
    Calculate velocity in the orbital plane.
    
    Parameters
    ----------
    a : float
        Semi-major axis (AU)
    e : float
        Eccentricity
    theta : float
        True anomaly in radians
        
    Returns
    -------
    tuple
        (vx, vy) velocity in AU/year
    """
    # Vis-viva components
    n = mean_motion_from_semi_major_axis(a)
    p = a * (1 - e**2)  # semi-latus rectum
    h = n * a**2 * np.sqrt(1 - e**2)  # specific angular momentum
    
    vx = -h / p * np.sin(theta)
    vy = h / p * (e + np.cos(theta))
    
    return vx, vy


# =============================================================================
# SYNODIC CALCULATIONS
# =============================================================================

def synodic_period(n1: float, n2: float) -> float:
    """
    Calculate synodic period between two bodies.
    
    Parameters
    ----------
    n1 : float
        Mean motion of inner body (rad/year)
    n2 : float
        Mean motion of outer body (rad/year)
        
    Returns
    -------
    float
        Synodic period in years
    """
    if n1 == n2:
        return np.inf
    return 2 * np.pi / abs(n1 - n2)


def synodic_phase(t: float, n1: float, n2: float, 
                  lambda1_0: float = 0.0, lambda2_0: float = 0.0) -> float:
    """
    Calculate synodic phase angle.
    
    φ(t) = (n1 - n2)*t + (λ1_0 - λ2_0)
    
    Parameters
    ----------
    t : float
        Time in years
    n1 : float
        Mean motion of body 1 (rad/year)
    n2 : float
        Mean motion of body 2 (rad/year)
    lambda1_0 : float
        Initial longitude of body 1 (rad)
    lambda2_0 : float
        Initial longitude of body 2 (rad)
        
    Returns
    -------
    float
        Synodic phase angle in radians
    """
    return (n1 - n2) * t + (lambda1_0 - lambda2_0)


# =============================================================================
# CIRCULAR ORBIT APPROXIMATION
# =============================================================================

def circular_longitude(t: float, n: float, lambda0: float = 0.0) -> float:
    """
    Longitude for circular orbit.
    
    Parameters
    ----------
    t : float
        Time in years
    n : float
        Mean motion (rad/year)
    lambda0 : float
        Initial longitude (rad)
        
    Returns
    -------
    float
        Longitude in radians (wrapped to [0, 2π))
    """
    return (lambda0 + n * t) % (2 * np.pi)


def circular_position(t: float, a: float, n: float, 
                      lambda0: float = 0.0) -> Tuple[float, float]:
    """
    Position for circular orbit in 2D.
    
    Parameters
    ----------
    t : float
        Time in years
    a : float
        Orbital radius (AU)
    n : float
        Mean motion (rad/year)
    lambda0 : float
        Initial longitude (rad)
        
    Returns
    -------
    tuple
        (x, y) position in AU
    """
    longitude = circular_longitude(t, n, lambda0)
    x = a * np.cos(longitude)
    y = a * np.sin(longitude)
    return x, y


if __name__ == "__main__":
    # Test with Uranus and Neptune
    from .constants import URANUS, NEPTUNE
    
    print("Orbital Mechanics Tests")
    print("=" * 50)
    
    # Create orbital elements
    uranus = OrbitalElements(
        a=URANUS.semi_major_axis,
        e=URANUS.eccentricity,
        M0=0.0
    )
    
    neptune = OrbitalElements(
        a=NEPTUNE.semi_major_axis,
        e=NEPTUNE.eccentricity,
        M0=0.0
    )
    
    print(f"Uranus period: {uranus.period:.2f} years")
    print(f"Neptune period: {neptune.period:.2f} years")
    print(f"Synodic period: {synodic_period(uranus.mean_motion, neptune.mean_motion):.2f} years")
