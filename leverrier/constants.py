"""
Physical and Astronomical Constants
====================================

Standardized constants for celestial mechanics calculations.
All values from CODATA 2018, IAU 2015, and JPL DE430 ephemerides.

References
----------
- CODATA 2018: https://physics.nist.gov/cuu/Constants/
- IAU 2015: IAU XXIX General Assembly Resolution B3
- JPL DE430: Folkner et al. (2014)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

# Gravitational constant (CODATA 2018)
G = 6.67430e-11  # m^3 kg^-1 s^-2
G_ERROR = 0.00015e-11  # uncertainty

# Speed of light (exact, SI definition)
C = 299792458  # m/s

# =============================================================================
# SOLAR SYSTEM CONSTANTS
# =============================================================================

# Solar mass (IAU 2015 nominal value)
M_SUN = 1.98892e30  # kg
M_SUN_ERROR = 0.00025e30  # uncertainty

# Astronomical Unit (IAU 2012 definition, exact)
AU = 1.495978707e11  # m

# Julian year (standard definition)
JULIAN_YEAR = 365.25  # days
JULIAN_YEAR_SECONDS = JULIAN_YEAR * 86400  # seconds

# Gaussian gravitational constant (historical, for reference)
K_GAUSS = 0.01720209895  # AU^(3/2) day^-1 M_sun^(-1/2)

# GM_sun in convenient units
GM_SUN_SI = G * M_SUN  # m^3 s^-2
GM_SUN_AU = 4 * np.pi**2  # AU^3 yr^-2 (from Kepler's third law)


# =============================================================================
# PLANETARY DATA
# =============================================================================

@dataclass(frozen=True)
class PlanetaryData:
    """Orbital and physical parameters for a planet."""
    name: str
    mass: float  # kg
    semi_major_axis: float  # AU
    eccentricity: float
    inclination: float  # degrees
    orbital_period: float  # years
    mean_motion: float  # rad/year
    
    @property
    def mass_ratio(self) -> float:
        """Mass ratio relative to Sun."""
        return self.mass / M_SUN


# Planetary data from JPL DE430 (Standish 1995, Folkner 2014)
PLANETS: Dict[str, PlanetaryData] = {
    'mercury': PlanetaryData(
        name='Mercury',
        mass=3.3011e23,
        semi_major_axis=0.38709927,
        eccentricity=0.20563593,
        inclination=7.00497902,
        orbital_period=0.2408467,
        mean_motion=26.0879
    ),
    'venus': PlanetaryData(
        name='Venus',
        mass=4.8675e24,
        semi_major_axis=0.72333566,
        eccentricity=0.00677672,
        inclination=3.39467605,
        orbital_period=0.61519726,
        mean_motion=10.2132
    ),
    'earth': PlanetaryData(
        name='Earth',
        mass=5.9722e24,
        semi_major_axis=1.00000261,
        eccentricity=0.01671123,
        inclination=0.00005,
        orbital_period=1.0000174,
        mean_motion=6.2832
    ),
    'mars': PlanetaryData(
        name='Mars',
        mass=6.4171e23,
        semi_major_axis=1.52371034,
        eccentricity=0.09339410,
        inclination=1.84969142,
        orbital_period=1.8808476,
        mean_motion=3.3407
    ),
    'jupiter': PlanetaryData(
        name='Jupiter',
        mass=1.8982e27,
        semi_major_axis=5.20288700,
        eccentricity=0.04838624,
        inclination=1.30439695,
        orbital_period=11.862615,
        mean_motion=0.5296
    ),
    'saturn': PlanetaryData(
        name='Saturn',
        mass=5.6834e26,
        semi_major_axis=9.53667594,
        eccentricity=0.05386179,
        inclination=2.48599187,
        orbital_period=29.447498,
        mean_motion=0.2133
    ),
    'uranus': PlanetaryData(
        name='Uranus',
        mass=8.6810e25,
        semi_major_axis=19.18916464,
        eccentricity=0.04725744,
        inclination=0.77263783,
        orbital_period=84.016846,
        mean_motion=0.0748
    ),
    'neptune': PlanetaryData(
        name='Neptune',
        mass=1.02413e26,
        semi_major_axis=30.06992276,
        eccentricity=0.00859048,
        inclination=1.77004347,
        orbital_period=164.79132,
        mean_motion=0.0381
    )
}

# Convenient aliases
URANUS = PLANETS['uranus']
NEPTUNE = PLANETS['neptune']


# =============================================================================
# CONVERSION FACTORS
# =============================================================================

# Angular conversions
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
ARCSEC_TO_RAD = DEG_TO_RAD / 3600.0
RAD_TO_ARCSEC = 3600.0 * RAD_TO_DEG
ARCMIN_TO_RAD = DEG_TO_RAD / 60.0
RAD_TO_ARCMIN = 60.0 * RAD_TO_DEG

# Time conversions
DAY_TO_YEAR = 1.0 / JULIAN_YEAR
YEAR_TO_DAY = JULIAN_YEAR
DAY_TO_SECOND = 86400.0
YEAR_TO_SECOND = JULIAN_YEAR_SECONDS


# =============================================================================
# HISTORICAL CONSTANTS (1846 values for comparison)
# =============================================================================

@dataclass(frozen=True)
class HistoricalConstants:
    """Constants as known in 1846 for historical comparison."""
    # Le Verrier's predicted values
    neptune_longitude_predicted: float = 325.0  # degrees
    neptune_distance_predicted: float = 36.15  # AU (his original estimate)
    neptune_mass_predicted: float = 35.0  # Earth masses
    
    # Actual discovery values
    neptune_longitude_actual: float = 326.2  # degrees at discovery
    neptune_discovery_date: str = "1846-09-23"
    
    # Bouvard observation parameters
    observation_start_year: int = 1781
    observation_end_year: int = 1845
    typical_precision_arcmin: float = 2.0


HISTORICAL = HistoricalConstants()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def mean_motion_from_period(period_years: float) -> float:
    """
    Calculate mean motion from orbital period.
    
    Parameters
    ----------
    period_years : float
        Orbital period in years
        
    Returns
    -------
    float
        Mean motion in radians per year
    """
    return 2 * np.pi / period_years


def period_from_semi_major_axis(a_au: float) -> float:
    """
    Calculate orbital period from semi-major axis using Kepler's third law.
    
    Parameters
    ----------
    a_au : float
        Semi-major axis in AU
        
    Returns
    -------
    float
        Orbital period in years
    """
    return np.sqrt(a_au**3)


def mean_motion_from_semi_major_axis(a_au: float) -> float:
    """
    Calculate mean motion from semi-major axis.
    
    Parameters
    ----------
    a_au : float
        Semi-major axis in AU
        
    Returns
    -------
    float
        Mean motion in radians per year
    """
    period = period_from_semi_major_axis(a_au)
    return mean_motion_from_period(period)


if __name__ == "__main__":
    # Verify constants
    print("Neptune Prediction - Physical Constants")
    print("=" * 50)
    print(f"G = {G:.5e} m^3 kg^-1 s^-2")
    print(f"M_sun = {M_SUN:.5e} kg")
    print(f"AU = {AU:.9e} m")
    print()
    print("Planetary Data:")
    for name, planet in PLANETS.items():
        print(f"  {planet.name}: a = {planet.semi_major_axis:.4f} AU, "
              f"n = {planet.mean_motion:.4f} rad/yr")
