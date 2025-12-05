#!/usr/bin/env python3
"""
N-Body Numerical Integration
============================

Three-body numerical integration for validation of perturbation theory.

This module provides:
- Numerical integration of Sun-Uranus-Neptune system
- Comparison with analytical perturbation theory
- Validation of forward model accuracy

Usage:
    python -m simulations.nbody
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverrier import (
    G, M_SUN, AU, URANUS, NEPTUNE,
    period_from_semi_major_axis,
    circular_position
)


# =============================================================================
# CONSTANTS (in AU, years, solar masses)
# =============================================================================

# Gravitational parameter in AU^3 / year^2
GM_SUN = 4 * np.pi**2  # This gives orbital period in years when a is in AU


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NBodyState:
    """State of N-body system."""
    positions: np.ndarray  # Shape (N, 3) in AU
    velocities: np.ndarray  # Shape (N, 3) in AU/year
    masses: np.ndarray  # Shape (N,) in solar masses
    time: float  # years
    
    @property
    def n_bodies(self) -> int:
        return len(self.masses)
    
    def kinetic_energy(self) -> float:
        """Total kinetic energy."""
        KE = 0.0
        for i in range(self.n_bodies):
            v_sq = np.sum(self.velocities[i]**2)
            KE += 0.5 * self.masses[i] * v_sq
        return KE
    
    def potential_energy(self) -> float:
        """Total gravitational potential energy."""
        PE = 0.0
        for i in range(self.n_bodies):
            for j in range(i+1, self.n_bodies):
                r = np.linalg.norm(self.positions[i] - self.positions[j])
                PE -= GM_SUN * self.masses[i] * self.masses[j] / r
        return PE
    
    def total_energy(self) -> float:
        """Total mechanical energy."""
        return self.kinetic_energy() + self.potential_energy()


@dataclass
class IntegrationResult:
    """Results from N-body integration."""
    times: np.ndarray
    uranus_positions: np.ndarray  # Shape (N_times, 3)
    neptune_positions: np.ndarray  # Shape (N_times, 3)
    uranus_longitudes: np.ndarray  # Shape (N_times,)
    neptune_longitudes: np.ndarray  # Shape (N_times,)
    energy_error: np.ndarray  # Relative energy error over time


# =============================================================================
# GRAVITATIONAL ACCELERATION
# =============================================================================

def gravitational_acceleration(positions: np.ndarray, 
                                masses: np.ndarray) -> np.ndarray:
    """
    Compute gravitational acceleration on each body.
    
    Parameters
    ----------
    positions : np.ndarray
        Positions of all bodies, shape (N, 3)
    masses : np.ndarray
        Masses of all bodies, shape (N,) in solar masses
        
    Returns
    -------
    np.ndarray
        Accelerations, shape (N, 3) in AU/year^2
    """
    n_bodies = len(masses)
    accelerations = np.zeros_like(positions)
    
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                accelerations[i] += GM_SUN * masses[j] * r_vec / r**3
    
    return accelerations


# =============================================================================
# INTEGRATORS
# =============================================================================

def rk4_step(state: NBodyState, dt: float) -> NBodyState:
    """
    Fourth-order Runge-Kutta integration step.
    
    Parameters
    ----------
    state : NBodyState
        Current state
    dt : float
        Time step (years)
        
    Returns
    -------
    NBodyState
        New state after dt
    """
    pos = state.positions
    vel = state.velocities
    masses = state.masses
    
    # k1
    a1 = gravitational_acceleration(pos, masses)
    
    # k2
    pos2 = pos + 0.5 * dt * vel
    vel2 = vel + 0.5 * dt * a1
    a2 = gravitational_acceleration(pos2, masses)
    
    # k3
    pos3 = pos + 0.5 * dt * vel2
    vel3 = vel + 0.5 * dt * a2
    a3 = gravitational_acceleration(pos3, masses)
    
    # k4
    pos4 = pos + dt * vel3
    vel4 = vel + dt * a3
    a4 = gravitational_acceleration(pos4, masses)
    
    # Combine
    new_pos = pos + (dt/6) * (vel + 2*vel2 + 2*vel3 + vel4)
    new_vel = vel + (dt/6) * (a1 + 2*a2 + 2*a3 + a4)
    
    return NBodyState(
        positions=new_pos,
        velocities=new_vel,
        masses=masses,
        time=state.time + dt
    )


def leapfrog_step(state: NBodyState, dt: float) -> NBodyState:
    """
    Leapfrog (Verlet) integration step - symplectic integrator.
    
    Parameters
    ----------
    state : NBodyState
        Current state
    dt : float
        Time step (years)
        
    Returns
    -------
    NBodyState
        New state after dt
    """
    # Half step velocity
    a = gravitational_acceleration(state.positions, state.masses)
    vel_half = state.velocities + 0.5 * dt * a
    
    # Full step position
    new_pos = state.positions + dt * vel_half
    
    # Full step velocity
    a_new = gravitational_acceleration(new_pos, state.masses)
    new_vel = vel_half + 0.5 * dt * a_new
    
    return NBodyState(
        positions=new_pos,
        velocities=new_vel,
        masses=state.masses,
        time=state.time + dt
    )


# =============================================================================
# INITIAL CONDITIONS
# =============================================================================

def create_initial_state(uranus_longitude: float = 0.0,
                          neptune_longitude: float = 0.0) -> NBodyState:
    """
    Create initial state for Sun-Uranus-Neptune system.
    
    Assumes circular orbits for simplicity.
    
    Parameters
    ----------
    uranus_longitude : float
        Initial longitude of Uranus (radians)
    neptune_longitude : float
        Initial longitude of Neptune (radians)
        
    Returns
    -------
    NBodyState
        Initial state
    """
    # Masses in solar masses
    m_sun = 1.0
    m_uranus = URANUS.mass / M_SUN
    m_neptune = NEPTUNE.mass / M_SUN
    masses = np.array([m_sun, m_uranus, m_neptune])
    
    # Semi-major axes
    a_uranus = URANUS.semi_major_axis
    a_neptune = NEPTUNE.semi_major_axis
    
    # Circular velocities (AU/year)
    v_uranus = 2 * np.pi * a_uranus / period_from_semi_major_axis(a_uranus)
    v_neptune = 2 * np.pi * a_neptune / period_from_semi_major_axis(a_neptune)
    
    # Positions (Sun at origin)
    positions = np.array([
        [0.0, 0.0, 0.0],  # Sun
        [a_uranus * np.cos(uranus_longitude), 
         a_uranus * np.sin(uranus_longitude), 0.0],  # Uranus
        [a_neptune * np.cos(neptune_longitude), 
         a_neptune * np.sin(neptune_longitude), 0.0]   # Neptune
    ])
    
    # Velocities (perpendicular to position for circular orbit)
    velocities = np.array([
        [0.0, 0.0, 0.0],  # Sun (stationary in this frame)
        [-v_uranus * np.sin(uranus_longitude), 
          v_uranus * np.cos(uranus_longitude), 0.0],  # Uranus
        [-v_neptune * np.sin(neptune_longitude), 
          v_neptune * np.cos(neptune_longitude), 0.0]   # Neptune
    ])
    
    return NBodyState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        time=0.0
    )


# =============================================================================
# INTEGRATION
# =============================================================================

def integrate_system(initial_state: NBodyState,
                      t_final: float,
                      dt: float = 0.01,
                      method: str = 'leapfrog',
                      output_interval: int = 10) -> IntegrationResult:
    """
    Integrate the three-body system.
    
    Parameters
    ----------
    initial_state : NBodyState
        Initial state
    t_final : float
        Final integration time (years)
    dt : float
        Time step (years)
    method : str
        Integration method: 'rk4' or 'leapfrog'
    output_interval : int
        Save state every N steps
        
    Returns
    -------
    IntegrationResult
        Integration results
    """
    # Select integrator
    if method == 'rk4':
        step_func = rk4_step
    else:
        step_func = leapfrog_step
    
    # Initialize storage
    n_steps = int(t_final / dt)
    n_outputs = n_steps // output_interval + 1
    
    times = np.zeros(n_outputs)
    uranus_pos = np.zeros((n_outputs, 3))
    neptune_pos = np.zeros((n_outputs, 3))
    energy = np.zeros(n_outputs)
    
    # Initial values
    state = initial_state
    E0 = state.total_energy()
    
    times[0] = state.time
    uranus_pos[0] = state.positions[1]
    neptune_pos[0] = state.positions[2]
    energy[0] = 0.0
    
    output_idx = 1
    
    # Integration loop
    for step in range(1, n_steps + 1):
        state = step_func(state, dt)
        
        if step % output_interval == 0 and output_idx < n_outputs:
            times[output_idx] = state.time
            uranus_pos[output_idx] = state.positions[1]
            neptune_pos[output_idx] = state.positions[2]
            energy[output_idx] = (state.total_energy() - E0) / abs(E0)
            output_idx += 1
    
    # Compute longitudes
    uranus_long = np.arctan2(uranus_pos[:, 1], uranus_pos[:, 0])
    neptune_long = np.arctan2(neptune_pos[:, 1], neptune_pos[:, 0])
    
    return IntegrationResult(
        times=times[:output_idx],
        uranus_positions=uranus_pos[:output_idx],
        neptune_positions=neptune_pos[:output_idx],
        uranus_longitudes=uranus_long[:output_idx],
        neptune_longitudes=neptune_long[:output_idx],
        energy_error=energy[:output_idx]
    )


def compute_longitude_deviation(result: IntegrationResult) -> np.ndarray:
    """
    Compute Uranus longitude deviation from two-body prediction.
    
    Parameters
    ----------
    result : IntegrationResult
        N-body integration results
        
    Returns
    -------
    np.ndarray
        Longitude deviations (arcmin)
    """
    # Two-body prediction
    n_uranus = 2 * np.pi / period_from_semi_major_axis(URANUS.semi_major_axis)
    two_body_longitude = result.uranus_longitudes[0] + n_uranus * result.times
    
    # Actual longitude (unwrapped)
    actual_longitude = np.unwrap(result.uranus_longitudes)
    
    # Deviation in radians
    deviation_rad = actual_longitude - two_body_longitude
    
    # Convert to arcminutes
    deviation_arcmin = np.degrees(deviation_rad) * 60
    
    return deviation_arcmin


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run N-body integration and compare with perturbation theory."""
    print("=" * 60)
    print("N-BODY INTEGRATION: Sun-Uranus-Neptune System")
    print("=" * 60)
    
    # Create initial state
    print("\nCreating initial conditions...")
    initial = create_initial_state(
        uranus_longitude=0.0,
        neptune_longitude=np.pi/4  # 45 degrees ahead
    )
    
    print(f"  Uranus mass: {initial.masses[1]:.3e} M_sun")
    print(f"  Neptune mass: {initial.masses[2]:.3e} M_sun")
    print(f"  Initial energy: {initial.total_energy():.6f}")
    
    # Integrate
    print("\nIntegrating for 100 years...")
    result = integrate_system(
        initial,
        t_final=100.0,
        dt=0.001,  # ~0.4 days
        method='leapfrog',
        output_interval=100
    )
    
    print(f"  Integration complete")
    print(f"  Time points: {len(result.times)}")
    print(f"  Max energy error: {np.max(np.abs(result.energy_error)):.2e}")
    
    # Compute deviations
    print("\nComputing longitude deviations...")
    deviations = compute_longitude_deviation(result)
    
    print(f"  Max deviation: {np.max(np.abs(deviations)):.2f} arcmin")
    print(f"  Final deviation: {deviations[-1]:.2f} arcmin")
    
    # Compare with expected
    # First-order estimate: Δλ ~ (m_N/M_sun) * (a_U/a_N)^2 * n_U * t
    mu = NEPTUNE.mass / M_SUN
    alpha = URANUS.semi_major_axis / NEPTUNE.semi_major_axis
    n_U = 2 * np.pi / period_from_semi_major_axis(URANUS.semi_major_axis)
    expected_rate = mu * alpha**2 * n_U * 3437.75  # arcmin/year
    
    print(f"\nComparison with first-order theory:")
    print(f"  Expected rate: {expected_rate:.4f} arcmin/year")
    print(f"  Actual rate: {deviations[-1]/100:.4f} arcmin/year")
    
    print("\n" + "=" * 60)
    print("Integration complete!")
    print("=" * 60)
    
    return result, deviations


if __name__ == "__main__":
    result, deviations = main()
