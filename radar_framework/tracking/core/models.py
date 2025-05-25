# radar_framework/tracking/core/models.py

import numpy as np
from dataclasses import asdict, dataclass

@dataclass
class RadarParams3D:
    Pt: float                 # Transmit power (W)
    G: float                  # Antenna gain (linear)
    wavelength: float         # Operating wavelength (m)
    sigma_max: float          # Target radar cross-section (m^2)
    k: float                  # Boltzmann's constant (J/K)
    T_noise: float            # System noise temperature (K)
    B: float                  # Receiver bandwidth (Hz)
    P_clutter: float          # Clutter power level
    P_int: float              # Interference power level
    sigma_az: float           # Pointing error std in azimuth (rad)
    sigma_el: float           # Pointing error std in elevation (rad)
    SNR_th: float             # Minimum SNR to initiate a new track

class RadarModel3D:
    """
    Radar propagation and SNR computation based on the radar range equation.
    Includes pointing loss model in az/el.
    """
    def __init__(self, params: RadarParams3D):
        # Ensure all parameters are valid
        for field, val in asdict(params).items():
            if val is None:
                raise ValueError(f"RadarParams3D.{field} cannot be None.")
        self._p = params

    def _orientation(self, az_t, el_t, az_b, el_b):
        """
        Gaussian antenna pointing loss based on deviation between target
        angles (az_t, el_t) and beam pointing (az_b, el_b).
        """
        da, de = az_t - az_b, el_t - el_b
        return np.exp(-0.5*((da/self._p.sigma_az)**2 + (de/self._p.sigma_el)**2))

    def instantaneous_snr(self, state, beam_angles, S_t=1.0):
        """
        Compute instantaneous SNR for a target in state [px, vx, py, vy, pz, vz].
        Applies the radar equation and noise/clutter/interference model.
        """
        px, vx, py, vy, pz, vz = state
        R = np.linalg.norm((px, py, pz))
        if R <= 0:
            raise ValueError("Invalid range; target too close or at origin.")
        # Angles of target
        az_t = np.arctan2(py, px)
        el_t = np.arcsin(pz/R)
        # Beam pointing angles
        az_b, el_b = beam_angles
        g = self._orientation(az_t, el_t, az_b, el_b)
        # Received power via radar equation
        Pr = (self._p.Pt * self._p.G**2 * self._p.wavelength**2 * self._p.sigma_max * g * S_t)
        Pr /= ((4*np.pi)**3 * R**4)
        # Total noise power
        Pn = self._p.k * self._p.T_noise * self._p.B + self._p.P_clutter + self._p.P_int
        return Pr / Pn

    def integrated_snr(self, state, beam_angles, T_int, S_t=1.0):
        """
        Compute coherent integration SNR over time T_int.
        Includes Doppler decorrelation loss for moving targets.
        """
        if T_int <= 0:
            raise ValueError("Integration time must be positive.")
        snr0 = self.instantaneous_snr(state, beam_angles, S_t)
        I = np.sqrt(T_int * self._p.B)  # Integration gain
        # Doppler velocity projection
        px, vx, py, vy, pz, vz = state
        R = np.linalg.norm((px, py, pz))
        vr = (px*vx + py*vy + pz*vz)/R
        D = np.exp(- (np.pi * 2 * vr * T_int / self._p.wavelength)**2)
        return snr0 * I * D