# radar_framework/radar/AESA/core/models.py

import numpy as np
from dataclasses import dataclass

@dataclass
class AESAParams:
    """
    Parameters for 3D AESA radar processing.

    Attributes:
      - wavelength: carrier wavelength (m)
      - element_positions: array of shape (M, 3) with element coordinates
      - B: bandwidth of LFM pulse (Hz)
      - T_p: pulse duration (s)
      - P_fa: desired probability of false alarm for CFAR
      - az_limits: tuple of (min_az, max_az) radians for scan limits
      - el_limits: tuple of (min_el, max_el) radians for scan limits
    """
    wavelength: float
    element_positions: np.ndarray  # (M, 3)
    B: float
    T_p: float
    P_fa: float
    az_limits: tuple = (-np.pi/2, np.pi/2)
    el_limits: tuple = (-np.pi/4, np.pi/4)

class AESA:
    """
    Implements core AESA radar processing functions:
      - Field-of-view checking against azimuth/elevation limits
      - MVDR beamforming (DBF)
      - MIMO covariance synthesis
      - STAP weight computation
      - Pulse compression
      - CFAR thresholding and detection
      - Clutter covariance modeling
      - Robust LCMV beamforming
    """
    def __init__(self, params: AESAParams):
        self.params = params

    def in_fov(self, theta: float, el: float) -> bool:
        """
        Check if (theta, el) lies within the configured field-of-view.
        """
        az_min, az_max = self.params.az_limits
        el_min, el_max = self.params.el_limits
        return az_min <= theta <= az_max and el_min <= el <= el_max

    def steering_vector(self, theta: float, el: float) -> np.ndarray:
        """
        Compute steering vector for azimuth theta and elevation el.
        Raises ValueError if outside field-of-view.
        """
        if not self.in_fov(theta, el):
            raise ValueError(f"Scan angle (az={theta:.2f}, el={el:.2f}) out of FOV")
        pos = self.params.element_positions
        k = 2 * np.pi / self.params.wavelength
        d = np.array([
            np.cos(el) * np.cos(theta),
            np.cos(el) * np.sin(theta),
            np.sin(el)
        ])
        return np.exp(1j * k * (pos @ d))

    def mvdr_weights(self, R: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Compute MVDR (Capon) weights: inv(R)*a / (a^H inv(R) a).
        """
        invR = np.linalg.inv(R)
        num = invR @ a
        den = a.conj().T @ num
        return num / den

    def dbf(self, X: np.ndarray, theta: float, el: float) -> np.ndarray:
        """
        Perform digital beamforming on snapshot matrix X (M x N).
        Will validate FOV first.
        """
        a = self.steering_vector(theta, el)
        R = (X @ X.conj().T) / X.shape[1]
        w = self.mvdr_weights(R, a)
        return w.conj().T @ X

    def mimo_covariance(self, A_R: np.ndarray, R_s: np.ndarray, A_T: np.ndarray) -> np.ndarray:
        """
        Compute MIMO covariance: A_R * R_s * A_T^H.
        """
        return A_R @ R_s @ A_T.conj().T

    def stap_weights(self, X_stap: np.ndarray, a_stap: np.ndarray) -> np.ndarray:
        """
        Compute STAP weights: inv(R_stap) * a_stap.
        """
        R_stap = (X_stap @ X_stap.conj().T) / X_stap.shape[1]
        return np.linalg.inv(R_stap) @ a_stap

    def pulse_compress(self, r: np.ndarray) -> np.ndarray:
        """
        Matched-filter pulse compression of received signal r.
        """
        t = np.linspace(0, self.params.T_p, len(r))
        K = self.params.B / self.params.T_p
        s = np.exp(1j * np.pi * K * t**2)
        h = np.conj(s[::-1])
        return np.convolve(r, h, mode='same')

    def cfar_threshold(self, window: np.ndarray) -> float:
        """
        Compute CFAR threshold using surrounding window cells.
        """
        N = len(window) // 2
        alpha = N * (self.params.P_fa**(-1/N) - 1)
        noise = (np.sum(window[:N]) + np.sum(window[N+1:])) / (2*N)
        return alpha * noise

    def detect(self, cell: complex, threshold: float) -> bool:
        """
        Return True if cell energy exceeds CFAR threshold.
        """
        return np.abs(cell)**2 >= threshold

    def clutter_covariance(self, a_c_list: list, sigma_c_list: list) -> np.ndarray:
        """
        Build clutter covariance from steering vectors and powers.
        """
        return sum(
            sigma**2 * np.outer(a_c, a_c.conj())
            for a_c, sigma in zip(a_c_list, sigma_c_list)
        )

    def robust_lcmv_weights(self, R: np.ndarray, a0: np.ndarray, A: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Robust LCMV via diagonal loading approximation.
        """
        R_loaded = R + (epsilon**2) * np.eye(R.shape[0])
        invR = np.linalg.inv(R_loaded)
        num = invR @ a0
        den = a0.conj().T @ num
        return num / den
