# radar_framework/radar/PESA/core/models.py

import numpy as np
from dataclasses import dataclass

from radar_framework.radar.PESA.exceptions import PESAError
from radar_framework.radar.AESA.core.models import AESA, AESAParams

@dataclass
class PESAParams:
    wavelength: float
    element_positions: np.ndarray
    delta_phi: float           # phase quantization step (rad)
    subarray_size: int         # number of elements per subarray
    B: float                   # bandwidth for pulse compression
    T_p: float                 # pulse duration
    P_fa: float                # false-alarm rate
    az_limits: tuple = (-np.pi/2, np.pi/2)
    el_limits: tuple = (-np.pi/4, np.pi/4)

class PESA:
    """
    Passive Electronically Scanned Array (PESA) radar model.
    Implements phase quantization, subarray beamforming,
    phase calibration, FOV check, STAP/pulse compression (via AESA),
    diversity receive, and EKF-based state estimation.
    """
    def __init__(self, params: PESAParams):
        self.params = params
        # reuse AESA functions for STAP and pulse compression
        aesa_params = AESAParams(
            wavelength=params.wavelength,
            element_positions=params.element_positions,
            B=params.B,
            T_p=params.T_p,
            P_fa=params.P_fa
        )
        self._aesa = AESA(aesa_params)

    def in_fov(self, theta: float, el: float) -> bool:
        """
        Check if scan angles are within the configured field-of-view.
        """
        az_min, az_max = self.params.az_limits
        el_min, el_max = self.params.el_limits
        return az_min <= theta <= az_max and el_min <= el <= el_max

    def phase_quantization(self, phases: np.ndarray) -> np.ndarray:
        """
        Apply DAC/PLL quantization error to steering phases.
        w_n = (1/sqrt(N)) e^{-j(phi_n + epsilon_n)},
        epsilon_n ~ U(-DeltaPhi/2, DeltaPhi/2)
        """
        if not hasattr(phases, 'size'):
            raise PESAError("phase_quantization: invalid input phases array")
        try:
            N = phases.size
            eps = (np.random.rand(N) - 0.5) * self.params.delta_phi
            quantized = phases + eps
            weights = np.exp(-1j * quantized) / np.sqrt(N)
            return weights
        except Exception as e:
            raise PESAError(f"phase_quantization error: {e}")

    def subarray_beamforming(self, theta: float) -> complex:
        """
        AF(theta) = sum_{m=1}^M w_m sum_{n=1}^{N/M} exp(j k d n sin(theta)).
        Uniform w_m = 1/sqrt(M). Checks FOV before beamforming.
        """
        if not self.in_fov(theta, 0.0):
            raise PESAError(f"subarray_beamforming: azimuth angle {theta} out of FOV")
        try:
            pos = self.params.element_positions
            N = pos.shape[0]
            M = self.params.subarray_size
            k = 2 * np.pi / self.params.wavelength
            AF = 0+0j
            for start in range(0, N, M):
                arr = pos[start:start+M]
                # assume uniform phase=0 weights per subarray
                w_m = 1/np.sqrt(M)
                AF += w_m * np.sum(np.exp(1j * k * (arr @ np.array([np.sin(theta), 0, 0]))))
            return AF
        except Exception as e:
            raise PESAError(f"subarray_beamforming error: {e}")

    def phase_calibration(self, phi_meas: np.ndarray, G_meas: np.ndarray, G_ref: np.ndarray) -> np.ndarray:
        """
        phi_cal = phi_meas - angle(G_meas / G_ref)
        """
        if phi_meas.shape != G_meas.shape or G_meas.shape != G_ref.shape:
            raise PESAError("phase_calibration: input arrays must match shape")
        try:
            delta = np.angle(G_meas / G_ref)
            return phi_meas - delta
        except Exception as e:
            raise PESAError(f"phase_calibration error: {e}")

    def stap_weights(self, X_stap: np.ndarray, a_stap: np.ndarray) -> np.ndarray:
        """STAP weights identical to AESA implementation."""
        try:
            return self._aesa.stap_weights(X_stap, a_stap)
        except Exception as e:
            raise PESAError(f"stap_weights error: {e}")

    def pulse_compress(self, r: np.ndarray) -> np.ndarray:
        """Pulse compression identical to AESA implementation."""
        try:
            return self._aesa.pulse_compress(r)
        except Exception as e:
            raise PESAError(f"pulse_compress error: {e}")

    def diversity_receive(self, y_list: list, sigma_list: list) -> np.ndarray:
        """
        y(t) = sum_{p=1}^P alpha_p y_p(t),  alpha_p = 1/sigma_p^2
        """
        if len(y_list) != len(sigma_list):
            raise PESAError("diversity_receive: signals and sigmas list must be same length")
        try:
            alphas = [1/(s**2) for s in sigma_list]
            return sum(a * y for a,y in zip(alphas, y_list))
        except Exception as e:
            raise PESAError(f"diversity_receive error: {e}")

    def ekf_update(self, x, P, F, Q, H, R, z):
        """EKF predict-update steps."""
        try:
            # predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            # update
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_upd = x_pred + K @ (z - H @ x_pred)
            P_upd = (np.eye(P.shape[0]) - K @ H) @ P_pred
            return x_upd, P_upd
        except Exception as e:
            raise PESAError(f"ekf_update error: {e}")