# radar_framework/radar/MSR/core/models.py
import numpy as np
from dataclasses import dataclass

from radar_framework.radar.MSR.exceptions import MSRError
from radar_framework.radar.PESA.core.models import PESA, PESAParams

@dataclass
class MSRParams:
    """
    Parameters for Mechanical Scanning Radar (MSR).
    T_scan: total dwell time available for one scan cycle.
    wn: natural frequency of the servo (rad/s).
    zeta: damping ratio of the servo.
    sigma_theta: std dev of angular error for phase correction.
    pesa_params: PESAParams for delegating EKF update.
    """
    T_scan: float
    wn: float
    zeta: float
    sigma_theta: float
    pesa_params: PESAParams

class MSR:
    """
    Implements core functions for a mechanically scanned radar:
      - revisit time optimization
      - servo transfer function
      - Track--Before--Detect accumulation
      - phase correction
      - coherent/noncoherent integration
      - MTI & MTD filtering
      - EKF state update (delegated to PESA)
    """
    def __init__(self, params: MSRParams):
        self.params = params
        # reuse PESA for EKF implementation
        self._pesa = PESA(params.pesa_params)

    def revisit_optimization(self, weights: np.ndarray, snr_funcs: list,
                              max_iter: int = 500, tol: float = 1e-6) -> np.ndarray:
        """
        Solve:
          max_{tau} sum_i w_i log(1+snr_i(tau_i))
          s.t. sum_i tau_i = T_scan, tau_i >= 0
        via projected gradient ascent.
        """
        try:
            n = weights.size
            tau = np.full(n, self.params.T_scan / n)
            alpha = self.params.T_scan * 0.1
            h = self.params.T_scan * 1e-4
            for _ in range(max_iter):
                snr = np.array([f(tau[i]) for i, f in enumerate(snr_funcs)])
                grad = np.zeros(n)
                for i, f in enumerate(snr_funcs):
                    t = tau[i]
                    d = (f(t + h) - f(t - h)) / (2 * h)
                    grad[i] = weights[i] * d / (1.0 + snr[i])
                tau_old = tau.copy()
                tau += alpha * grad
                tau = self._project_simplex(tau, self.params.T_scan)
                if np.linalg.norm(tau - tau_old) < tol:
                    break
            return tau
        except Exception as e:
            raise MSRError(f"revisit_optimization failed: {e}")

    @staticmethod
    def _project_simplex(v: np.ndarray, s: float) -> np.ndarray:
        """
        Project v onto simplex {x >= 0, sum x = s}.
        """
        if s < 0:
            raise ValueError("Simplex sum must be non-negative")
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - s))[0]
        rho = rho[-1] if rho.size > 0 else 0
        theta = (cssv[rho] - s) / (rho + 1)
        return np.maximum(v - theta, 0)

    def servo_transfer(self) -> tuple:
        """
        Returns numerator and denominator coefficients for H(s).
        """
        try:
            wn = self.params.wn
            zeta = self.params.zeta
            num = [wn**2]
            den = [1.0, 2 * zeta * wn, wn**2]
            return num, den
        except Exception as e:
            raise MSRError(f"servo_transfer failed: {e}")

    def track_before_detect(self, z_list: list, pdf_H1) -> float:
        """
        Unnormalized P(H1|{z_k}) ∝ ∏_k pdf_H1(z_k).
        """
        try:
            prod = 1.0
            for z in z_list:
                prod *= pdf_H1(z)
            return prod
        except Exception as e:
            raise MSRError(f"track_before_detect failed: {e}")

    def phase_correction(self, theta_cmd: float) -> float:
        """
        θ_actual = θ_cmd + δθ, where δθ ~ N(0, σ_θ^2).
        """
        try:
            delta = np.random.randn() * self.params.sigma_theta
            return theta_cmd + delta
        except Exception as e:
            raise MSRError(f"phase_correction failed: {e}")

    def coherent_integration(self, y_list: list) -> complex:
        """y_coh = sum(y_list)"""
        try:
            return sum(y_list)
        except Exception as e:
            raise MSRError(f"coherent_integration failed: {e}")

    def noncoherent_integration(self, y_list: list) -> float:
        """y_noncoh = sum(|y|^2 for y in y_list)"""
        try:
            return sum(np.abs(y)**2 for y in y_list)
        except Exception as e:
            raise MSRError(f"noncoherent_integration failed: {e}")

    def mti_filter(self, x: np.ndarray) -> np.ndarray:
        """y[n] = x[n] - 2x[n-1] + x[n-2]"""
        try:
            if x.size < 3:
                raise MSRError("Input length must be >=3 for MTI filter")
            return x - 2 * np.roll(x, 1) + np.roll(x, 2)
        except Exception as e:
            raise MSRError(f"mti_filter failed: {e}")

    def mtd_spectrum(self, x: np.ndarray) -> np.ndarray:
        """X[k] = FFT(x)"""
        try:
            return np.fft.fft(x)
        except Exception as e:
            raise MSRError(f"mtd_spectrum failed: {e}")

    def ekf_update(self, x, P, F, Q, H, R, z):
        """
        Delegate EKF predict-update to PESA implementation.
        """
        try:
            return self._pesa.ekf_update(x, P, F, Q, H, R, z)
        except Exception as e:
            raise MSRError(f"ekf_update failed: {e}")