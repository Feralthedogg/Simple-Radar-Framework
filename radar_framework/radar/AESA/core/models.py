# radar_framework/radar/AESA/core/models.py

import numpy as np
from dataclasses import dataclass

from radar_framework.radar.AESA import exceptions
from radar_framework.radar.AESA.utils.validation import (
    assert_ndarray,
    assert_scalar,
    assert_positive_scalar,
    assert_min_length,
    assert_square_matrix,
    assert_list,
    assert_list_length,
    assert_list_of_ndarray,
)

@dataclass
class AESAParams:
    wavelength: float
    element_positions: np.ndarray
    B: float
    T_p: float
    P_fa: float
    az_limits: tuple = (-np.pi / 2, np.pi / 2)
    el_limits: tuple = (-np.pi / 4, np.pi / 4)

    def __post_init__(self):
        # Validate scalar parameters
        assert_positive_scalar(self.wavelength, "wavelength")
        assert_positive_scalar(self.B, "B")
        assert_positive_scalar(self.T_p, "T_p")
        assert_positive_scalar(self.P_fa, "P_fa")
        if not (0 < self.P_fa < 1):
            raise exceptions.AESAError(f"P_fa must be in (0,1), got {self.P_fa}")
        # Validate array parameters
        assert_ndarray(self.element_positions, "element_positions", 2)
        # Validate limits
        min_az, max_az = self.az_limits
        if min_az >= max_az:
            raise exceptions.AESAError(f"az_limits must be (min,max), got {self.az_limits}")
        min_el, max_el = self.el_limits
        if min_el >= max_el:
            raise exceptions.AESAError(f"el_limits must be (min,max), got {self.el_limits}")

class AESA:
    """
    Implements core AESA radar processing functions.
    """
    def __init__(self, params: AESAParams):
        self.params = params

    def in_fov(self, theta: float, el: float) -> bool:
        az_min, az_max = self.params.az_limits
        el_min, el_max = self.params.el_limits
        return az_min <= theta <= az_max and el_min <= el <= el_max

    def steering_vector(self, theta: float, el: float) -> np.ndarray:
        # Input validation
        assert_scalar(theta, "theta")
        assert_scalar(el, "el")
        pos = self.params.element_positions
        assert_ndarray(pos, "element_positions", 2)

        try:
            if not self.in_fov(theta, el):
                raise exceptions.AESAError(f"Scan angle out of FOV: az={theta}, el={el}")
            k = 2 * np.pi / self.params.wavelength
            d = np.array([
                np.cos(el) * np.cos(theta),
                np.cos(el) * np.sin(theta),
                np.sin(el)
            ])
            return np.exp(1j * k * (pos @ d))
        except exceptions.AESAError:
            raise
        except Exception as e:
            raise exceptions.AESAError(f"steering_vector error: {e}")

    def mvdr_weights(self, R: np.ndarray, a: np.ndarray) -> np.ndarray:
        # Input validation
        assert_square_matrix(R, "R")
        assert_ndarray(a, "a", 1)
        if a.shape[0] != R.shape[0]:
            raise exceptions.AESAError(f"Vector a length ({a.shape[0]}) must match R.shape[0] ({R.shape[0]})")

        try:
            invR = np.linalg.inv(R)
            num = invR @ a
            den = a.conj().T @ num
            return num / den
        except np.linalg.LinAlgError as e:
            raise exceptions.AESAError(f"mvdr_weights inversion failed: {e}")
        except Exception as e:
            raise exceptions.AESAError(f"mvdr_weights error: {e}")

    def dbf(self, X: np.ndarray, theta: float, el: float) -> np.ndarray:
        # Input validation
        assert_ndarray(X, "X", 2)
        assert_scalar(theta, "theta")
        assert_scalar(el, "el")

        try:
            a = self.steering_vector(theta, el)
            R = (X @ X.conj().T) / X.shape[1]
            w = self.mvdr_weights(R, a)
            return w.conj().T @ X
        except exceptions.AESAError:
            raise
        except Exception as e:
            raise exceptions.AESAError(f"dbf processing failed: {e}")

    def mimo_covariance(self, A_R: np.ndarray, R_s: np.ndarray, A_T: np.ndarray) -> np.ndarray:
        # Input validation
        assert_ndarray(A_R, "A_R", 2)
        assert_ndarray(R_s, "R_s", 2)
        assert_ndarray(A_T, "A_T", 2)

        try:
            return A_R @ R_s @ A_T.conj().T
        except Exception as e:
            raise exceptions.AESAError(f"mimo_covariance error: {e}")

    def stap_weights(self, X_stap: np.ndarray, a_stap: np.ndarray) -> np.ndarray:
        # Input validation
        assert_ndarray(X_stap, "X_stap", 2)
        assert_ndarray(a_stap, "a_stap", 1)
        assert_min_length(X_stap, "X_stap", 1)

        try:
            R_stap = (X_stap @ X_stap.conj().T) / X_stap.shape[1]
            return np.linalg.inv(R_stap) @ a_stap
        except np.linalg.LinAlgError as e:
            raise exceptions.AESAError(f"stap_weights inversion failed: {e}")
        except Exception as e:
            raise exceptions.AESAError(f"stap_weights error: {e}")

    def pulse_compress(self, r: np.ndarray) -> np.ndarray:
        # Input validation
        assert_ndarray(r, "r", 1)

        try:
            t = np.linspace(0, self.params.T_p, len(r))
            K = self.params.B / self.params.T_p
            s = np.exp(1j * np.pi * K * t**2)
            h = np.conj(s[::-1])
            return np.convolve(r, h, mode='same')
        except Exception as e:
            raise exceptions.AESAError(f"pulse_compress error: {e}")

    def cfar_threshold(self, window: np.ndarray) -> float:
        # Input validation
        assert_ndarray(window, "window", 1)
        assert_min_length(window, "window", 3)

        try:
            N = len(window) // 2
            alpha = N * (self.params.P_fa ** (-1/N) - 1)
            noise = (np.sum(window[:N]) + np.sum(window[N+1:])) / (2 * N)
            return alpha * noise
        except Exception as e:
            raise exceptions.AESAError(f"cfar_threshold error: {e}")

    def detect(self, cell: complex, threshold: float) -> bool:
        # Input validation
        assert_scalar(cell, "cell")
        assert_scalar(threshold, "threshold")

        try:
            return np.abs(cell)**2 >= threshold
        except Exception as e:
            raise exceptions.AESAError(f"detect error: {e}")

    def clutter_covariance(self, a_c_list: list, sigma_c_list: list) -> np.ndarray:
        # Input validation
        assert_list(a_c_list, "a_c_list")
        assert_list(sigma_c_list, "sigma_c_list")
        assert_list_length(sigma_c_list, "sigma_c_list", len(a_c_list))
        # Each steering vector
        assert_list_of_ndarray(a_c_list, "a_c_list", 1)
        # Each sigma scalar
        for idx, sigma in enumerate(sigma_c_list):
            assert_scalar(sigma, f"sigma_c_list[{idx}]")

        try:
            return sum(
                sigma**2 * np.outer(a_c, a_c.conj())
                for a_c, sigma in zip(a_c_list, sigma_c_list)
            )
        except Exception as e:
            raise exceptions.AESAError(f"clutter_covariance error: {e}")

    def robust_lcmv_weights(self, R: np.ndarray, a0: np.ndarray, A: np.ndarray, epsilon: float) -> np.ndarray:
        # Input validation
        assert_square_matrix(R, "R")
        assert_ndarray(a0, "a0", 1)
        assert_ndarray(A, "A", 2)
        assert_scalar(epsilon, "epsilon")
        if R.shape[0] != A.shape[0] or A.shape[1] != a0.shape[0]:
            raise exceptions.AESAError("Dimension mismatch among R, A, and a0")

        try:
            R_loaded = R + (epsilon**2) * np.eye(R.shape[0])
            invR = np.linalg.inv(R_loaded)
            num = invR @ a0
            den = a0.conj().T @ num
            return num / den
        except np.linalg.LinAlgError as e:
            raise exceptions.AESAError(f"robust_lcmv_weights inversion failed: {e}")
        except Exception as e:
            raise exceptions.AESAError(f"robust_lcmv_weights error: {e}")