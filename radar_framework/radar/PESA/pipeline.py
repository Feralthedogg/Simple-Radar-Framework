# radar_framework/radar/PESA/pipeline.py

import numpy as np

from radar_framework.radar.PESA.core.models import PESA, PESAParams
from radar_framework.radar.PESA.exceptions import PESAError

class PESPAPipeline:
    """
    Orchestrates PESA processing: phase quantization, subarray beamforming,
    pulse compression, STAP, diversity receive, EKF updates, and FOV checking.
    """
    def __init__(
        self,
        params: PESAParams
    ):
        self.pesa = PESA(params)

    def process(
        self,
        phases: np.ndarray,
        theta: float,
        r: np.ndarray,
        X_stap: np.ndarray = None,
        a_stap: np.ndarray = None,
        diversity_signals: list = None,
        diversity_sigmas: list = None,
        ekf_state: dict = None
    ):
        try:
            # FOV check
            if not self.pesa.in_fov(theta, 0.0):
                raise PESAError(f"Process: azimuth angle {theta} out of FOV")
            # 1) Apply phase quantization to steering
            w = self.pesa.phase_quantization(phases)
            # 2) Subarray beamforming
            AF = self.pesa.subarray_beamforming(theta)
            # 3) Pulse compression
            y_pc = self.pesa.pulse_compress(r)
            # 4) STAP
            stap_out = None
            if X_stap is not None and a_stap is not None:
                stap_out = self.pesa.stap_weights(X_stap, a_stap)
            # 5) Diversity receive
            div_out = None
            if diversity_signals is not None and diversity_sigmas is not None:
                div_out = self.pesa.diversity_receive(diversity_signals, diversity_sigmas)
            # 6) EKF update
            ekf_out = None
            if ekf_state is not None:
                ekf_out = self.pesa.ekf_update(**ekf_state)

            return {
                'in_fov': True,
                'weights': w,
                'af': AF,
                'pulse_compressed': y_pc,
                'stap': stap_out,
                'diversity': div_out,
                'ekf': ekf_out
            }
        except PESAError:
            raise
        except Exception as e:
            raise PESAError(f"PESPAPipeline process failed: {e}")