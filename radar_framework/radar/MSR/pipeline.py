# radar_framework/radar/MSR/pipeline.py

import numpy as np

from radar_framework.radar.MSR.core.models import MSRParams, MSR
from radar_framework.radar.MSR.exceptions import MSRError

class MSRPipeline:
    """
    End-to-end MSR processing pipeline.
    """
    def __init__(self, params: MSRParams):
        self.msr = MSR(params)

    def process(
        self,
        weights: np.ndarray,
        snr_funcs: list,
        theta_cmd: float,
        measurements: list,
        pdf_H1,
        y_list: list,
        x, P, F, Q, H, R, z
    ) -> dict:
        """
        1) revisit optimization
        2) servo TF
        3) phase correction
        4) Track-Before-Detect
        5) coherent/noncoherent integration
        6) MTI & MTD filtering
        7) EKF update
        """
        try:
            taus = self.msr.revisit_optimization(weights, snr_funcs)
            num, den = self.msr.servo_transfer()
            theta_act = self.msr.phase_correction(theta_cmd)
            tbd_score = self.msr.track_before_detect(measurements, pdf_H1)
            y_coh = self.msr.coherent_integration(y_list)
            y_noncoh = self.msr.noncoherent_integration(y_list)
            y_mti = self.msr.mti_filter(np.array(y_list))
            Y_mtd = self.msr.mtd_spectrum(np.array(y_list))
            x_upd, P_upd = self.msr.ekf_update(x, P, F, Q, H, R, z)
            return {
                'taus': taus,
                'servo_tf': {'num': num, 'den': den},
                'theta_actual': theta_act,
                'tbd_score': tbd_score,
                'y_coherent': y_coh,
                'y_noncoherent': y_noncoh,
                'y_mti': y_mti,
                'Y_mtd': Y_mtd,
                'ekf': {'x': x_upd, 'P': P_upd}
            }
        except MSRError:
            raise
        except Exception as e:
            raise MSRError(f"MSRPipeline process failed: {e}")