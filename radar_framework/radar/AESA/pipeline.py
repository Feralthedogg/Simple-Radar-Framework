# radar_framework/radar/AESA/pipeline.py

import numpy as np

from .core.models import AESA, AESAParams
from .core.jpda import JPDA
from .core.mht import MHT
from radar_framework.radar.AESA import exceptions

class AESAPipeline:
    """
    Orchestrates AESA processing: beamforming, pulse compression,
    CFAR detection, and optional JPDA/MHT association.
    """
    def __init__(
        self,
        params: AESAParams,
        Pd: float,
        clutter_rate: float,
        gate_threshold: float,
        max_hypotheses: int = 100,
        use_mht: bool = False
    ):
        self.aesa = AESA(params)
        try:
            if use_mht:
                self.assoc_engine = MHT(
                    Pd=Pd,
                    lambda_c=clutter_rate,
                    gate_threshold=gate_threshold,
                    max_hypotheses=max_hypotheses
                )
            else:
                self.assoc_engine = JPDA(
                    gate_threshold=gate_threshold,
                    Pd=Pd,
                    clutter_rate=clutter_rate
                )
        except Exception as e:
            raise exceptions.AESAError(f"Failed to initialize association engine: {e}")

    def process(
        self,
        X: np.ndarray,
        theta: float,
        el: float,
        window: np.ndarray,
        tracks=None,
        measurements=None,
        Hs=None,
        Rs=None
    ):
        try:
            # 1) Beamforming
            y = self.aesa.dbf(X, theta, el)
            # 2) Pulse compression
            y_pc = self.aesa.pulse_compress(y)
            # 3) CFAR detection
            threshold = self.aesa.cfar_threshold(window)
            detections = [self.aesa.detect(cell, threshold) for cell in y_pc]
            # 4) Association
            association = None
            if tracks is not None and measurements is not None and Hs is not None and Rs is not None:
                if isinstance(self.assoc_engine, JPDA):
                    association = self.assoc_engine.associate(tracks, measurements, Hs, Rs)
                else:
                    association = self.assoc_engine.update(tracks, measurements, Hs, Rs)
            return {
                'beamformed': y,
                'pulse_compressed': y_pc,
                'detections': detections,
                'association': association
            }
        except exceptions.AESAError:
            raise
        except Exception as e:
            raise exceptions.AESAError(f"AESAPipeline process failed: {e}")