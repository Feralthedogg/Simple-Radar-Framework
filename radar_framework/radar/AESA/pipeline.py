# radar_framework/radar/AESA/pipeline.py

import numpy as np

from .core.models import AESA, AESAParams
from radar_framework.radar.AESA.utils import AssociationEngine
from radar_framework.radar.AESA import exceptions
from .core.jpda import JPDA
from .core.mht import MHT

class AESAPipeline:
    """
    End-to-end AESA processing pipeline.
    Performs beamforming, pulse compression, CFAR detection, and data association.
    Supports both legacy API (Pd, clutter_rate, gate_threshold, max_hypotheses, use_mht)
    and injection API (assoc_engine).
    """
    def __init__(
        self,
        params: AESAParams,
        Pd: float = None,
        clutter_rate: float = None,
        gate_threshold: float = None,
        max_hypotheses: int = 100,
        use_mht: bool = False,
        assoc_engine: AssociationEngine = None
    ):
        self.aesa = AESA(params)

        # Injection API: assoc_engine provided
        if assoc_engine is not None:
            if not isinstance(assoc_engine, AssociationEngine):
                raise exceptions.AESAError("assoc_engine must inherit AssociationEngine")
            self.assoc_engine = assoc_engine
            return

        # Legacy API: use Pd, clutter_rate, gate_threshold
        if Pd is None or clutter_rate is None or gate_threshold is None:
            raise exceptions.AESAError(
                "Must provide either assoc_engine or Pd, clutter_rate, gate_threshold"
            )
        try:
            if use_mht:
                self.assoc_engine = MHT(
                    gate_threshold=gate_threshold,
                    Pd=Pd,
                    lambda_c=clutter_rate,
                    max_hypotheses=max_hypotheses
                )
            else:
                self.assoc_engine = JPDA(
                    gate_threshold=gate_threshold,
                    Pd=Pd,
                    lambda_c=clutter_rate
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
        """
        Run one frame of radar processing.

        1) `dbf` beamforming
        2) `pulse_compress`
        3) `cfar_threshold` + detection
        4) association via configured engine

        Returns a dict with keys: 'beamformed', 'pulse_compressed', 'detections', 'association'.
        """
        try:
            y = self.aesa.dbf(X, theta, el)
            y_pc = self.aesa.pulse_compress(y)
            thr = self.aesa.cfar_threshold(window)
            dets = [self.aesa.detect(c, thr) for c in y_pc]
            assoc = None
            if tracks is not None and measurements is not None and Hs is not None and Rs is not None:
                assoc = self.assoc_engine.associate(tracks, measurements, Hs, Rs)
            return {
                'beamformed': y,
                'pulse_compressed': y_pc,
                'detections': dets,
                'association': assoc
            }
        except exceptions.AESAError:
            raise
        except Exception as e:
            raise exceptions.AESAError(f"AESAPipeline process failed: {e}")
