# radar_framework/radar/AESA/pipeline.py

import numpy as np

from .core.models import AESA, AESAParams
from radar_framework.radar.AESA.utils import AssociationEngine
from radar_framework.radar.AESA import exceptions
from .core.jpda import JPDA
from .core.mht import MHT
from radar_framework.radar.AESA.utils.validation import (
    assert_ndarray,
    assert_scalar,
    assert_positive_scalar,
    assert_list,
    assert_list_length,
    assert_list_of_ndarray
)

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
        # Input validation
        if not isinstance(params, AESAParams):
            raise exceptions.AESAError("params must be an AESAParams instance")
        # Validate optional scalar arguments
        if assoc_engine is None:
            assert_positive_scalar(Pd, "Pd")
            assert_positive_scalar(clutter_rate, "clutter_rate")
            assert_positive_scalar(gate_threshold, "gate_threshold")
            if not isinstance(max_hypotheses, int) or max_hypotheses <= 0:
                raise exceptions.AESAError(f"max_hypotheses must be a positive integer, got {max_hypotheses}")
            if not isinstance(use_mht, bool):
                raise exceptions.AESAError(f"use_mht must be a boolean, got {type(use_mht).__name__}")

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
        # Input validation
        assert_ndarray(X, "X", 2)
        assert_scalar(theta, "theta")
        assert_scalar(el, "el")
        assert_ndarray(window, "window", 1)

        try:
            y = self.aesa.dbf(X, theta, el)
            y_pc = self.aesa.pulse_compress(y)
            thr = self.aesa.cfar_threshold(window)
            dets = [self.aesa.detect(c, thr) for c in y_pc]
            assoc = None
            if tracks is not None and measurements is not None and Hs is not None and Rs is not None:
                # Validate association inputs
                assert_list(tracks, "tracks")
                assert_list(measurements, "measurements")
                assert_list(Hs, "Hs")
                assert_list(Rs, "Rs")
                n = len(tracks)
                assert_list_length(Hs, "Hs", n)
                assert_list_length(Rs, "Rs", n)
                assert_list_of_ndarray(measurements, "measurements", 1)
                assert_list_of_ndarray(Hs, "Hs", 2)
                assert_list_of_ndarray(Rs, "Rs", 2)
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
