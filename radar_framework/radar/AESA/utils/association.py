# radar_framework/radar/AESA/utils/association.py

from abc import ABC, abstractmethod
import numpy as np

from radar_framework.radar.AESA import exceptions
from radar_framework.radar.AESA.utils.validation import (
    assert_ndarray,
    assert_shape,
    assert_list,
    assert_list_length,
    assert_list_of_ndarray,
)

class AssociationEngine(ABC):
    """
    Abstract base class for data association engines.
    """
    def __init__(self, gate_threshold: float, Pd: float, lambda_c: float):
        self.gate_threshold = gate_threshold  # chi-square gating threshold
        self.Pd = Pd                          # detection probability
        self.lambda_c = lambda_c              # spatial clutter rate

    @abstractmethod
    def gate(self, track, measurements: np.ndarray, H: np.ndarray, R: np.ndarray) -> list:
        """
        Return a list of measurement indices that pass the gate for the given track.
        """
        # Input validation
        if not hasattr(track, 'filter') or not hasattr(track.filter, 'x'):
            raise exceptions.AESAError("track must have 'filter.x' attribute")
        assert_ndarray(measurements, "measurements", 2)
        M = measurements.shape[1]
        N = track.filter.x.shape[0]
        assert_shape(H, "H", (M, N))
        assert_shape(R, "R", (M, M))

        try:
            gated = []
            x = track.filter.x
            S = H @ track.filter.P @ H.T + R
            invS = np.linalg.inv(S)
            for j, z in enumerate(measurements):
                v = z - H @ x
                # Normalize angle residual
                v[1] = (v[1] + np.pi) % (2 * np.pi) - np.pi
                dist2 = float(v.T @ invS @ v)
                if dist2 <= self.gate_threshold:
                    gated.append(j)
            return gated

        except np.linalg.LinAlgError as e:
            raise exceptions.AESAError(f"Gate computation failed (matrix inversion error): {e}")
        except AttributeError as e:
            raise exceptions.AESAError(f"Invalid track or measurement data: {e}")
        except Exception as e:
            raise exceptions.AESAError(f"Unexpected error in gating: {e}")

    @abstractmethod
    def associate(self, tracks: list, measurements: list, Hs: list, Rs: list):
        """
        Perform association: JPDA returns association probabilities (betas),
        MHT returns or updates hypothesis tree and applies the best hypothesis.
        """
        # Input validation
        assert_list(tracks, "tracks")
        assert_list(measurements, "measurements")
        assert_list_of_ndarray(measurements, "measurements", 1)
        assert_list(Hs, "Hs")
        assert_list(Rs, "Rs")
        n = len(tracks)
        assert_list_length(Hs, "Hs", n)
        assert_list_length(Rs, "Rs", n)
        assert_list_of_ndarray(Hs, "Hs", 2)
        assert_list_of_ndarray(Rs, "Rs", 2)

        try:
            # Concrete implementations must override this method
            raise NotImplementedError
        except NotImplementedError:
            raise
        except Exception as e:
            raise exceptions.AESAError(f"Unexpected error in association: {e}")
