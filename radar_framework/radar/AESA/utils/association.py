# radar_framework/radar/AESA/utils/association.py

from abc import ABC, abstractmethod
import numpy as np

from radar_framework.radar.AESA import exceptions

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
        try:
            # Concrete implementations must override this method
            raise NotImplementedError
        except NotImplementedError:
            raise
        except Exception as e:
            raise exceptions.AESAError(f"Unexpected error in association: {e}")
