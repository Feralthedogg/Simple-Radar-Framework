# radar_framework/radar/AESA/core/jpda.py

import numpy as np

from radar_framework.radar.AESA import exceptions
from radar_framework.radar.AESA.utils.validation import (
    assert_ndarray,
    assert_list,
    assert_list_length,
    assert_list_of_ndarray,
    assert_square_matrix
)
from radar_framework.radar.AESA.utils import AssociationEngine

class JPDA(AssociationEngine):
    """
    Joint Probabilistic Data Association engine.
    Computes association probabilities for tracks and measurements.
    """
    def __init__(self, gate_threshold: float, Pd: float, lambda_c: float):
        super().__init__(gate_threshold, Pd, lambda_c)

    def gate(self, track, measurements: np.ndarray, H: np.ndarray, R: np.ndarray) -> list:
        """
        Return list of measurement indices within the chi-square gate for a single track.
        """
        # Input validation
        if not hasattr(track, 'filter') or not hasattr(track.filter, 'x'):
            raise exceptions.AESAError("track must have 'filter.x' attribute")
        assert_ndarray(measurements, 'measurements', 2)
        assert_ndarray(H, 'H', 2)
        assert_square_matrix(R, 'R')
        M, N = measurements.shape[1], track.filter.x.shape[0]
        if H.shape != (M, N):
            raise exceptions.AESAError(f"H must have shape ({M},{N}), got {H.shape}")

        try:
            x = track.filter.x
            S = H @ track.filter.P @ H.T + R
            invS = np.linalg.inv(S)
            gated = []
            for j, z in enumerate(measurements):
                assert_ndarray(z, f'measurements[{j}]', 1)
                v = z - H @ x
                # normalize the angular component
                v[1] = (v[1] + np.pi) % (2 * np.pi) - np.pi
                dist2 = float(v.T @ invS @ v)
                if dist2 <= self.gate_threshold:
                    gated.append(j)
            return gated
        except np.linalg.LinAlgError as e:
            raise exceptions.AESAError(f"JPDA gate linear algebra error: {e}")
        except AttributeError as e:
            raise exceptions.AESAError(f"JPDA gate attribute error: {e}")
        except Exception as e:
            raise exceptions.AESAError(f"JPDA gate unexpected error: {e}")

    def associate(self, tracks: list, measurements: list, Hs: list, Rs: list) -> list:
        """
        Build joint association hypotheses and compute beta_ij probabilities.
        Returns a list of beta arrays, one per track (last element is missed detection).
        """
        # Input validation
        assert_list(tracks, 'tracks')
        assert_list(measurements, 'measurements')
        assert_list_of_ndarray(measurements, 'measurements', 1)
        assert_list(Hs, 'Hs')
        assert_list(Rs, 'Rs')
        n = len(tracks)
        assert_list_length(Hs, 'Hs', n)
        assert_list_length(Rs, 'Rs', n)
        # Validate each H and R
        for i in range(n):
            assert_ndarray(Hs[i], f'Hs[{i}]', 2)
            assert_square_matrix(Rs[i], f'Rs[{i}]')
            # H shape consistency check
            M, N = (measurements[0].shape[0] if measurements else None), tracks[i].filter.x.shape[0]
            if measurements:
                if Hs[i].shape != (M, N):
                    raise exceptions.AESAError(f"Hs[{i}] must have shape ({M},{N}), got {Hs[i].shape}")

        try:
            # 1) Perform gating for each track
            gated_indices = [self.gate(tracks[i], measurements, Hs[i], Rs[i])
                             for i in range(n)]
            # 2) Generate all joint hypotheses recursively
            hyps = []
            def recurse(i, assignment):
                if i == n:
                    hyps.append(assignment.copy())
                    return
                for m in gated_indices[i] + [None]:
                    if m is not None and m in assignment.values():
                        continue
                    assignment[i] = m
                    recurse(i + 1, assignment)
            recurse(0, {})

            # 3) Compute likelihood L(h) for each hypothesis
            L = np.zeros(len(hyps))
            for h, ass in enumerate(hyps):
                Lh = 1.0
                for i, m in ass.items():
                    if m is None:
                        # missed detection
                        Lh *= (1 - self.Pd)
                    else:
                        H, R = Hs[i], Rs[i]
                        z = measurements[m]
                        z_pred = H @ tracks[i].filter.x
                        v = z - z_pred
                        S = H @ tracks[i].filter.P @ H.T + R
                        invS = np.linalg.inv(S)
                        exponent = np.exp(-0.5 * (v.T @ invS @ v))
                        norm = np.sqrt((2 * np.pi) ** len(v) * np.linalg.det(S))
                        Lh *= self.Pd * exponent / norm
                # include clutter term for each assigned measurement
                count = sum(1 for m in ass.values() if m is not None)
                Lh *= (self.lambda_c ** count)
                L[h] = Lh

            # 4) Normalize to get hypothesis probabilities
            Lsum = np.sum(L)
            probs = L / Lsum if Lsum > 0 else L

            # 5) Compute beta_ij for each track i and measurement j (plus miss)
            betas = []
            for i in range(n):
                beta_i = np.zeros(len(measurements) + 1)
                for h, ass in enumerate(hyps):
                    idx = ass[i] if ass[i] is not None else len(measurements)
                    beta_i[idx] += probs[h]
                betas.append(beta_i)

            return betas
        except np.linalg.LinAlgError as e:
            raise exceptions.AESAError(f"JPDA association linear algebra error: {e}")
        except Exception as e:
            raise exceptions.AESAError(f"JPDA association unexpected error: {e}")
