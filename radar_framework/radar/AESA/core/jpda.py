# radar_framework/radar/AESA/core/jpda.py

import numpy as np

from radar_framework.radar.AESA import exceptions
from radar_framework.radar.AESA.utils import AssociationEngine

class JPDA(AssociationEngine):
    """
    Joint Probabilistic Data Association engine.
    Computes association probabilities for tracks and measurements.
    """
    def __init__(self, gate_threshold: float, Pd: float, lambda_c: float):
        super().__init__(gate_threshold, Pd, lambda_c)

    def gate(self, track, measurements, H, R):
        """
        Return list of measurement indices within the chi-square gate for a single track.
        """
        try:
            x = track.filter.x
            S = H @ track.filter.P @ H.T + R
            invS = np.linalg.inv(S)
            gated = []
            for j, z in enumerate(measurements):
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

    def associate(self, tracks, measurements, Hs, Rs):
        """
        Build joint association hypotheses and compute beta_ij probabilities.
        Returns a list of beta arrays, one per track (last element is missed detection).
        """
        try:
            Ntr, Nme = len(tracks), len(measurements)
            # 1) Perform gating for each track
            gated_indices = [self.gate(tracks[i], measurements, Hs[i], Rs[i])
                             for i in range(Ntr)]
            # 2) Generate all joint hypotheses recursively
            hyps = []
            def recurse(i, assignment):
                if i == Ntr:
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
            for i in range(Ntr):
                beta_i = np.zeros(Nme + 1)
                for h, ass in enumerate(hyps):
                    idx = ass[i] if ass[i] is not None else Nme
                    beta_i[idx] += probs[h]
                betas.append(beta_i)

            return betas
        except np.linalg.LinAlgError as e:
            raise exceptions.AESAError(f"JPDA association linear algebra error: {e}")
        except Exception as e:
            raise exceptions.AESAError(f"JPDA association unexpected error: {e}")
