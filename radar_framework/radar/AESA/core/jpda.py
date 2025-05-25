# radar_framework/radar/AESA/core/jpda.py

import numpy as np

class JPDA:
    """
    Joint Probabilistic Data Association (JPDA) engine.
    Computes association probabilities for tracks and measurements.
    """
    def __init__(self, gate_threshold, Pd, clutter_rate):
        self.gate_threshold = gate_threshold # chi-square gating threshold
        self.Pd = Pd # detection probability
        self.lambda_c = clutter_rate # spatial clutter rate

    def gate(self, track, measurements, H, R):
        """"Return list of measurements within gate for a single track."""
        gated = []
        x = track.filter.x
        S = H @ track.filter.P @ H.T + R
        invS = np.linalg.inv(S)
        for j, z in enumerate(measurements):
            v = z - H @ x
            v[1] = (v[1] + np.pi) % (2*np.pi) - np.pi
            dist2 = float(v.T @ invS @ v)
            if dist2 <= self.gate_threshold:
                gated.append(j)
        return gated
    
    def associate(self, tracks, measurements, Hs, Rs):
        """
        Build association events add compute beta_ij probabilities.
        Hs, Rs: lists of H, R for each track
        Returns betas: list of arrays, one per track.
        """
        Ntr = len(tracks)
        Nme = len(measurements)
        # Gating
        gated_indices = [self.gate(tr, measurements, Hs[i], Rs[i])
                         for i, tr in enumerate(tracks)]
        # Generate all joint association hypotheses (valid assingments)
        hyps = [] # list of dict: track->measurement or None
        def recurese(i, assignment):
            if i==Ntr:
                hyps.append(assignment.copy())
                return
            for m in gated_indices[i] + [None]:
                if m is not None and m in assignment.values():
                    continue
                assignment[i] = m
                recurese(i+1, assignment)
        recurese(0, {})
        # Compute likeligood for each hypothesis
        L = np.zeros(len(hyps))
        for h, ass in enumerate(hyps):
            Lh = 1.0
            for i, m in ass.items():
                if m is None:
                    Lh *= (1 - self.Pd)
                else:
                    H, R = Hs[i], Rs[i]
                    z = measurements[m]
                    z_pred = H @ tracks[i].filter.x
                    v = z - z_pred
                    S = H @ tracks[i].filter.P @ H.T + R
                    Lh *= self.Pd * np.exp(-0.5 * v.T @ np.linalg.inv(S) @ v) / np.sqrt((2*np.pi)**len(v) * np.linalg.det(S))
            # clutter term
            Lh *= (self.lambda_c)**(sum(1 for m in ass.values() if m is not None))
            L[h] = Lh
        # Normalize
        Lsum = np.sum(L)
        probs = L / Lsum if Lsum>0 else L
        # Compute beta_ij per track
        betas = []
        for i in range(Ntr):
            beta_i = np.zeros(Nme+1) # last element for miss
            for h, ass in enumerate(hyps):
                m = ass[i]
                idx = m if m is not None else Nme
                beta_i[idx] += probs[h]
            betas.append(beta_i)
        return betas

            