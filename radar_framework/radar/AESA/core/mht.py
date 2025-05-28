# radar_framework/radar/AESA/core/mht.py

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
from .jpda import JPDA

class MHT(AssociationEngine):
    """
    Multi-Hypothesis Tracker engine.
    Manages a tree of association hypotheses with pruning.
    """
    def __init__(self, gate_threshold: float, Pd: float, lambda_c: float, max_hypotheses: int = 100):
        super().__init__(gate_threshold, Pd, lambda_c)
        self.max_hypotheses = max_hypotheses
        self.hypothesis_tree = []

    def gate(self, track, measurements: np.ndarray, H: np.ndarray, R: np.ndarray) -> list:
        """
        Reuse JPDA gate method for hypothesis generation.
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
            return JPDA(self.gate_threshold, self.Pd, self.lambda_c).gate(track, measurements, H, R)
        except exceptions.AESAError:
            # propagate JPDA-specific gating errors
            raise
        except Exception as e:
            raise exceptions.AESAError(f"MHT gate error: {e}")

    def associate(self, tracks: list, measurements: list, Hs: list, Rs: list) -> list:
        """
        Alias for update to conform to AssociationEngine API.
        """
        # Input validation
        assert_list(tracks, 'tracks')
        assert_list(measurements, 'measurements')
        assert_list(Hs, 'Hs')
        assert_list(Rs, 'Rs')
        n = len(tracks)
        assert_list_length(Hs, 'Hs', n)
        assert_list_length(Rs, 'Rs', n)
        # Validate each H and R
        assert_list_of_ndarray(Hs, 'Hs', 2)
        for i in range(n):
            assert_square_matrix(Rs[i], f'Rs[{i}]')

        try:
            return self.update(tracks, measurements, Hs, Rs)
        except exceptions.AESAError:
            raise
        except Exception as e:
            raise exceptions.AESAError(f"MHT associate error: {e}")

    def update(self, tracks: list, measurements: list, Hs: list, Rs: list) -> list:
        """
        Generate and prune hypotheses, then apply the best assignment.
        Returns the updated hypothesis tree.
        """
        # Input validation
        assert_list(tracks, 'tracks')
        assert_list(measurements, 'measurements')
        assert_list(Hs, 'Hs')
        assert_list(Rs, 'Rs')
        n = len(tracks)
        assert_list_length(Hs, 'Hs', n)
        assert_list_length(Rs, 'Rs', n)
        assert_list_of_ndarray(Hs, 'Hs', 2)
        for i in range(n):
            assert_square_matrix(Rs[i], f'Rs[{i}]')

        try:
            # 1) Expand hypotheses from existing tree
            new_hyps = []
            base_hyps = self.hypothesis_tree or [{}]
            for parent in base_hyps:
                gated = {i: self.gate(tr, measurements, Hs[i], Rs[i])
                         for i, tr in enumerate(tracks)}
                def recurse(i, assignment):
                    if i == n:
                        new_hyps.append({**parent, **assignment})
                        return
                    for m in gated[i] + [None]:
                        if m is not None and m in assignment.values():
                            continue
                        assignment[i] = m
                        recurse(i + 1, assignment)
                recurse(0, {})

            # 2) Score and prune: sort by number of assigned measurements desc
            scored = sorted(
                new_hyps,
                key=lambda hyp: -sum(1 for v in hyp.values() if v is not None)
            )
            self.hypothesis_tree = scored[:self.max_hypotheses]
            if not self.hypothesis_tree:
                return []

            # 3) Apply best hypothesis to tracks
            best = self.hypothesis_tree[0]
            for i, m in best.items():
                if m is not None:
                    tracks[i].filter.update(measurements[m])
                else:
                    tracks[i].record_miss()
            return self.hypothesis_tree
        except exceptions.AESAError:
            raise
        except Exception as e:
            raise exceptions.AESAError(f"MHT update error: {e}")
