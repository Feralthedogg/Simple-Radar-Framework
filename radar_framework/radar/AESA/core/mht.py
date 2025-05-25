# radar_framework/radar/AESA/core/mht.py

import numpy as np

from radar_framework.radar.AESA.core.jpda import JPDA

class MHT:
    """
    Multi-Hypothesis Tracker (MHT) engine.
    Manages a tree of hypotheses over time, with pruning.
    """
    def __init__(self, Pd, lambda_c, gate_threshold, max_hypotheses=100):
        self.Pd = Pd
        self.lambda_c = lambda_c
        self.gate_threshold = gate_threshold
        self.max_hypotheses = max_hypotheses
        self.hypothesis_tree = [] # list of dicts: {track_id: measurement_id or None}

    def update(self, tracks, measurements, Hs, Rs):
        # 1) For each existing hypothesis, generate child hypotheses
        new_hyps = []
        for parent in self.hypothesis_tree or [{}]:
            # gating like JPDA
            gated = {i: JPDA(self.gate_threshold, self.Pd, self.lambda_c).gate(tr, measurements, Hs[i], Rs[i])
                     for i, tr in enumerate(tracks)}
            def recurse(i, assignment):
                if i==len(tracks):
                    new_hyps.append({**parent, **assignment})
                    return
                for m in gated[i] + [None]:
                    if m is not None and m in assignment.values(): continue
                    assignment[i] = m
                    recurse(i+1, assignment)
            recurse(0, {})
        # 2) Score and prune
        scored = []
        for hyp in new_hyps:
            score = sum(1 for m in hyp.values() if m is not None)
            scored.append((score, hyp))
        scored.sort(key=lambda x: -x[0])
        self.hypothesis_tree = [h for _, h in scored[:self.max_hypotheses]]
        # 3) Extract best hypothesis for track updates
        best = self.hypothesis_tree[0]
        # apply measurements to tracks accordingly
        for i, m in best.items():
            if m is not None:
                tracks[i].filter.update(measurements[m])
            else:
                tracks[i].record_miss()
        return self.hypothesis_tree
