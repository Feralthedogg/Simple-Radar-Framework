import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass

from radar_framework.exceptions import RadarFrameworkError
from radar_framework.manager.logger import logger
from radar_framework.manager.track import Track3D

@dataclass
class ManagerParams3D:
    P0: np.ndarray            # Initial state covariance (6x6)
    snr_th: float             # SNR threshold for new track creation
    miss_limit: int           # Max consecutive misses before deletion
    gate_chi2: float          # Chi-square threshold for gating (df=3)
    confirm_thr: int          # Hits required to confirm a track

class TrackManager3D:
    """
    Manages track prediction, gating (chi-square), assignment (Hungarian),
    confirmations, deletions, and initiates new tracks.
    """
    def __init__(self, params: ManagerParams3D, filter_factory):
        # Validate covariance size
        if params.P0.shape != (6,6):
            raise ValueError("ManagerParams3D.P0 must be 6x6.")
        self.P0 = params.P0
        self.snr_th = params.snr_th
        self.miss_limit = params.miss_limit
        self.gate_chi2 = params.gate_chi2
        self.confirm_thr = params.confirm_thr
        self.filter_factory = filter_factory
        self.tracks = []
        self.next_id = 0

    def step(self, measurements, snr_list):
        """
        Run one cycle: predict all tracks, gate & assign measurements,
        update confirmed tracks, delete old, and spawn new.
        """
        if len(measurements) != len(snr_list):
            raise RadarFrameworkError("Mismatch between measurements and SNR list.")
        # 1) Predict existing tracks
        for tr in self.tracks:
            tr.filter.predict()
        N, M = len(self.tracks), len(measurements)
        # 2) Compute cost matrix for assignment if both exist
        if N and M:
            cost = np.full((N, M), np.inf)
            for i, tr in enumerate(self.tracks):
                x = tr.filter.x
                px, py, pz = x[0], x[2], x[4]
                Rpred, az_pred, el_pred = (np.linalg.norm((px,py,pz)),
                                          np.arctan2(py,px),
                                          np.arcsin(pz/np.linalg.norm((px,py,pz))))
                # Compute Jacobian H & innovation for each measurement
                H = np.zeros((3,6)); H[0,[0,2,4]] = [px/Rpred, py/Rpred, pz/Rpred]
                H[1,[0,2]] = [-py/(Rpred**2), px/(Rpred**2)]
                denom = (Rpred**2 * np.sqrt(px**2+py**2))
                H[2,0] = -px*pz/denom; H[2,2] = -py*pz/denom
                H[2,4] = np.sqrt(px**2+py**2)/(Rpred**2)
                S = H @ tr.filter.P @ H.T + tr.filter.R
                invS = np.linalg.inv(S)
                for j, zm in enumerate(measurements):
                    y = zm - np.array([Rpred, az_pred, el_pred])
                    y[1] = (y[1] + np.pi)%(2*np.pi) - np.pi
                    cost[i,j] = float(y @ invS @ y)
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind, col_ind = np.array([], int), np.array([], int)
        assigned_tr, assigned_meas = set(), set()
        # 3) Update assigned tracks
        for i, j in zip(row_ind, col_ind):
            if cost[i,j] <= self.gate_chi2:
                tr = self.tracks[i]
                tr.filter.update(measurements[j])
                tr.record_hit()
                assigned_tr.add(i); assigned_meas.add(j)
        # 4) Handle misses & deletion
        new_tracks = []
        for idx, tr in enumerate(self.tracks):
            if idx not in assigned_tr:
                tr.record_miss()
            # Retain if not both confirmed and over miss limit
            if not (tr.status=='confirmed' and tr.misses>self.miss_limit):
                new_tracks.append(tr)
            else:
                logger.info(f"Deleting track {tr.id} due to misses")
        self.tracks = new_tracks
        # 5) Spawn new tracks from unassigned measurements above SNR threshold
        for j, zm in enumerate(measurements):
            if j not in assigned_meas and snr_list[j] >= self.snr_th:
                filt = self.filter_factory()
                R, az, el = zm
                px = R * np.cos(el) * np.cos(az)
                py = R * np.cos(el) * np.sin(az)
                pz = R * np.sin(el)
                filt.init_state(np.array([px,0,py,0,pz,0]), self.P0)
                tr = Track3D(filt, self.next_id, self.confirm_thr)
                tr.record_hit()
                self.tracks.append(tr)
                self.next_id += 1
        return self.tracks