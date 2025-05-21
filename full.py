# =============================================================================
# TODO:
#   - Add unit tests for each component
#   - Profile and optimize cost matrix computation
#   - Integrate dynamic clutter and terrain masking models
#   - Validate gating thresholds for non-Gaussian noise
#   - Consider Cython for heavy numpy loops if profiling indicates
# =============================================================================

import logging
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import linear_sum_assignment

# Configure root logger for the entire framework
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Custom Exceptions
# =============================================================================

class RadarFrameworkError(Exception):
    """Base exception for all radar framework errors."""
    pass

# =============================================================================
# 0) Parameter Dataclasses
#
# Encapsulate all configuration parameters to ensure consistency and
# ease of experimentation via simple object creation or config files.
# =============================================================================
@dataclass
class RadarParams3D:
    Pt: float                 # Transmit power (W)
    G: float                  # Antenna gain (linear)
    wavelength: float         # Operating wavelength (m)
    sigma_max: float          # Target radar cross-section (m^2)
    k: float                  # Boltzmann's constant (J/K)
    T_noise: float            # System noise temperature (K)
    B: float                  # Receiver bandwidth (Hz)
    P_clutter: float          # Clutter power level
    P_int: float              # Interference power level
    sigma_az: float           # Pointing error std in azimuth (rad)
    sigma_el: float           # Pointing error std in elevation (rad)
    SNR_th: float             # Minimum SNR to initiate a new track

@dataclass
class EKF3DParams:
    dt: float                 # Time step between updates (s)
    process_noise_std: float  # Std of process noise (m/s^2)
    meas_noise_std: tuple     # Measurement noise stds (range, angle)

@dataclass
class IMM3DParams:
    ekf_params_list: list     # List of EKF3DParams for each motion model
    PI: np.ndarray            # Markov transition matrix between models
    mu0: np.ndarray           # Initial mode probabilities

@dataclass
class ManagerParams3D:
    P0: np.ndarray            # Initial state covariance (6x6)
    snr_th: float             # SNR threshold for new track creation
    miss_limit: int           # Max consecutive misses before deletion
    gate_chi2: float          # Chi-square threshold for gating (df=3)
    confirm_thr: int          # Hits required to confirm a track

@dataclass
class SchedulerParams3D:
    default_beams: list       # Initial scanning beam angles list

# =============================================================================
# 1) Radar Measurement & SNR Model
# =============================================================================
class RadarModel3D:
    """
    Radar propagation and SNR computation based on the radar range equation.
    Includes pointing loss model in az/el.
    """
    def __init__(self, params: RadarParams3D):
        # Ensure all parameters are valid
        for field, val in asdict(params).items():
            if val is None:
                raise ValueError(f"RadarParams3D.{field} cannot be None.")
        self._p = params

    def _orientation(self, az_t, el_t, az_b, el_b):
        """
        Gaussian antenna pointing loss based on deviation between target
        angles (az_t, el_t) and beam pointing (az_b, el_b).
        """
        da, de = az_t - az_b, el_t - el_b
        return np.exp(-0.5*((da/self._p.sigma_az)**2 + (de/self._p.sigma_el)**2))

    def instantaneous_snr(self, state, beam_angles, S_t=1.0):
        """
        Compute instantaneous SNR for a target in state [px, vx, py, vy, pz, vz].
        Applies the radar equation and noise/clutter/interference model.
        """
        px, vx, py, vy, pz, vz = state
        R = np.linalg.norm((px, py, pz))
        if R <= 0:
            raise ValueError("Invalid range; target too close or at origin.")
        # Angles of target
        az_t = np.arctan2(py, px)
        el_t = np.arcsin(pz/R)
        # Beam pointing angles
        az_b, el_b = beam_angles
        g = self._orientation(az_t, el_t, az_b, el_b)
        # Received power via radar equation
        Pr = (self._p.Pt * self._p.G**2 * self._p.wavelength**2 * self._p.sigma_max * g * S_t)
        Pr /= ((4*np.pi)**3 * R**4)
        # Total noise power
        Pn = self._p.k * self._p.T_noise * self._p.B + self._p.P_clutter + self._p.P_int
        return Pr / Pn

    def integrated_snr(self, state, beam_angles, T_int, S_t=1.0):
        """
        Compute coherent integration SNR over time T_int.
        Includes Doppler decorrelation loss for moving targets.
        """
        if T_int <= 0:
            raise ValueError("Integration time must be positive.")
        snr0 = self.instantaneous_snr(state, beam_angles, S_t)
        I = np.sqrt(T_int * self._p.B)  # Integration gain
        # Doppler velocity projection
        px, vx, py, vy, pz, vz = state
        R = np.linalg.norm((px, py, pz))
        vr = (px*vx + py*vy + pz*vz)/R
        D = np.exp(- (np.pi * 2 * vr * T_int / self._p.wavelength)**2)
        return snr0 * I * D

# =============================================================================
# 2) Extended Kalman Filter (EKF3D)
# =============================================================================
class EKF3D:
    """
    Implements a constant-velocity EKF in Cartesian space with spherical
    range/azimuth/elevation measurements.
    """
    def __init__(self, params: EKF3DParams):
        if params.dt <= 0:
            raise ValueError("EKF3D dt must be positive.")
        self.dt = params.dt
        # State transition matrix F for [pos; vel]
        F_top = np.hstack([np.eye(3), self.dt*np.eye(3)])
        F_bot = np.hstack([np.zeros((3,3)), np.eye(3)])
        self.F = np.vstack([F_top, F_bot])
        # Process noise covariance Q (constant acceleration model)
        q = params.process_noise_std**2
        dt, dt2, dt3, dt4 = self.dt, self.dt**2, self.dt**3/2, self.dt**4/4
        Q1 = np.array([[dt4, dt3],[dt3, dt2]])
        self.Q = q * np.block([
            [Q1, np.zeros((2,2)), np.zeros((2,2))],
            [np.zeros((2,2)), Q1, np.zeros((2,2))],
            [np.zeros((2,2)), np.zeros((2,2)), Q1]
        ])
        # Measurement noise
        r_std, ang_std = params.meas_noise_std
        self.R = np.diag([r_std**2, ang_std**2, ang_std**2])
        # Initialize filter state and covariance
        self.x = np.zeros(6)
        self.P = np.eye(6)

    def init_state(self, x0, P0):
        """Initialize filter with state x0 (6,) and covariance P0 (6x6)."""
        x0_arr, P0_arr = np.asarray(x0), np.asarray(P0)
        if x0_arr.shape!=(6,) or P0_arr.shape!=(6,6):
            raise ValueError("Incorrect init shapes for EKF3D.")
        self.x, self.P = x0_arr.copy(), P0_arr.copy()

    def predict(self):
        """Propagate state and covariance forward one time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas):
        """
        Update step with spherical measurement z = [range, az, el].
        Accounts for angle wrapping in azimuth.
        """
        z = np.asarray(meas)
        if z.shape != (3,):
            raise ValueError("Measurement must be 3-vector.")
        # Predicted measurement
        px, vx, py, vy, pz, vz = self.x
        R_pred = np.linalg.norm((px, py, pz))
        az_pred = np.arctan2(py, px)
        el_pred = np.arcsin(pz / R_pred)
        z_pred = np.array([R_pred, az_pred, el_pred])
        # Build measurement Jacobian H
        H = np.zeros((3,6))
        rho2 = R_pred**2
        H[0,[0,2,4]] = [px/R_pred, py/R_pred, pz/R_pred]
        H[1,[0,2]] = [-py/rho2, px/rho2]
        denom = rho2 * np.sqrt(px**2+py**2)
        H[2,0] = -px*pz/denom; H[2,2] = -py*pz/denom
        H[2,4] = np.sqrt(px**2+py**2)/rho2
        # Innovation with angle normalization
        y = z - z_pred
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        # State & covariance update
        self.x += K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

# =============================================================================
# 3) Interacting Multiple Model Filter (IMM3D)
# =============================================================================
class IMM3D:
    """
    Combines multiple EKF3D models with different dynamics.
    Performs mixing, mode-conditional prediction & update, and
    fuses mode probabilities.
    """
    def __init__(self, params: IMM3DParams):
        # Create one EKF per model
        self.models = [EKF3D(p) for p in params.ekf_params_list]
        self.PI = params.PI
        self.mu = params.mu0.copy()
        if self.PI.shape[0] != self.PI.shape[1] or len(self.mu) != self.PI.shape[0]:
            raise ValueError("IMM3D PI and mu0 dimension mismatch.")
        self.M = len(self.models)

    def predict(self):
        """IMixed prediction: mix states, then predict each model."""
        # Compute mixing probabilities
        c = self.PI.T @ self.mu             # normalization terms
        mu_ij = (self.PI * self.mu[:,None]) / c[None,:]
        mix_x, mix_P = [], []
        # Mix initial conditions for each model
        for j in range(self.M):
            xj = sum(mu_ij[i,j] * self.models[i].x for i in range(self.M))
            Pj = sum(mu_ij[i,j] * (self.models[i].P + np.outer(self.models[i].x - xj,
                                                              self.models[i].x - xj))
                     for i in range(self.M))
            mix_x.append(xj); mix_P.append(Pj)
        # Set mixed states & predict
        for j, m in enumerate(self.models):
            m.x, m.P = mix_x[j], mix_P[j]
            m.predict()
        # Update global mode probabilities
        self.mu = c

    def update(self, z):
        """Mode-conditional update: compute likelihoods, update each EKF, and fuse mu."""
        logL = np.zeros(self.M)
        # Compute likelihood per model
        for j, m in enumerate(self.models):
            px, _, py, _, pz, _ = m.x
            R_pred = np.linalg.norm((px, py, pz))
            az_pred = np.arctan2(py, px)
            el_pred = np.arcsin(pz / R_pred)
            z_pred = np.array([R_pred, az_pred, el_pred])
            # Innovation & Jacobian reuse from EKF
            y = z - z_pred
            y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi
            # Recompute H & S for likelihood
            H = np.zeros((3,6))
            rho2 = R_pred**2
            H[0,[0,2,4]] = [px/R_pred, py/R_pred, pz/R_pred]
            H[1,[0,2]] = [-py/rho2, px/rho2]
            denom = rho2 * np.sqrt(px**2+py**2)
            H[2,0] = -px*pz/denom; H[2,2] = -py*pz/denom
            H[2,4] = np.sqrt(px**2+py**2)/rho2
            S = H @ m.P @ H.T + m.R
            try:
                invS = np.linalg.inv(S)
                _, logdet = np.linalg.slogdet(2*np.pi * S)
                logL[j] = -0.5*(y @ invS @ y) - 0.5 * logdet
            except np.linalg.LinAlgError:
                logger.warning("IMM3D: singular S, assigning -inf likelihood.")
                logL[j] = -np.inf
            # Update model's EKF
            m.update(z)
        # Fuse mode probabilities
        maxL = np.max(logL)
        L = np.exp(logL - maxL)
        self.mu = (self.mu * L) / np.sum(self.mu * L)

    def estimate(self):
        """Compute combined state estimate and covariance over all modes."""
        x = sum(self.mu[j] * self.models[j].x for j in range(self.M))
        P = sum(self.mu[j] * (self.models[j].P + np.outer(self.models[j].x - x,
                                                            self.models[j].x - x))
                for j in range(self.M))
        return x, P

# =============================================================================
# 4) Sensor Fusion Utility
# =============================================================================
class SensorFusion:
    @staticmethod
    def fuse(states, covariances):
        """
        Fuse N state estimates with covariances via optimal linear fusion.
        P_f = inv(sum(inv(P_i)))
        x_f = P_f * sum(inv(P_i) * x_i)
        """
        inv_sum = sum(np.linalg.inv(P) for P in covariances)
        P_f = np.linalg.inv(inv_sum)
        x_f = P_f @ sum(np.linalg.inv(P) @ x for x, P in zip(states, covariances))
        return x_f, P_f

# =============================================================================
# 5) Track Lifecycle & Gating (TrackManager3D)
# =============================================================================
class Track3D:
    """Represents one track with hit/miss counters and status."""
    def __init__(self, filt, track_id, confirm_thr):
        self.filter = filt
        self.id = track_id
        self.hits = 0; self.misses = 0
        self.status = 'tentative'
        self.confirm_thr = confirm_thr

    def record_hit(self):
        self.hits += 1
        if self.status == 'tentative' and self.hits >= self.confirm_thr:
            self.status = 'confirmed'

    def record_miss(self):
        self.misses += 1

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

# =============================================================================
# 6) Beam Scheduler (3D)
# =============================================================================
class BeamScheduler3D:
    """
    Manages scanning beams: defaults cycle through list, or prioritizes
    confirmed tracks with highest hit counts.
    """
    def __init__(self, params: SchedulerParams3D):
        if not params.default_beams:
            raise ValueError("Need at least one default beam.")
        self.angles = list(params.default_beams)
        self.idx = 0

    def update(self, tracks):
        """
        After a step, prioritize beams pointing to confirmed tracks
        sorted by hit count (descending).
        """
        confirmed = [t for t in tracks if t.status=='confirmed']
        confirmed.sort(key=lambda t: t.hits, reverse=True)
        if confirmed:
            # Compute beam angles to each confirmed track
            new_angles = []
            for t in confirmed:
                x = t.filter.x
                az = np.arctan2(x[2], x[0])
                el = np.arcsin(x[4] / np.linalg.norm(x[[0,2,4]]))
                new_angles.append((az, el))
            self.angles = new_angles
        # Reset index modulo new list length
        self.idx %= len(self.angles)

    def next_beam(self):
        """Return next beam angle to scan."""
        angle = self.angles[self.idx]
        self.idx = (self.idx + 1) % len(self.angles)
        return angle

# =============================================================================
# 7) Complete Tracking Pipeline
# =============================================================================
class TrackingPipeline3D:
    """
    Orchestrates: radar model, filter (EKF or IMM), track manager,
    beam scheduler, optional sensor fusion, in one step call.
    """
    def __init__(
        self,
        radar_params: RadarParams3D,
        ekf_params: EKF3DParams=None,
        imm_params: IMM3DParams=None,
        fusion: bool=False,
        manager_params: ManagerParams3D=None,
        scheduler_params: SchedulerParams3D=None
    ):
        if manager_params is None or scheduler_params is None:
            raise ValueError("Need both ManagerParams3D and SchedulerParams3D.")
        self.radar = RadarModel3D(radar_params)
        # Choose filter factory: IMM if provided else EKF
        self.filter_factory = (
            lambda: IMM3D(imm_params) if imm_params
            else EKF3D(ekf_params)
        )
        self.manager = TrackManager3D(manager_params, self.filter_factory)
        self.scheduler = BeamScheduler3D(scheduler_params)
        self.fusion = fusion

    def step(self, measurements, true_states, T_int):
        """
        Perform one pipeline cycle:
          1) Select next beam
          2) Compute integrated SNRs
          3) Manage tracks (predict, update, spawn)
          4) Optional fusion on confirmed
          5) Update scheduler priorities
        Returns (tracks, beam_angles).
        """
        if not isinstance(measurements, list) or not isinstance(true_states, list):
            raise RadarFrameworkError("Inputs must be lists.")
        if T_int <= 0:
            raise RadarFrameworkError("Integration time must be positive.")
        # 1) Beam selection
        beam = self.scheduler.next_beam()
        # 2) Compute SNRs for all true targets
        snrs = [self.radar.integrated_snr(s, beam, T_int) for s in true_states]
        # 3) Track management: gating, updating, spawning
        tracks = self.manager.step(measurements, snrs)
        # 4) Optional multi-track sensor fusion
        if self.fusion:
            confirmed = [t for t in tracks if t.status=='confirmed']
            if len(confirmed) > 1:
                states = [t.filter.x for t in confirmed]
                covs = [t.filter.P for t in confirmed]
                xf, Pf = SensorFusion.fuse(states, covs)
                logger.info(f"Fused state: {xf}")
        # 5) Scheduler reprioritization
        self.scheduler.update(tracks)
        return tracks, beam
