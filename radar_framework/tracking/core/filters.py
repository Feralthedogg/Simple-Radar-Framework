# radar_framework/tracking/core/filters.py

import numpy as np
from dataclasses import dataclass

from radar_framework.tracking.exceptions import RadarFrameworkError

@dataclass
class EKF3DParams:
    dt: float                 # Time step between updates (s)
    process_noise_std: float  # Std of process noise (m/s^2)
    meas_noise_std: tuple     # Measurement noise stds (range, angle)

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