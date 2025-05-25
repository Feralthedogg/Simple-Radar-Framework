# radar_framework/tracking/core/imm.py

import numpy as np
from dataclasses import dataclass

from radar_framework.tracking.core.filters import EKF3D
from radar_framework.tracking.manager.logger import logger

@dataclass
class IMM3DParams:
    ekf_params_list: list     # List of EKF3DParams for each motion model
    PI: np.ndarray            # Markov transition matrix between models
    mu0: np.ndarray           # Initial mode probabilities

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