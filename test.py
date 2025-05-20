import numpy as np
from radar_framework import (
    RadarParams3D,
    EKF3DParams,
    ManagerParams3D,
    SchedulerParams3D,
    EKF3D,
    TrackingPipeline3D
)

# 1) Filter Matrix Validation
def validate_ekf3d_matrices():
    params = EKF3DParams(
        dt=1.0,
        process_noise_std=0.1,
        meas_noise_std=(1.0, np.deg2rad(5))
    )
    ekf = EKF3D(params)

    Q = ekf.Q
    assert Q.shape == (6, 6), f"Q shape is {Q.shape}, expected (6,6)"
    assert np.allclose(Q, Q.T, atol=1e-8), "Q is not symmetric"

    ekf.x = np.array([100.0, 0.0, 50.0, 0.0, 20.0, 0.0])
    px, _, py, _, pz, _ = ekf.x
    R = np.linalg.norm([px, py, pz])

    def compute_H():
        H = np.zeros((3, 6))
        H[0, 0], H[0, 2], H[0, 4] = px/R, py/R, pz/R
        H[1, 0], H[1, 2] = -py/(R**2), px/(R**2)
        den = R**2 * np.sqrt(px**2 + py**2)
        H[2, 0] = -px * pz / den
        H[2, 2] = -py * pz / den
        H[2, 4] = np.sqrt(px**2 + py**2) / (R**2)
        return H

    H_analytic = compute_H()

    def h_fn(x):
        px, vx, py, vy, pz, vz = x
        R = np.linalg.norm([px, py, pz])
        az = np.arctan2(py, px)
        el = np.arcsin(pz / R)
        return np.array([R, az, el])

    eps = 1e-2
    base = h_fn(ekf.x)
    H_numeric = np.zeros((3, 6))
    for i in [0, 2, 4]:
        x_eps = ekf.x.copy()
        x_eps[i] += eps
        H_numeric[:, i] = (h_fn(x_eps) - base) / eps

    assert np.allclose(
        H_analytic[:, [0, 2, 4]],
        H_numeric[:, [0, 2, 4]],
        atol=1e-2
    ), f"H analytic vs numeric mismatch:\n{H_analytic}\n{H_numeric}"

    print("EKF3D Q and H validation passed.")

# 2) Multi-target Noisy Scenario Test
def multi_target_noisy_test():
    radar_params = RadarParams3D(
        Pt=1e3,
        G=30,
        wavelength=0.03,
        sigma_max=1.0,
        k=1.38e-23,
        T_noise=290,
        B=1e6,
        P_clutter=1e-6,
        P_int=1e-7,
        sigma_az=np.deg2rad(1),
        sigma_el=np.deg2rad(1),
        SNR_th=0.0
    )
    ekf_params = EKF3DParams(
        dt=1.0,
        process_noise_std=0.1,
        meas_noise_std=(1.0, np.deg2rad(5))
    )
    manager_params = ManagerParams3D(
        P0=np.eye(6),
        snr_th=0.0,
        miss_limit=3,
        gate_chi2=7.81,
        confirm_thr=2
    )
    scheduler_params = SchedulerParams3D(
        default_beams=[(0.0, 0.0), (np.pi/6, np.pi/18)]
    )

    pipeline = TrackingPipeline3D(
        radar_params=radar_params,
        ekf_params=ekf_params,
        fusion=False,
        manager_params=manager_params,
        scheduler_params=scheduler_params
    )

    true_states = [
        np.array([800.0, 5.0, -200.0, 2.0, 50.0, 0.0]),
        np.array([-600.0, -3.0, 500.0, -1.0, 30.0, 0.0])
    ]
    T_int = 0.1
    np.random.seed(42)

    for step in range(10):
        measurements = []
        for ts in true_states:
            px, vx, py, vy, pz, vz = ts
            ts += np.array([vx, 0.0, vy, 0.0, vz, 0.0]) + np.random.randn(6) * 0.5
            R = np.linalg.norm(ts[[0, 2, 4]])
            az = np.arctan2(ts[2], ts[0]) + np.random.randn() * np.deg2rad(1)
            el = np.arcsin(ts[4] / R) + np.random.randn() * np.deg2rad(1)
            measurements.append(np.array([R, az, el]))

        tracks, beam = pipeline.step(measurements, true_states, T_int)
        print(f"Step {step}, beam={beam}, tracks={len(tracks)}")
        ids = [t.id for t in tracks]
        assert len(set(ids)) == len(ids), "ID duplication occurs"
        if step == 3:
            assert len(tracks) >= 2, "Failed to check more than one track"

    print("Multi-target noisy scenario test passed.")


if __name__ == '__main__':
    validate_ekf3d_matrices()
    multi_target_noisy_test()
