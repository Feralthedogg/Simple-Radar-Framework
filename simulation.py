import numpy as np
import matplotlib.pyplot as plt
import logging

from radar_framework.radar.AESA import AESAPipeline, AESAParams
from radar_framework.radar.PESA import PESPAPipeline, PESAParams
from radar_framework.tracking import (
    RadarParams3D, EKF3DParams,
    ManagerParams3D, SchedulerParams3D,
    TrackingPipeline3D
)

# Configure logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# === 1) Pipeline Setup ===
M = 16
element_positions = np.stack([
    np.linspace(-(M-1)/2, (M-1)/2, M),
    np.zeros(M),
    np.zeros(M)
], axis=1) * 0.5

# AESA pipeline
aesa_params = AESAParams(
    wavelength=0.03,
    element_positions=element_positions,
    B=1e6,
    T_p=1e-3,
    P_fa=1e-6
)
aesa_pipeline = AESAPipeline(
    params=aesa_params,
    Pd=0.9,
    clutter_rate=1e-4,
    gate_threshold=7.815,
    use_mht=False
)

# PESA pipeline
pesa_params = PESAParams(
    wavelength=0.03,
    element_positions=element_positions,
    delta_phi=np.deg2rad(5),
    subarray_size=4,
    B=1e6,
    T_p=1e-3,
    P_fa=1e-6
)
pesa_pipeline = PESPAPipeline(params=pesa_params)

# Tracking pipeline params (recreated per sim)
radar_params = RadarParams3D(
    Pt=1e3, G=30.0, wavelength=0.03, sigma_max=1.0,
    k=1.38e-23, T_noise=290.0, B=1e6,
    P_clutter=1e-3, P_int=1e-3,
    sigma_az=0.01, sigma_el=0.01,
    SNR_th=5.0
)
ekf_params = EKF3DParams(
    dt=0.1,
    process_noise_std=5.0,
    meas_noise_std=(100.0, 0.01)
)
manager_params = ManagerParams3D(
    P0=np.eye(6)*1e6,
    snr_th=5.0,
    miss_limit=3,
    gate_chi2=7.815,
    confirm_thr=2
)
scheduler_params = SchedulerParams3D(default_beams=[(0.0,0.0)])

# Engagement & guidance parameters
dt = 0.1
max_time = 2000.0
intercept_radius = 20.0    # 20 m
R_switch = 1000.0
aN_max = 50.0 * 9.81
c = 340.0
V_m = 4.0 * c
vel_t0 = np.array([0.0, 1.2*c, 0.0])

def run_simulation(mode):
    track_pipeline = TrackingPipeline3D(
        radar_params=radar_params,
        ekf_params=ekf_params,
        imm_params=None,
        fusion=False,
        manager_params=manager_params,
        scheduler_params=scheduler_params
    )

    pos_m = np.array([0.0, 0.0, 0.0])
    pos_t = np.array([80000.0, 0.0, 0.0])
    vel_t = vel_t0.copy()

    traj_m, traj_t = [], []
    los_prev = (pos_t - pos_m) / np.linalg.norm(pos_t - pos_m)
    t = 0.0
    launch_delay = 15.0
    omega_start = launch_delay + 2.0
    omega = -0.03
    intercept_point = None

    while t < max_time:
        # 4.1) Scan beam
        beam_theta, beam_el = track_pipeline.scheduler.next_beam()

        # 4.2) Simulate snapshot
        rel = pos_t - pos_m
        R_true = np.linalg.norm(rel)
        a = aesa_pipeline.aesa.steering_vector(beam_theta, beam_el)
        echo_amp = 1.0 / (R_true**2)
        noise = (np.random.normal(size=(M,1)) + 1j*np.random.normal(size=(M,1))) * 1e-3
        X = a[:,None] * echo_amp + noise

        # 4.3) AESA vs PESA processing
        if mode == 'AESA':
            result = aesa_pipeline.process(
                X, beam_theta, beam_el, np.ones(3),
                tracks=track_pipeline.manager.tracks,
                measurements=None, Hs=None, Rs=None
            )
            detections = result['detections']
        else:  # PESA
            pesa_out = pesa_pipeline.process(
                phases=np.angle(a),
                theta=beam_theta,
                r=X.flatten()
            )
            y_pc = pesa_out['pulse_compressed']
            thresh = aesa_pipeline.aesa.cfar_threshold(np.ones(3))
            detections = [aesa_pipeline.aesa.detect(cell, thresh) for cell in y_pc]
            result = pesa_out

        # 4.4) Form measurement if any detection
        if any(detections):
            meas = np.array([
                R_true + np.random.normal(0, ekf_params.meas_noise_std[0]),
                beam_theta + np.random.normal(0, ekf_params.meas_noise_std[1]),
                beam_el    + np.random.normal(0, ekf_params.meas_noise_std[1])
            ])
        else:
            meas = None

        # 4.5) Tracking update
        if meas is not None:
            true_state = np.array([rel[0], vel_t[0], rel[1], vel_t[1], 0.0, 0.0])
            tracks, _ = track_pipeline.step([meas], [true_state], T_int=dt)
        else:
            tracks = track_pipeline.manager.tracks

        # awaiting lock
        if t < launch_delay:
            traj_m.append(pos_m.copy()); traj_t.append(pos_t.copy())
            pos_t += vel_t * dt
            t += dt
            continue

        # 4.6) Guidance Law (PN + Endgame)
        if tracks:
            x = tracks[0].filter.x
            pos_est = np.array([x[0], x[2], 0.0])
            vel_est = np.array([x[1], x[3], 0.0])
        else:
            pos_est, vel_est = pos_t.copy(), vel_t.copy()

        rel_est = pos_est - pos_m
        dist = np.linalg.norm(rel_est)
        closing_v = V_m - np.dot(vel_est, rel_est) / dist
        t_go = dist / closing_v
        intercept_pt_est = pos_est + vel_est * t_go
        dir_lead = (intercept_pt_est - pos_m); dir_lead /= np.linalg.norm(dir_lead)

        los = rel_est / dist
        los_prev_3d = np.array([los_prev[0], los_prev[1], 0.0])
        los_3d = np.array([los[0], los[1], 0.0])
        lambda_dot = np.cross(los_prev_3d, los_3d)[2] / dt
        N_dyn = 4.0 * (1.0 + (R_switch - min(dist, R_switch)) / R_switch * 2.0)
        a_N = np.clip(N_dyn * V_m * lambda_dot, -aN_max, aN_max)
        perp = np.array([-los[1], los[0], 0.0])
        v_raw = V_m * dir_lead + a_N * dt * perp
        v_cmd = v_raw * (V_m / np.linalg.norm(v_raw))
        if dist < R_switch:
            v_cmd = V_m * los_3d

        logging.info(f"t={t:.1f}s | {mode} | dist={dist:.1f} m, aN={a_N:.1f}")

        # 4.7) Update positions
        pos_m += v_cmd * dt
        if t >= omega_start:
            th = omega * dt
            c_t, s_t = np.cos(th), np.sin(th)
            vx, vy = vel_t[0], vel_t[1]
            vel_t = np.array([vx*c_t - vy*s_t, vx*s_t + vy*c_t, 0.0])
        pos_t += vel_t * dt

        traj_m.append(pos_m.copy()); traj_t.append(pos_t.copy())
        los_prev = los.copy()
        t += dt

        # 4.8) Intercept check
        if np.linalg.norm(pos_m - pos_t) <= intercept_radius:
            intercept_point = pos_t.copy()
            logging.info(f"*** {mode} Intercept at t={t:.1f}s ***")
            break

    if intercept_point is not None:
        print(f"{mode}: Target Destroyed")
    else:
        print(f"{mode}: Miss the Target")

    return np.array(traj_m), np.array(traj_t), intercept_point

if __name__ == '__main__':
    traj_m_aesa, traj_t_aesa, ip_aesa = run_simulation('AESA')
    traj_m_pesa, traj_t_pesa, ip_pesa = run_simulation('PESA')

    # === Plot AESA ===
    plt.figure(figsize=(8,8))
    plt.plot(traj_t_aesa[:,0], traj_t_aesa[:,1], 'r-',  label='AESA Target')
    plt.plot(traj_m_aesa[:,0], traj_m_aesa[:,1], 'r--', label='AESA Missile')
    if ip_aesa is not None:
        plt.scatter(ip_aesa[0], ip_aesa[1], c='r', marker='x', s=100, label='Intercept')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.title('Engagement Trajectory – AESA Only')
    plt.legend(); plt.grid(True); plt.axis('equal')

    # === Plot PESA ===
    plt.figure(figsize=(8,8))
    plt.plot(traj_t_pesa[:,0], traj_t_pesa[:,1], 'b-',  label='PESA Target')
    plt.plot(traj_m_pesa[:,0], traj_m_pesa[:,1], 'b--', label='PESA Missile')
    if ip_pesa is not None:
        plt.scatter(ip_pesa[0], ip_pesa[1], c='b', marker='x', s=100, label='Intercept')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.title('Engagement Trajectory – PESA Only')
    plt.legend(); plt.grid(True); plt.axis('equal')

    plt.show()
