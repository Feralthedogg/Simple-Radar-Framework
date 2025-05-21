import numpy as np
import matplotlib.pyplot as plt
import logging

from radar_framework import (
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

# 1) Define radar and EKF parameters
radar_params = RadarParams3D(
    Pt=1e3, G=30.0, wavelength=0.03, sigma_max=1.0,
    k=1.38e-23, T_noise=290.0, B=1e6,
    P_clutter=1e-3, P_int=1e-3,
    sigma_az=0.01, sigma_el=0.01,
    SNR_th=5.0
)
ekf_params = EKF3DParams(
    dt=0.1, process_noise_std=5.0,
    meas_noise_std=(100.0, 0.01)
)
manager_params = ManagerParams3D(
    P0=np.eye(6)*1e6, snr_th=5.0,
    miss_limit=3, gate_chi2=7.815,
    confirm_thr=2
)
scheduler_params = SchedulerParams3D(
    default_beams=[(0.0, 0.0)]
)

# Initialize tracking pipeline
pipeline = TrackingPipeline3D(
    radar_params=radar_params,
    ekf_params=ekf_params,
    imm_params=None,
    fusion=False,
    manager_params=manager_params,
    scheduler_params=scheduler_params
)

# Simulation settings
dt = 0.1                     # Time step (s)
max_time = 2000.0            # Max simulation time (s)

# Intercept / endgame parameters
intercept_radius = 10.0      # 최종 명중 기준: 0~10 m 이내
R_switch = 1000.0            # PN → Pure Pursuit 전환 거리 (m)
aN_max = 50.0 * 9.81         # 최대 편향가속도 제한 (m/s^2), 예: 50g

# Speeds
c = 340.0                    # Speed of sound (m/s)
V_m = 4.0 * c                # Missile speed (Mach 4)
vel_t = np.array([0.0, 1.2*c, 0.0])  # Target speed (x, y, z)

# Navigation constant
N = 4.0

# Initial positions
pos_m = np.array([0.0, 0.0, 0.0])        # Missile
pos_t = np.array([80000.0, 0.0, 0.0])    # Target 80 km away

# Trajectory storage
traj_m, traj_t = [], []
los_prev = (pos_t - pos_m) / np.linalg.norm(pos_t - pos_m)

# Launch and maneuver timing
launch_delay = 15.0                     # Missile launch delay (s)
omega_start = launch_delay + 2.0       # Start maneuver after launch + 2s
omega = -0.03                           # Target turn rate (rad/s)

# Variables to mark intercept
intercept_point = None
intercept_time = None

for step in range(int(max_time / dt)):
    t = step * dt

    # Generate noisy measurement
    rel_true = pos_t - pos_m
    R_true = np.linalg.norm(rel_true)
    az_true = np.arctan2(rel_true[1], rel_true[0])
    el_true = 0.0

    meas = np.array([
        R_true + np.random.normal(0, ekf_params.meas_noise_std[0]),
        az_true + np.random.normal(0, ekf_params.meas_noise_std[1]),
        el_true + np.random.normal(0, ekf_params.meas_noise_std[1])
    ])
    true_state = np.array([
        rel_true[0], vel_t[0],
        rel_true[1], vel_t[1],
        0.0,        0.0
    ])

    tracks, beam = pipeline.step([meas], [true_state], T_int=dt)

    # Missile awaiting lock phase
    if t < launch_delay:
        logging.info(f"t={t:.1f}s: awaiting lock until t={launch_delay}s")
        traj_m.append(pos_m.copy())
        traj_t.append(pos_t.copy())
        pos_t += vel_t * dt
        los_prev = rel_true / R_true
        continue

    # Lead pursuit / state estimate
    if tracks:
        x = tracks[0].filter.x
        pos_est = np.array([x[0], x[2], 0.0])
        vel_est = np.array([x[1], x[3], 0.0])
    else:
        pos_est = pos_t.copy()
        vel_est = vel_t.copy()

    rel = pos_est - pos_m
    dist = np.linalg.norm(rel)
    closing_v = V_m - np.dot(vel_est, rel) / dist
    t_go = dist / closing_v
    intercept_pt = pos_est + vel_est * t_go
    dir_lead = (intercept_pt - pos_m)
    dir_lead /= np.linalg.norm(dir_lead)

    # Proportional Navigation with endgame enhancements
    los = rel / dist
    los_prev_3d = np.array([los_prev[0], los_prev[1], 0.0])
    los_3d = np.array([los[0], los[1], 0.0])
    lambda_dot = np.cross(los_prev_3d, los_3d)[2] / dt

    # Dynamic navigation constant: increase as missile closes
    N_dyn = N * (1.0 + (R_switch - min(dist, R_switch)) / R_switch * 2.0)
    a_N_raw = N_dyn * V_m * lambda_dot
    # Saturate to max lateral acceleration
    a_N = np.clip(a_N_raw, -aN_max, aN_max)

    perp = np.array([-los[1], los[0], 0.0])
    v_raw = V_m * dir_lead + a_N * dt * perp
    v_cmd = v_raw * (V_m / np.linalg.norm(v_raw))

    # Switch to Pure Pursuit in endgame zone
    if dist < R_switch:
        v_cmd = V_m * los_3d

    logging.info(f"t={t:.1f}s: dist={dist:.1f} m, N_dyn={N_dyn:.1f}, aN={a_N:.1f}")

    # Update missile position
    pos_m += v_cmd * dt

    # Target maneuver after omega_start
    if t >= omega_start:
        theta = omega * dt
        c_t, s_t = np.cos(theta), np.sin(theta)
        # vel_t is 3-element [vx, vy, vz]
        vx, vy, _ = vel_t
        vel_t[0] = vx * c_t - vy * s_t
        vel_t[1] = vx * s_t + vy * c_t
        # vel_t[2] stays zero

    # Update target position
    pos_t += vel_t * dt

    traj_m.append(pos_m.copy())
    traj_t.append(pos_t.copy())
    los_prev = los.copy()

    # Optionally tighten control loop in endgame
    if dist < R_switch and dt > 0.01:
        dt = 0.01

    # Check intercept (0~10 m 이내)
    if np.linalg.norm(pos_m - pos_t) <= intercept_radius:
        intercept_point = pos_t.copy()
        intercept_time = t
        logging.info(f"*** Intercept at t={t:.1f}s ***")
        print("Target Destroyed")
        break
else:
    logging.info("Missed the target.")
    print("Miss the Target")

# Convert to arrays for plotting
traj_m = np.array(traj_m)
traj_t = np.array(traj_t)

# Plot trajectories
plt.figure(figsize=(8, 8))
plt.plot(traj_t[:, 0], traj_t[:, 1], 'r-', label='Target')
plt.plot(traj_m[:, 0], traj_m[:, 1], 'b--', label='Missile')

# Plot intercept splash if occurred
if intercept_point is not None:
    plt.scatter(
        intercept_point[0], intercept_point[1],
        c='k', s=50, marker='x', label='Intercept'
    )

plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title('Missile vs Target Trajectory')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
