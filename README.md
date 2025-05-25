# Simple Radar Framework

A Python framework for simulating and managing 3D AESA radar processing and multi-target tracking using EKF or IMM filters, sensor fusion, track management, and beam scheduling.

---

## Features

- **AESA Front-End**  
  - Configurable element geometry  
  - Field-of-view limits (azimuth & elevation)  
  - Digital beamforming (MVDR)  
  - Pulse compression (matched filtering of LFM waveforms)  
  - CFAR detection  
  - Optional JPDA / MHT association engines  

- **Tracking Back-End**  
  - 3D constant-velocity EKF or IMM  
  - Gating, assignment (Hungarian), JPDA/MHT  
  - Track management: confirmation, deletion, spawning  
  - Optional sensor fusion of confirmed tracks  
  - Beam scheduler that prioritizes confirmed tracks  

- **Guidance Simulation**  
  - Proportional navigation with dynamic navigation constant  
  - Pure-pursuit switch in terminal phase  
  - Detailed logging of range, NAV constant, lateral acceleration  

---

## Installation

```bash
pip install numpy scipy
# (optional) for plotting
pip install matplotlib
````

---

## Quick Start

### 1) AESA Pipeline

```python
from radar_framework.radar.AESA import AESAPipeline, AESAParams
import numpy as np

# 16-element ULA with 0.5 m spacing
M = 16
element_positions = np.stack([
    np.linspace(-(M-1)/2, (M-1)/2, M),
    np.zeros(M),
    np.zeros(M)
], axis=1) * 0.5

params = AESAParams(
    wavelength=0.03,
    element_positions=element_positions,
    B=1e6,
    T_p=1e-3,
    P_fa=1e-6,
    az_limits=(-np.pi/3, np.pi/3),   # ±60° azimuth
    el_limits=(-np.pi/6, np.pi/6)    # ±30° elevation
)

aesa = AESAPipeline(
    params=params,
    Pd=0.9,
    clutter_rate=1e-4,
    gate_threshold=7.815,
    use_mht=False
)

# On each scan:
X = ...          # (M×1) complex snapshot for current beam
theta, el = ...  # scan angles
window = np.ones(3)
out = aesa.process(X, theta, el, window)

# out['detections'] → list of booleans
# out['beamformed'], out['pulse_compressed'], out['association']
```

### 2) Tracking Pipeline

```python
from radar_framework.tracking import (
    RadarParams3D, EKF3DParams,
    ManagerParams3D, SchedulerParams3D,
    TrackingPipeline3D
)
import numpy as np

# Radar model parameters
radar_params = RadarParams3D(
    Pt=1e3, G=30, wavelength=0.03, sigma_max=1,
    k=1.38e-23, T_noise=290, B=1e6,
    P_clutter=1e-3, P_int=1e-3,
    sigma_az=0.01, sigma_el=0.01,
    SNR_th=5.0
)

# EKF parameters
ekf_params = EKF3DParams(
    dt=0.1,
    process_noise_std=5.0,
    meas_noise_std=(100.0, 0.01)
)

# Track manager parameters
P0 = np.eye(6) * 1e6
manager_params = ManagerParams3D(
    P0=P0, snr_th=5.0,
    miss_limit=3, gate_chi2=7.815,
    confirm_thr=2
)

# Beam scheduler defaults
scheduler_params = SchedulerParams3D(
    default_beams=[(0.0, 0.0), (0.4, 0.1)]
)

pipeline = TrackingPipeline3D(
    radar_params=radar_params,
    ekf_params=ekf_params,
    fusion=False,
    manager_params=manager_params,
    scheduler_params=scheduler_params
)

# Run one step
measurements = [ [r1, az1, el1], … ]
true_states  = [ np.array([x,vx,y,vy,z,vz]), … ]
tracks, beam_angle = pipeline.step(measurements, true_states, T_int=0.1)

print(f"Beam → az: {beam_angle[0]:.2f}, el: {beam_angle[1]:.2f}")
for tr in tracks:
    print(f"Track {tr.id}: status={tr.status}, state={tr.filter.x}")
```

---

## Example: Missile Engagement Simulation

See `simulation.py` for a full end-to-end demo:

* AESA scanning within ±60°/±30° FOV
* CFAR to generate measurements
* EKF tracking, beam scheduling
* Proportional navigation guidance
* Detailed `logging.info(...)` of lock, range, NAV constant, lateral accel

Run:

```bash
python simulation.py
```

You’ll see log lines like:

```
2025-05-25 22:25:45,588 [INFO] root: t=15.0s: awaiting lock until t=15.0s
2025-05-25 22:25:45,589 [INFO] root: t=15.1s: dist=80236.9 m, N_dyn=4.0, aN=27.6
2025-05-25 22:25:45,589 [INFO] root: t=15.2s: dist=80109.8 m, N_dyn=4.0, aN=1.0
...
2025-05-25 22:25:45,944 [INFO] root: t=92.4s: dist=17.2 m, N_dyn=11.9, aN=-490.5
2025-05-25 22:25:45,944 [INFO] root: *** Intercept at t=92.4s ***
Target Destroyed
```

And a matplotlib plot of the missile & target trajectories.

---

## Extensibility

* Swap in `IMM3DParams` / IMM3D for maneuvering targets
* Turn on `fusion=True` to fuse confirmed tracks
* Tweak FOV, thresholds, noise specs via dataclasses

---

## License

[MIT](LICENSE)

---

## Future Extensions

- **PESA Radar** support (Passive Electronically Scanned Array)  
- **Mechanically Scanned** (conventional) radar processing  