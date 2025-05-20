# Radar Tracking Framework

A Python framework for simulating and managing 3D radar multi-target tracking using EKF or IMM filters, sensor fusion, track management, and beam scheduling.

## Prerequisites
- Python 3.7 +
- `numpy`
- `scipy`
- (Optional) `matplotlib` for plotting results
Install dependencies via pip:
```bash
pip install numpy scipy
```

## Quick Start
1. **Define parameters**
    * Create instances of the parameter dataclasses (e.g. `RadarParams3D`, `EKF3DParams`, etc.) with your desired values.
2. **Initialize pipeline**
```python
from radar_framework import (
    RadarParams3D, EKF3DParams, ManagerParams3D, SchedulerParams3D,
    TrackingPipeline3D
)

# 1) Radar model parameters
radar_params = RadarParams3D(
    Pt=1e3, G=30, wavelength=0.03, sigma_max=1,
    k=1.38e-23, T_noise=290, B=1e6,
    P_clutter=1e-9, P_int=1e-9,
    sigma_az=0.01, sigma_el=0.01,
    SNR_th=10.0
)

# 2) EKF filter parameters
ekf_params = EKF3DParams(
    dt=0.1,
    process_noise_std=0.5,
    meas_noise_std=(1.0, 0.01)
)

# 3) Track manager parameters
import numpy as np
P0 = np.eye(6) * 100.0
manager_params = ManagerParams3D(
    P0=P0, snr_th=10.0,
    miss_limit=5, gate_chi2=7.8,
    confirm_thr=3
)

# 4) Beam scheduler defaults
scheduler_params = SchedulerParams3D(
    default_beams=[(0.0, 0.0), (0.5, 0.2)]
)

# 5) Build tracking pipeline
pipeline = TrackingPipeline3D(
    radar_params=radar_params,
    ekf_params=ekf_params,
    fusion=False,
    manager_params=manager_params,
    scheduler_params=scheduler_params
)
```
3. **Run a tracking step**
Provide a list of simulated measurements and corresponding true states:
```python
measurements = [
    [range1, az1, el1],
    [range2, az2, el2],
]
true_states = [
    np.array([x1, vx1, y1, vy1, z1, vz1]),
    np.array([x2, vx2, y2, vy2, z2, vz2]),
]

T_int = 0.05 # integration time in seconds

tracks, beam_angle = pipeline.step(measurements, true_states, T_int)

print("Current beam:", beam_angle)
for tr in tracks:
    print(f"Track {tr.id}: status={tr.status}, state={tr.filter.x}")
```
4. **Extend or customize**
* Swap in `IMM3DParams` to use an IMM filter instead of EKF.
* Enable `fusion=True` in the pipeline to fuse multiple confirmed tracks.
* Adjust thresholds and noise parameters via the dataclasses.

---

## LICENSE
[LICENSE](LICENSE)

---

For more details, refer to the inline comments in `radar_framework.py`