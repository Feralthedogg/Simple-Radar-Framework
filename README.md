# Simple Radar Framework

A Python framework for simulating and managing 3D AESA and PESA radar processing, multi-target tracking using EKF or IMM filters, sensor fusion, track management, beam scheduling, and guidance simulation.

---

## Features

* **AESA Front-End**

  * Configurable element geometry
  * Field-of-view limits (azimuth & elevation)
  * Digital beamforming (MVDR)
  * Pulse compression (matched filtering of LFM waveforms)
  * CFAR detection
  * Optional JPDA / MHT association engines

* **PESA Front-End**

  * Phase-quantized steering weights
  * Subarray beamforming
  * Phase calibration (gain/phase mismatch correction)
  * STAP & pulse compression via AESA core
  * Diversity receive combining
  * EKF-based channel state estimation

* **Tracking Back-End**

  * 3D constant-velocity EKF or IMM
  * Gating, assignment (Hungarian), JPDA/MHT
  * Track management: confirmation, deletion, spawning
  * Optional sensor fusion of confirmed tracks
  * Beam scheduler that prioritizes confirmed tracks

* **Guidance Simulation**

  * Proportional navigation with dynamic navigation constant
  * Pure-pursuit switch in terminal phase
  * Detailed logging of range, NAV constant, lateral acceleration

---

## Installation

```bash
pip install numpy scipy
# (optional) for plotting
pip install matplotlib
```

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

### 2) PESA Pipeline

```python
from radar_framework.radar.PESA import PESPAPipeline, PESAParams
import numpy as np

# 16-element ULA with 0.5 m spacing
M = 16
element_positions = np.stack([
    np.linspace(-(M-1)/2, (M-1)/2, M),
    np.zeros(M),
    np.zeros(M)
], axis=1) * 0.5

params = PESAParams(
    wavelength=0.03,
    element_positions=element_positions,
    delta_phi=np.deg2rad(5),   # phase quant. step
    subarray_size=4,            # elements per subarray
    B=1e6,
    T_p=1e-3,
    P_fa=1e-6
)

pesa = PESPAPipeline(params=params)

# On each scan:
a = ...           # steering vector from AESA core
phases = np.angle(a)
# r: flattened complex returns
r = X.flatten()
out = pesa.process(phases=phases, theta=theta, r=r)

# out['weights'], out['af'], out['pulse_compressed'], out['stap'], out['diversity'], out['ekf']
```

### 3) Tracking Pipeline

```python
from radar_framework.tracking import (
    RadarParams3D, EKF3DParams,
    ManagerParams3D, SchedulerParams3D,
    TrackingPipeline3D
)
import numpy as np

# ... (same as AESA section) ...
```

---

## Example: Missile Engagement Simulation

See `simulation.py` for full AESA & PESA demos:

```bash
python simulation.py
```

Logs and plots will show separate AESA-only and PESA-only engagements with intercept markers.

---

## License

[MIT](LICENSE)

---

## Future Extensions

* Support for mechanically scanned radars

---