# radar_framework/tracking/pipeline.py

from radar_framework.tracking.core.models import RadarParams3D, RadarModel3D
from radar_framework.tracking.core.filters import EKF3DParams, EKF3D
from radar_framework.tracking.core.imm import IMM3DParams, IMM3D
from radar_framework.tracking.core.sensor_fusion import SensorFusion
from radar_framework.tracking.manager.track_manager import ManagerParams3D, TrackManager3D
from radar_framework.tracking.scheduler.beam_scheduler import SchedulerParams3D, BeamScheduler3D
from radar_framework.tracking.exceptions import RadarFrameworkError
from radar_framework.tracking.manager.logger import logger

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