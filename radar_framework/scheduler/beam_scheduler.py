import numpy as np
from dataclasses import dataclass

@dataclass
class SchedulerParams3D:
    default_beams: list       # Initial scanning beam angles list

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