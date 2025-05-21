class Track3D:
    """Represents one track with hit/miss counters and status."""
    def __init__(self, filt, track_id, confirm_thr):
        self.filter = filt
        self.id = track_id
        self.hits = 0; self.misses = 0
        self.status = 'tentative'
        self.confirm_thr = confirm_thr

    def record_hit(self):
        self.hits += 1
        if self.status == 'tentative' and self.hits >= self.confirm_thr:
            self.status = 'confirmed'

    def record_miss(self):
        self.misses += 1