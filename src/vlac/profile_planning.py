"""Measured work planning for Powell; finalization remains part of the run."""
import time


class FinalizeProfile(Exception):
    """Return the best completed evaluation and run its validation/export stages."""


class ProfilePlanner:
    def __init__(self, seconds, *, started=None):
        self.seconds = seconds
        self.started = time.monotonic() if started is None else started
        self.seconds_per_sample = 0.

    def observe(self, seconds, samples):
        # The slowest observed grid includes cache misses and setup overhead.
        self.seconds_per_sample = max(self.seconds_per_sample, seconds/max(1, samples))

    def estimate(self, samples):
        return 1.5*self.seconds_per_sample*samples

    def allows(self, samples, validation_samples):
        if self.seconds is None or self.seconds_per_sample == 0:
            return True
        # Reserve a complete fresh midpoint grid, another evaluation's margin,
        # and 15 minutes for report generation, publication and session cleanup.
        reserve = self.estimate(validation_samples+samples)+900.
        return time.monotonic()-self.started+self.estimate(samples)+reserve < self.seconds

    def report(self, samples, validation_samples):
        return {'method': 'bounded_powell_time_planned' if self.seconds is not None else 'bounded_powell',
                'completion_budget_seconds': self.seconds,
                'elapsed_seconds': time.monotonic()-self.started,
                'slowest_seconds_per_sample': self.seconds_per_sample,
                'estimated_next_grid_seconds': self.estimate(samples),
                'reserved_validation_and_export_seconds': self.estimate(validation_samples+samples)+900.,
                'timing_is_estimate': True}
