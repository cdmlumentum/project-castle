"""Endpoint-preserving stick-breaking coordinates for monotone cubic splines."""
import numpy as np


def controls(unit, start, end):
    """Every adjacent coefficient difference has the sign of end-start.

    Nonnegative B-spline derivative basis functions then guarantee the same
    derivative sign everywhere, including between sampled grid points.
    """
    unit = np.asarray(unit,dtype=float)
    if unit.ndim!=1 or not np.all(np.isfinite(unit)) or np.any((unit<0)|(unit>1)):
        raise ValueError('Monotone coordinates must be finite and in [0,1].')
    values = [float(start)]
    for fraction in unit:
        values.append(values[-1]+float(fraction)*(end-values[-1]))
    return np.r_[values,float(end)]


def linear_coordinates(count):
    return 1/np.arange(count-1,1,-1,dtype=float)


def is_monotone(values):
    values = np.asarray(values)
    return bool(np.all(np.isfinite(values)) and np.all(np.diff(values)*(1 if values[-1]>=values[0] else -1)>=-1e-18))
