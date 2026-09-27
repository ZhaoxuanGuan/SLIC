from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
from .config import SLICConfig
from .geometry import auxiliary_plane, wrap_360, wrap_180


def _array(values: Iterable, *, dtype=np.float64) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(list(values), dtype=dtype))


@dataclass(frozen=True)
class FocalMechanismCatalog:

    s1: np.ndarray
    d1: np.ndarray
    r1: np.ndarray
    s2: np.ndarray
    d2: np.ndarray
    r2: np.ndarray
    uncertainty_deg: np.ndarray
    weights: np.ndarray
    event_id: np.ndarray

    @property
    def size(self) -> int:
        return int(self.s1.size)

    @classmethod
    def from_single_planes(cls, strike: Iterable, dip: Iterable, rake: Iterable, *, uncertainty_deg: Iterable | float | None=None, weights: Iterable | float | None=None, event_id: Iterable | None=None) -> 'FocalMechanismCatalog':
        s1 = _array(strike)
        d1 = _array(dip)
        r1 = _array(rake)
        if not s1.size == d1.size == r1.size:
            raise ValueError('strike, dip, and rake must have the same length.')
        if not s1.size:
            raise ValueError('The focal-mechanism catalogue cannot be empty.')
        conjugates = np.asarray([auxiliary_plane(float(s), float(d), float(r)) for s, d, r in zip(s1, d1, r1)], dtype=np.float64)
        return cls.from_nodal_planes(s1, d1, r1, conjugates[:, 0], conjugates[:, 1], conjugates[:, 2], uncertainty_deg=uncertainty_deg, weights=weights, event_id=event_id)

    @classmethod
    def from_nodal_planes(cls, s1: Iterable, d1: Iterable, r1: Iterable, s2: Iterable, d2: Iterable, r2: Iterable, *, uncertainty_deg: Iterable | float | None=None, weights: Iterable | float | None=None, event_id: Iterable | None=None) -> 'FocalMechanismCatalog':
        arrays = [_array(values) for values in (s1, d1, r1, s2, d2, r2)]
        sizes = {array.size for array in arrays}
        if len(sizes) != 1 or not arrays[0].size:
            raise ValueError('All nodal-plane columns must have one equal, nonzero length.')
        n_events = int(arrays[0].size)

        def expand(value, default, dtype=np.float64):
            if value is None:
                return np.full(n_events, default, dtype=dtype)
            if np.isscalar(value):
                return np.full(n_events, value, dtype=dtype)
            result = np.asarray(list(value), dtype=dtype)
            if result.size != n_events:
                raise ValueError('Optional catalogue columns must match event count.')
            return np.ascontiguousarray(result)
        uncertainty = expand(uncertainty_deg, np.nan)
        event_weights = expand(weights, 1.0)
        ids = np.arange(1, n_events + 1, dtype=np.int64).astype(object) if event_id is None else np.asarray(list(event_id), dtype=object)
        if ids.size != n_events:
            raise ValueError('event_id must match event count.')
        catalog = cls(s1=np.ascontiguousarray(wrap_360(arrays[0])), d1=arrays[1], r1=np.ascontiguousarray(wrap_180(arrays[2])), s2=np.ascontiguousarray(wrap_360(arrays[3])), d2=arrays[4], r2=np.ascontiguousarray(wrap_180(arrays[5])), uncertainty_deg=uncertainty, weights=event_weights, event_id=np.ascontiguousarray(ids))
        catalog.validate()
        return catalog

    def validate(self) -> None:
        if self.size < 1:
            raise ValueError('The focal-mechanism catalogue cannot be empty.')
        data = np.column_stack((self.s1, self.d1, self.r1, self.s2, self.d2, self.r2))
        if not np.all(np.isfinite(data)):
            raise ValueError('Nodal-plane angles must be finite.')
        if np.any((self.d1 < 0.0) | (self.d1 > 90.0)):
            raise ValueError('Plane-1 dips must lie within [0, 90] degrees.')
        if np.any((self.d2 < 0.0) | (self.d2 > 90.0)):
            raise ValueError('Plane-2 dips must lie within [0, 90] degrees.')
        if np.any(~np.isfinite(self.weights)) or np.any(self.weights <= 0.0):
            raise ValueError('Event weights must be finite and positive.')

    def subset(self, indices: np.ndarray) -> 'FocalMechanismCatalog':
        indices = np.asarray(indices, dtype=np.int64)
        return FocalMechanismCatalog(**{name: np.ascontiguousarray(getattr(self, name)[indices]) for name in ('s1', 'd1', 'r1', 's2', 'd2', 'r2', 'uncertainty_deg', 'weights', 'event_id')})


@dataclass(frozen=True)
class PreparedCatalog:
    n1: np.ndarray
    n2: np.ndarray
    strike1: np.ndarray
    dip1: np.ndarray
    strike2: np.ndarray
    dip2: np.ndarray
    cos_rake1: np.ndarray
    sin_rake1: np.ndarray
    cos_rake2: np.ndarray
    sin_rake2: np.ndarray
    weights: np.ndarray
    weight_sum: float
    eq6: np.ndarray
    uncertainty_deg: np.ndarray
    event_id: np.ndarray


def prepare_catalog(catalog: FocalMechanismCatalog, config: SLICConfig) -> PreparedCatalog:
    config.validate()
    catalog.validate()
    eq6 = np.round(np.column_stack((catalog.s1, catalog.d1, catalog.r1, catalog.s2, catalog.d2, catalog.r2)), int(config.rounding_decimals)).astype(np.float64)
    uncertainty = np.asarray(catalog.uncertainty_deg, dtype=np.float64).copy()
    uncertainty[~np.isfinite(uncertainty)] = config.default_mechanism_uncertainty_deg
    uncertainty = np.clip(uncertainty, config.minimum_mechanism_uncertainty_deg, config.maximum_mechanism_uncertainty_deg)
    uncertainty = np.round(uncertainty, int(config.rounding_decimals))

    def plane_arrays(strike_deg, dip_deg, rake_deg):
        strike, dip, rake = np.deg2rad(np.column_stack((strike_deg, dip_deg, rake_deg))).T
        cs, ss = (np.cos(strike), np.sin(strike))
        cd, sd = (np.cos(dip), np.sin(dip))
        normal = np.stack((-sd * ss, sd * cs, -cd), axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        strike_vector = np.stack((cs, ss, np.zeros_like(cs)), axis=1)
        dip_vector = np.stack((ss * cd, -cs * cd, -sd), axis=1)
        return (normal, strike_vector, dip_vector, np.cos(rake), np.sin(rake))
    p1 = plane_arrays(eq6[:, 0], eq6[:, 1], eq6[:, 2])
    p2 = plane_arrays(eq6[:, 3], eq6[:, 4], eq6[:, 5])

    def f32(value):
        return np.ascontiguousarray(np.round(value, int(config.rounding_decimals)), dtype=np.float32)
    weights = np.ascontiguousarray(catalog.weights, dtype=np.float64)
    weights = weights / np.max(weights)
    return PreparedCatalog(n1=f32(p1[0]), n2=f32(p2[0]), strike1=f32(p1[1]), dip1=f32(p1[2]), strike2=f32(p2[1]), dip2=f32(p2[2]), cos_rake1=f32(p1[3]), sin_rake1=f32(p1[4]), cos_rake2=f32(p2[3]), sin_rake2=f32(p2[4]), weights=weights, weight_sum=float(weights.sum()), eq6=np.ascontiguousarray(eq6), uncertainty_deg=np.ascontiguousarray(uncertainty), event_id=np.ascontiguousarray(catalog.event_id))


def read_input_dat(path: str | Path) -> FocalMechanismCatalog:

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            "Place input.dat in the same folder as SLIC.py."
        )
    try:
        values = np.loadtxt(path, dtype=np.float64, comments="#", ndmin=2)
    except ValueError as exc:
        raise ValueError(
            "input.dat must be a headerless, whitespace-delimited numeric "
            "file with exactly three columns: strike dip rake."
        ) from exc
    if values.shape[0] < 1 or values.shape[1] != 3:
        raise ValueError(
            "input.dat must contain exactly three columns: strike dip rake; "
            f"found shape {values.shape}."
        )
    return FocalMechanismCatalog.from_single_planes(
        values[:, 0], values[:, 1], values[:, 2],
        event_id=np.arange(1, values.shape[0] + 1),
    )
