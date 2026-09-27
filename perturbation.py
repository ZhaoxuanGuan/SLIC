from __future__ import annotations

import numpy as np
from .catalog import PreparedCatalog
from .config import SLICConfig


def resolve_catalog_uncertainty(prepared: PreparedCatalog, config: SLICConfig) -> tuple[float, str]:


    if config.mechanism_uncertainty_deg is not None:
        return (float(config.mechanism_uncertainty_deg), 'explicit configuration')
    values = np.asarray(prepared.uncertainty_deg, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size and np.allclose(finite, finite[0], rtol=0.0, atol=1e-09):
        return (float(finite[0]), 'common catalogue value')
    return (float(config.default_mechanism_uncertainty_deg), 'default because event uncertainties are not common')


def make_perturbation_ensemble(prepared: PreparedCatalog, config: SLICConfig, *, seed: int | None=None) -> tuple[np.ndarray, np.ndarray, float, str]:


    draws = int(config.perturbation_draws)
    if draws < 1:
        raise ValueError('perturbation_draws must be at least 1.')
    uncertainty_deg, uncertainty_mode = resolve_catalog_uncertainty(prepared, config)
    rng = np.random.default_rng(int(config.random_seed if seed is None else seed))
    mechanisms = prepared.eq6
    n_events = mechanisms.shape[0]
    strike0, dip0, rake0 = np.deg2rad(mechanisms[:, :3]).T
    normal0 = np.stack((-np.sin(dip0) * np.sin(strike0), np.sin(dip0) * np.cos(strike0), -np.cos(dip0)), axis=1)
    strike_vector0 = np.stack((np.cos(strike0), np.sin(strike0), np.zeros(n_events)), axis=1)
    dip_vector0 = np.stack((np.sin(strike0) * np.cos(dip0), -np.cos(strike0) * np.cos(dip0), -np.sin(dip0)), axis=1)
    slip0 = np.cos(rake0)[:, None] * strike_vector0 + np.sin(rake0)[:, None] * dip_vector0
    sign1 = np.sign(np.sum(-normal0 * -prepared.n1, axis=1))
    sign2 = np.sign(np.sum(slip0 * -prepared.n2, axis=1))
    sign1[sign1 == 0.0] = 1.0
    sign2[sign2 == 0.0] = 1.0
    normals1 = np.empty((n_events, draws, 3), dtype=np.float32)
    normals2 = np.empty((n_events, draws, 3), dtype=np.float32)
    uncertainty = np.full(n_events, uncertainty_deg, dtype=np.float64)
    for draw in range(draws):
        noise = rng.uniform(-uncertainty[:, None], uncertainty[:, None], size=(n_events, 3))
        strike = np.deg2rad(np.mod(mechanisms[:, 0] + noise[:, 0], 360.0))
        dip = np.deg2rad(np.clip(mechanisms[:, 1] + noise[:, 1], 0.0, 90.0))
        rake = np.deg2rad(np.mod(mechanisms[:, 2] + noise[:, 2] + 180.0, 360.0) - 180.0)
        normal = np.stack((-np.sin(dip) * np.sin(strike), np.sin(dip) * np.cos(strike), -np.cos(dip)), axis=1)
        strike_vector = np.stack((np.cos(strike), np.sin(strike), np.zeros(n_events)), axis=1)
        dip_vector = np.stack((np.sin(strike) * np.cos(dip), -np.cos(strike) * np.cos(dip), -np.sin(dip)), axis=1)
        slip = np.cos(rake)[:, None] * strike_vector + np.sin(rake)[:, None] * dip_vector
        normals1[:, draw, :] = sign1[:, None] * -normal
        normals2[:, draw, :] = sign2[:, None] * slip
    return (np.ascontiguousarray(normals1), np.ascontiguousarray(normals2), float(uncertainty_deg), uncertainty_mode)
