from __future__ import annotations

import numpy as np
import pandas as pd
from .catalog import FocalMechanismCatalog
from .config import SLICConfig
from .geometry import basis_rotation_angle_deg, axis_angle_deg
from .inversion import invert
from .results import SLICResult, BootstrapResult


def _bootstrap_indices(n_events: int, repeats: int, random_seed: int) -> list[np.ndarray]:
    output = []
    for run_id in range(1, repeats + 1):
        seed = np.random.SeedSequence([int(random_seed), 1, int(run_id)])
        rng = np.random.default_rng(seed)
        output.append(rng.integers(0, n_events, size=n_events))
    return output


def _bootstrap_worker(payload):
    run_id, catalog, indices, config = payload
    result = invert(catalog.subset(indices), config)
    return (run_id, int(np.unique(indices).size), result)


def _result_row(run_id: int, unique_events: int, n_events: int, result: SLICResult, original: SLICResult) -> dict:
    row = {'run_id': int(run_id)}
    row.update(result.to_dict())
    row.update(bootstrap_unique_events=int(unique_events), bootstrap_unique_fraction=float(unique_events / n_events), rotation_from_original_deg=basis_rotation_angle_deg(result.basis, original.basis), sigma1_axis_angle_deg=axis_angle_deg(result.basis[:, 0], original.basis[:, 0]), sigma2_axis_angle_deg=axis_angle_deg(result.basis[:, 1], original.basis[:, 1]), sigma3_axis_angle_deg=axis_angle_deg(result.basis[:, 2], original.basis[:, 2]))
    return row


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    columns = ['sigma1_azimuth_deg', 'sigma1_plunge_deg', 'sigma2_azimuth_deg', 'sigma2_plunge_deg', 'sigma3_azimuth_deg', 'sigma3_plunge_deg', 'Phi', 'objective', 'rotation_from_original_deg', 'sigma1_axis_angle_deg', 'sigma2_axis_angle_deg', 'sigma3_axis_angle_deg']
    rows = []
    for column in columns:
        values = pd.to_numeric(runs[column], errors='coerce').dropna().to_numpy()
        if not values.size:
            continue
        rows.append({'quantity': column, 'mean': float(np.mean(values)), 'standard_deviation': float(np.std(values, ddof=1) if values.size > 1 else 0.0), 'median': float(np.median(values)), 'q025': float(np.quantile(values, 0.025)), 'q975': float(np.quantile(values, 0.975)), 'n': int(values.size)})
    return pd.DataFrame(rows)


def bootstrap(catalog: FocalMechanismCatalog, config: SLICConfig | None=None, *, repeats: int=100, workers: int=1, progress=None, cancelled=None) -> BootstrapResult:


    config = (config or SLICConfig()).validate()
    repeats = int(repeats)
    workers = int(workers)
    if repeats < 1:
        raise ValueError('repeats must be at least 1.')
    if workers != 1:
        raise ValueError('Serial execution requires workers=1.')
    def checkpoint(done):
        if cancelled is not None and cancelled():
            raise InterruptedError('Calculation cancelled.')
        if progress is not None:
            progress(done, repeats + 1)
    checkpoint(0)
    original = invert(catalog, config)
    checkpoint(1)
    index_sets = _bootstrap_indices(catalog.size, repeats, config.random_seed)
    outputs: dict[int, tuple[int, SLICResult]] = {}
    for run_id, indices in enumerate(index_sets, start=1):
        _, unique_events, result = _bootstrap_worker((run_id, catalog, indices, config))
        outputs[run_id] = (unique_events, result)
        checkpoint(run_id + 1)
    rows = [_result_row(run_id, outputs[run_id][0], catalog.size, outputs[run_id][1], original) for run_id in range(1, repeats + 1)]
    runs = pd.DataFrame(rows)
    return BootstrapResult(original=original, runs=runs, summary=_summarize(runs))
