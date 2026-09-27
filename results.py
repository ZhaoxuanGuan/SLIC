from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SLICResult:
    sigma1_azimuth_deg: float
    sigma1_plunge_deg: float
    sigma2_azimuth_deg: float
    sigma2_plunge_deg: float
    sigma3_azimuth_deg: float
    sigma3_plunge_deg: float
    phi: float
    objective: float
    friction_mu: float
    search_mode: str
    a1_deg: float
    a2_deg: float
    a3_deg: float
    coarse_a1_deg: float | None
    coarse_a2_deg: float | None
    coarse_a3_deg: float | None
    coarse_phi: float | None
    mechanism_uncertainty_deg: float
    uncertainty_mode: str
    perturbation_draws: int
    perturbation_seed: int
    basis: np.ndarray
    event_id: np.ndarray
    plane1_weight: np.ndarray
    plane2_weight: np.ndarray
    direction_score1: np.ndarray
    direction_score2: np.ndarray
    mean_criticality1: np.ndarray
    mean_criticality2: np.ndarray
    event_log_likelihood: np.ndarray

    def to_dict(self) -> dict:

        return {'sigma1_azimuth_deg': float(self.sigma1_azimuth_deg), 'sigma1_plunge_deg': float(self.sigma1_plunge_deg), 'sigma2_azimuth_deg': float(self.sigma2_azimuth_deg), 'sigma2_plunge_deg': float(self.sigma2_plunge_deg), 'sigma3_azimuth_deg': float(self.sigma3_azimuth_deg), 'sigma3_plunge_deg': float(self.sigma3_plunge_deg), 'Phi': float(self.phi), 'objective': float(self.objective), 'friction_mu': float(self.friction_mu), 'search_mode': self.search_mode, 'a1_deg': float(self.a1_deg), 'a2_deg': float(self.a2_deg), 'a3_deg': float(self.a3_deg), 'coarse_a1_deg': None if self.coarse_a1_deg is None else float(self.coarse_a1_deg), 'coarse_a2_deg': None if self.coarse_a2_deg is None else float(self.coarse_a2_deg), 'coarse_a3_deg': None if self.coarse_a3_deg is None else float(self.coarse_a3_deg), 'coarse_Phi': None if self.coarse_phi is None else float(self.coarse_phi), 'mechanism_uncertainty_deg': float(self.mechanism_uncertainty_deg), 'uncertainty_mode': self.uncertainty_mode, 'perturbation_draws': int(self.perturbation_draws), 'perturbation_seed': int(self.perturbation_seed), 'objective_definition': 'mean(log(w1*exp(g1) + w2*exp(g2))); g_j = mean perturbed unit shear dot observed slip'}

    def diagnostics_frame(self) -> pd.DataFrame:
        return pd.DataFrame({'event_id': self.event_id, 'plane1_weight': self.plane1_weight, 'plane2_weight': self.plane2_weight, 'direction_score1': self.direction_score1, 'direction_score2': self.direction_score2, 'mean_criticality1': self.mean_criticality1, 'mean_criticality2': self.mean_criticality2, 'event_log_likelihood': self.event_log_likelihood})


@dataclass(frozen=True)
class BootstrapResult:
    original: SLICResult
    runs: pd.DataFrame
    summary: pd.DataFrame


def _result_table(results: BootstrapResult) -> pd.DataFrame:

    scalar_columns = [
        "sigma1_azimuth_deg", "sigma1_plunge_deg",
        "sigma2_azimuth_deg", "sigma2_plunge_deg",
        "sigma3_azimuth_deg", "sigma3_plunge_deg",
        "Phi", "objective",
    ]
    optimal = {column: results.original.to_dict()[column] for column in scalar_columns}
    optimal.update(
        result_type="optimal",
        run_id=0,
        bootstrap_unique_events=np.nan,
        bootstrap_unique_fraction=np.nan,
        rotation_from_optimal_deg=0.0,
        sigma1_axis_angle_deg=0.0,
        sigma2_axis_angle_deg=0.0,
        sigma3_axis_angle_deg=0.0,
    )
    runs = results.runs.rename(
        columns={"rotation_from_original_deg": "rotation_from_optimal_deg"}
    ).copy()
    runs.insert(0, "result_type", "bootstrap")
    ordered = [
        "result_type", "run_id", *scalar_columns,
        "bootstrap_unique_events", "bootstrap_unique_fraction",
        "rotation_from_optimal_deg", "sigma1_axis_angle_deg",
        "sigma2_axis_angle_deg", "sigma3_axis_angle_deg",
    ]
    return pd.concat(
        [pd.DataFrame([optimal]), runs[ordered]], ignore_index=True
    )[ordered]
