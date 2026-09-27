from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SLICConfig:


    friction_mu: float = 0.6
    perturbation_draws: int = 100
    random_seed: int = 20251201
    mechanism_uncertainty_deg: float | None = None
    default_mechanism_uncertainty_deg: float = 10.0
    minimum_mechanism_uncertainty_deg: float = 0.0
    maximum_mechanism_uncertainty_deg: float = 50.0
    search_mode: str = 'two_stage'
    one_stage_angle_step_deg: float = 5.0
    one_stage_phi_step: float = 0.1
    coarse_angle_step_deg: float = 5.0
    coarse_phi_step: float = 0.1
    refine_angle_half_width_deg: float = 5.0
    refine_phi_half_width: float = 0.1
    refine_angle_step_deg: float = 1.0
    refine_phi_step: float = 0.01
    grid_dtype: str = 'float32'
    rounding_decimals: int = 6
    numba_threads: int = 1

    def validate(self) -> 'SLICConfig':
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'{name} must be finite.')
        for name in ('perturbation_draws', 'random_seed', 'rounding_decimals', 'numba_threads'):
            value = getattr(self, name)
            if int(value) != value or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer.')
        mode = str(self.search_mode).strip().lower()
        if mode not in {'one_stage', 'two_stage'}:
            raise ValueError("search_mode must be 'one_stage' or 'two_stage'.")
        if float(self.friction_mu) < 0.0:
            raise ValueError('friction_mu must be nonnegative.')
        if int(self.perturbation_draws) < 1:
            raise ValueError('perturbation_draws must be at least 1.')
        if self.numba_threads != 1:
            raise ValueError('Serial execution requires numba_threads=1.')
        if str(self.grid_dtype) not in {'float32', 'float64'}:
            raise ValueError("grid_dtype must be 'float32' or 'float64'.")
        if int(self.rounding_decimals) < 0:
            raise ValueError('rounding_decimals cannot be negative.')
        positive = {'one_stage_angle_step_deg': self.one_stage_angle_step_deg, 'one_stage_phi_step': self.one_stage_phi_step, 'coarse_angle_step_deg': self.coarse_angle_step_deg, 'coarse_phi_step': self.coarse_phi_step, 'refine_angle_step_deg': self.refine_angle_step_deg, 'refine_phi_step': self.refine_phi_step}
        bad = [name for name, value in positive.items() if float(value) <= 0.0]
        if bad:
            raise ValueError(f"Grid steps must be positive: {', '.join(bad)}.")
        if float(self.refine_angle_half_width_deg) < 0.0:
            raise ValueError('refine_angle_half_width_deg cannot be negative.')
        if float(self.refine_phi_half_width) < 0.0:
            raise ValueError('refine_phi_half_width cannot be negative.')
        lo = float(self.minimum_mechanism_uncertainty_deg)
        default = float(self.default_mechanism_uncertainty_deg)
        hi = float(self.maximum_mechanism_uncertainty_deg)
        if not 0.0 <= lo <= default <= hi <= 90.0:
            raise ValueError('Mechanism-uncertainty bounds must satisfy 0 <= minimum <= default <= maximum <= 90 degrees.')
        if self.mechanism_uncertainty_deg is not None:
            value = float(self.mechanism_uncertainty_deg)
            if not lo <= value <= hi:
                raise ValueError('mechanism_uncertainty_deg must lie within the configured uncertainty bounds.')
        return self

    def resolved_friction_mu_grid(self) -> tuple[float, ...]:

        return (float(self.friction_mu),)

    def to_dict(self) -> dict:
        return asdict(self)
