from __future__ import annotations

import math
import numpy as np
from numba import njit
from .catalog import FocalMechanismCatalog, PreparedCatalog, prepare_catalog
from .config import SLICConfig
from .criticality import failure_criticality, failure_criticality_numba
from .geometry import euler_basis, tensor_to_basis_phi, basis_phi_to_tensor, vector_to_azimuth_plunge
from .perturbation import make_perturbation_ensemble
from .results import SLICResult


def _grid_dtype(config: SLICConfig):
    return np.float32 if config.grid_dtype == 'float32' else np.float64


def full_search_grids(angle_step_deg: float, phi_step: float, dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.arange(0.0, 180.0 + 1e-12, angle_step_deg, dtype=dtype), np.arange(0.0, 90.0 + 1e-12, angle_step_deg, dtype=dtype), np.arange(0.0, 180.0 + 1e-12, angle_step_deg, dtype=dtype), np.arange(0.0, 1.0 + 1e-12, phi_step, dtype=dtype))


def _clamped_grid(center: float, half_width: float, step: float, minimum: float, maximum: float, dtype) -> np.ndarray:
    start = max(minimum, float(center) - float(half_width))
    stop = min(maximum, float(center) + float(half_width))
    values = np.arange(start, stop + 1e-12, step, dtype=np.float64)
    if values.size == 0:
        values = np.array([np.clip(center, minimum, maximum)])
    return values.astype(dtype)


def refinement_grids(coarse: tuple[float, float, float, float], config: SLICConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a1, a2, a3, phi = coarse
    dtype = _grid_dtype(config)
    return (_clamped_grid(a1, config.refine_angle_half_width_deg, config.refine_angle_step_deg, 0.0, 180.0, dtype), _clamped_grid(a2, config.refine_angle_half_width_deg, config.refine_angle_step_deg, 0.0, 90.0, dtype), _clamped_grid(a3, config.refine_angle_half_width_deg, config.refine_angle_step_deg, 0.0, 180.0, dtype), _clamped_grid(phi, config.refine_phi_half_width, config.refine_phi_step, 0.0, 1.0, dtype))


@njit(cache=True, inline='always')
def _plane_values(normal, sigma2, v11, v12, v13, v21, v22, v23, v31, v32, v33):
    p1 = normal[0] * v11 + normal[1] * v12 + normal[2] * v13
    p2 = normal[0] * v21 + normal[1] * v22 + normal[2] * v23
    p3 = normal[0] * v31 + normal[1] * v32 + normal[2] * v33
    sigma_n = p1 * p1 + sigma2 * p2 * p2 - p3 * p3
    q1 = (1.0 - sigma_n) * p1
    q2 = (sigma2 - sigma_n) * p2
    q3 = (-1.0 - sigma_n) * p3
    tau = math.sqrt(max(q1 * q1 + q2 * q2 + q3 * q3, 0.0))
    if tau <= 1e-14:
        return (sigma_n, tau, 0.0, 0.0, 0.0)
    return (sigma_n, tau, (q1 * v11 + q2 * v21 + q3 * v31) / tau, (q1 * v12 + q2 * v22 + q3 * v32) / tau, (q1 * v13 + q2 * v23 + q3 * v33) / tau)


@njit(cache=True)
def _mean_objective(v11, v12, v13, v21, v22, v23, v31, v32, v33, phi_values, original_slip1, original_slip2, perturbed_normal1, perturbed_normal2, event_weights, weight_sum, friction_mu):
    draws = perturbed_normal1.shape[1]
    inverse_draws = 1.0 / draws
    best_value = -1e+300
    best_phi = 0.0
    for phi_index in range(phi_values.size):
        sigma2 = 2.0 * phi_values[phi_index] - 1.0
        total = 0.0
        for event in range(event_weights.size):
            wins1 = 0.0
            mean_ux1 = mean_uy1 = mean_uz1 = 0.0
            mean_ux2 = mean_uy2 = mean_uz2 = 0.0
            for draw in range(draws):
                sigma1, tau1, ux1, uy1, uz1 = _plane_values(perturbed_normal1[event, draw], sigma2, v11, v12, v13, v21, v22, v23, v31, v32, v33)
                sigma2_n, tau2, ux2, uy2, uz2 = _plane_values(perturbed_normal2[event, draw], sigma2, v11, v12, v13, v21, v22, v23, v31, v32, v33)
                criticality1 = failure_criticality_numba(sigma1, tau1, friction_mu)
                criticality2 = failure_criticality_numba(sigma2_n, tau2, friction_mu)
                mean_ux1 += ux1
                mean_uy1 += uy1
                mean_uz1 += uz1
                mean_ux2 += ux2
                mean_uy2 += uy2
                mean_uz2 += uz2
                if abs(criticality1 - criticality2) <= 1e-12:
                    wins1 += 0.5
                elif criticality1 > criticality2:
                    wins1 += 1.0
            score1 = inverse_draws * (mean_ux1 * original_slip1[event, 0] + mean_uy1 * original_slip1[event, 1] + mean_uz1 * original_slip1[event, 2])
            score2 = inverse_draws * (mean_ux2 * original_slip2[event, 0] + mean_uy2 * original_slip2[event, 1] + mean_uz2 * original_slip2[event, 2])
            weight1 = wins1 * inverse_draws
            if weight1 <= 1e-15:
                log_mixture = score2
            elif weight1 >= 1.0 - 1e-15:
                log_mixture = score1
            else:
                term1 = math.log(weight1) + score1
                term2 = math.log(1.0 - weight1) + score2
                maximum = max(term1, term2)
                log_mixture = maximum + math.log(math.exp(term1 - maximum) + math.exp(term2 - maximum))
            total += log_mixture * event_weights[event]
        value = total / weight_sum
        if value > best_value:
            best_value = value
            best_phi = phi_values[phi_index]
    return (best_value, best_phi)


@njit(cache=True, nogil=True)
def _search_grid(a1_values, a2_values, a3_values, phi_values, cos_a1, sin_a1, cos_a2, sin_a2, cos_a3, sin_a3, original_slip1, original_slip2, perturbed_normal1, perturbed_normal2, event_weights, weight_sum, friction_mu):
    jobs = a1_values.size * a3_values.size
    values = np.full(jobs, -1e+30, dtype=np.float64)
    parameters = np.zeros((jobs, 4), dtype=np.float64)
    for job in range(jobs):
        index1 = job // a3_values.size
        index3 = job % a3_values.size
        local_best = -1e+300
        for index2 in range(a2_values.size):
            c1, s1 = (cos_a1[index1], sin_a1[index1])
            c2, s2 = (cos_a2[index2], sin_a2[index2])
            c3, s3 = (cos_a3[index3], sin_a3[index3])
            value, phi = _mean_objective(c1 * c3 - s1 * c2 * s3, s1 * c3 + c1 * c2 * s3, s2 * s3, -c1 * s3 - s1 * c2 * c3, -s1 * s3 + c1 * c2 * c3, s2 * c3, s1 * s2, -c1 * s2, c2, phi_values, original_slip1, original_slip2, perturbed_normal1, perturbed_normal2, event_weights, weight_sum, friction_mu)
            if value > local_best:
                local_best = value
                parameters[job, 0] = a1_values[index1]
                parameters[job, 1] = a2_values[index2]
                parameters[job, 2] = a3_values[index3]
                parameters[job, 3] = phi
        values[job] = local_best
    return (values, parameters)


def _observed_slips(prepared: PreparedCatalog) -> tuple[np.ndarray, np.ndarray]:
    slip1 = prepared.cos_rake1[:, None] * prepared.strike1 + prepared.sin_rake1[:, None] * prepared.dip1
    slip2 = prepared.cos_rake2[:, None] * prepared.strike2 + prepared.sin_rake2[:, None] * prepared.dip2
    return (np.ascontiguousarray(slip1), np.ascontiguousarray(slip2))


def _invert_grid(grids: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], prepared: PreparedCatalog, perturbations: tuple[np.ndarray, np.ndarray], friction_mu: float) -> tuple[float, float, float, float, float]:
    a1, a2, a3, phi = grids
    slip1, slip2 = _observed_slips(prepared)

    def trig(values):
        radians = np.asarray(values, dtype=np.float64) * np.pi / 180.0
        return (np.cos(radians), np.sin(radians))
    cos_a1, sin_a1 = trig(a1)
    cos_a2, sin_a2 = trig(a2)
    cos_a3, sin_a3 = trig(a3)
    values, parameters = _search_grid(a1, a2, a3, phi, cos_a1, sin_a1, cos_a2, sin_a2, cos_a3, sin_a3, slip1, slip2, perturbations[0], perturbations[1], prepared.weights, prepared.weight_sum, float(friction_mu))
    index = int(np.argmax(values))
    return (float(parameters[index, 0]), float(parameters[index, 1]), float(parameters[index, 2]), float(parameters[index, 3]), float(values[index]))


def _traction_for_normals(normals: np.ndarray, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traction = np.einsum('...j,ij->...i', normals, tensor)
    sigma_n = np.sum(normals * traction, axis=-1)
    shear = traction - sigma_n[..., None] * normals
    tau = np.linalg.norm(shear, axis=-1)
    unit_shear = np.zeros_like(shear)
    mask = tau > 1e-14
    unit_shear[mask] = shear[mask] / tau[mask][:, None]
    return (sigma_n, tau, unit_shear)


def _diagnostics(prepared: PreparedCatalog, perturbations: tuple[np.ndarray, np.ndarray], basis: np.ndarray, phi: float, friction_mu: float) -> dict[str, np.ndarray]:
    tensor = basis_phi_to_tensor(basis, phi)
    sigma1, tau1, unit1 = _traction_for_normals(perturbations[0], tensor)
    sigma2, tau2, unit2 = _traction_for_normals(perturbations[1], tensor)
    criticality1 = failure_criticality(sigma1, tau1, friction_mu)
    criticality2 = failure_criticality(sigma2, tau2, friction_mu)
    ties = np.isclose(criticality1, criticality2, rtol=0.0, atol=1e-12)
    wins1 = np.where(ties, 0.5, (criticality1 > criticality2).astype(np.float64))
    weight1 = wins1.mean(axis=1)
    weight2 = 1.0 - weight1
    slip1, slip2 = _observed_slips(prepared)
    score1 = np.sum(unit1.mean(axis=1) * slip1, axis=1)
    score2 = np.sum(unit2.mean(axis=1) * slip2, axis=1)
    log_likelihood = np.empty(prepared.weights.size, dtype=np.float64)
    for index, (w1, w2, g1, g2) in enumerate(zip(weight1, weight2, score1, score2)):
        if w1 <= 1e-15:
            log_likelihood[index] = g2
        elif w2 <= 1e-15:
            log_likelihood[index] = g1
        else:
            log_likelihood[index] = np.logaddexp(math.log(w1) + g1, math.log(w2) + g2)
    return {'weight1': weight1, 'weight2': weight2, 'score1': score1, 'score2': score2, 'criticality1': criticality1.mean(axis=1), 'criticality2': criticality2.mean(axis=1), 'log_likelihood': log_likelihood}


def invert(catalog: FocalMechanismCatalog, config: SLICConfig | None=None) -> SLICResult:

    config = (config or SLICConfig()).validate()
    prepared = prepare_catalog(catalog, config)
    perturbation_seed = int(config.random_seed)
    normals1, normals2, uncertainty_deg, uncertainty_mode = make_perturbation_ensemble(prepared, config, seed=perturbation_seed)
    perturbations = (normals1, normals2)
    dtype = _grid_dtype(config)
    mode = str(config.search_mode).strip().lower()
    best_result = None
    best_coarse_result = None
    best_friction_mu = None
    for friction_mu in config.resolved_friction_mu_grid():
        coarse_result = None
        if mode == 'one_stage':
            result = _invert_grid(full_search_grids(config.one_stage_angle_step_deg, config.one_stage_phi_step, dtype), prepared, perturbations, friction_mu)
        else:
            coarse_result = _invert_grid(full_search_grids(config.coarse_angle_step_deg, config.coarse_phi_step, dtype), prepared, perturbations, friction_mu)
            result = _invert_grid(refinement_grids(coarse_result[:4], config), prepared, perturbations, friction_mu)
            if coarse_result[4] > result[4]:
                result = coarse_result
        if best_result is None or result[4] > best_result[4]:
            best_result = result
            best_coarse_result = coarse_result
            best_friction_mu = float(friction_mu)
    if best_result is None or best_friction_mu is None:
        raise RuntimeError('No friction candidate was evaluated.')
    result = best_result
    coarse_result = best_coarse_result
    a1, a2, a3, phi_grid, objective = result
    raw_basis = euler_basis(a1, a2, a3)
    basis, phi = tensor_to_basis_phi(basis_phi_to_tensor(raw_basis, phi_grid))
    axes = [vector_to_azimuth_plunge(basis[:, index]) for index in range(3)]
    diagnostics = _diagnostics(prepared, perturbations, basis, phi, best_friction_mu)
    return SLICResult(sigma1_azimuth_deg=axes[0][0], sigma1_plunge_deg=axes[0][1], sigma2_azimuth_deg=axes[1][0], sigma2_plunge_deg=axes[1][1], sigma3_azimuth_deg=axes[2][0], sigma3_plunge_deg=axes[2][1], phi=phi, objective=objective, friction_mu=best_friction_mu, search_mode=mode, a1_deg=a1, a2_deg=a2, a3_deg=a3, coarse_a1_deg=None if coarse_result is None else coarse_result[0], coarse_a2_deg=None if coarse_result is None else coarse_result[1], coarse_a3_deg=None if coarse_result is None else coarse_result[2], coarse_phi=None if coarse_result is None else coarse_result[3], mechanism_uncertainty_deg=uncertainty_deg, uncertainty_mode=uncertainty_mode, perturbation_draws=int(config.perturbation_draws), perturbation_seed=perturbation_seed, basis=basis, event_id=prepared.event_id, plane1_weight=diagnostics['weight1'], plane2_weight=diagnostics['weight2'], direction_score1=diagnostics['score1'], direction_score2=diagnostics['score2'], mean_criticality1=diagnostics['criticality1'], mean_criticality2=diagnostics['criticality2'], event_log_likelihood=diagnostics['log_likelihood'])
