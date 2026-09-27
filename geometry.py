from __future__ import annotations

import math
import numpy as np

TINY = 1e-12


def wrap_360(angle):

    value = np.mod(angle, 360.0)
    return float(value) if np.isscalar(angle) else value


def wrap_180(angle):

    value = (np.asarray(angle) + 180.0) % 360.0 - 180.0
    return float(value) if np.isscalar(angle) else value


def strike_dip_to_normal(strike_deg: float, dip_deg: float) -> np.ndarray:
    strike, dip = np.deg2rad([strike_deg, dip_deg])
    normal = np.array([-np.sin(dip) * np.sin(strike), np.sin(dip) * np.cos(strike), -np.cos(dip)], dtype=np.float64)
    return normal / np.linalg.norm(normal)


def strike_vector(strike_deg: float) -> np.ndarray:
    strike = math.radians(float(strike_deg))
    return np.array([math.cos(strike), math.sin(strike), 0.0], dtype=np.float64)


def dip_vector(strike_deg: float, dip_deg: float) -> np.ndarray:
    direction = np.cross(strike_dip_to_normal(strike_deg, dip_deg), strike_vector(strike_deg))
    direction /= np.linalg.norm(direction)
    return -direction if direction[2] > 0.0 else direction


def strike_dip_rake_to_vectors(strike_deg: float, dip_deg: float, rake_deg: float) -> tuple[np.ndarray, np.ndarray]:
    normal = strike_dip_to_normal(strike_deg, dip_deg)
    rake = math.radians(float(rake_deg))
    slip = math.cos(rake) * strike_vector(strike_deg) + math.sin(rake) * dip_vector(strike_deg, dip_deg)
    return (normal, slip / np.linalg.norm(slip))


def vectors_to_sdr(normal_in: np.ndarray, slip_in: np.ndarray) -> tuple[float, float, float]:
    normal = np.asarray(normal_in, dtype=np.float64).copy()
    slip = np.asarray(slip_in, dtype=np.float64).copy()
    if np.linalg.norm(normal) <= TINY or np.linalg.norm(slip) <= TINY:
        raise ValueError('Normal and slip vectors must be nonzero.')
    normal /= np.linalg.norm(normal)
    slip /= np.linalg.norm(slip)
    if normal[2] > 0.0:
        normal, slip = (-normal, -slip)
    strike = math.degrees(math.atan2(-normal[0], normal[1])) % 360.0
    dip = math.degrees(math.acos(float(np.clip(-normal[2], -1.0, 1.0))))
    rake = math.degrees(math.atan2(float(np.dot(slip, dip_vector(strike, dip))), float(np.dot(slip, strike_vector(strike)))))
    return (float(strike), float(dip), float(wrap_180(rake)))


def auxiliary_plane(strike_deg: float, dip_deg: float, rake_deg: float) -> tuple[float, float, float]:

    normal, slip = strike_dip_rake_to_vectors(strike_deg, dip_deg, rake_deg)
    return vectors_to_sdr(slip, normal)


def euler_basis(a1_deg: float, a2_deg: float, a3_deg: float) -> np.ndarray:

    a1, a2, a3 = np.deg2rad([a1_deg, a2_deg, a3_deg])
    c1, s1 = (math.cos(a1), math.sin(a1))
    c2, s2 = (math.cos(a2), math.sin(a2))
    c3, s3 = (math.cos(a3), math.sin(a3))
    axis1 = np.array([c1 * c3 - s1 * c2 * s3, s1 * c3 + c1 * c2 * s3, s2 * s3])
    axis2 = np.array([-c1 * s3 - s1 * c2 * c3, -s1 * s3 + c1 * c2 * c3, s2 * c3])
    axis3 = np.array([s1 * s2, -c1 * s2, c2])
    return np.column_stack((axis1, axis2, axis3)).astype(np.float64)


def basis_phi_to_tensor(basis: np.ndarray, phi: float) -> np.ndarray:
    basis = np.asarray(basis, dtype=np.float64)
    return basis @ np.diag([1.0, 2.0 * float(phi) - 1.0, -1.0]) @ basis.T


def vector_to_azimuth_plunge(vector: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(vector, dtype=np.float64).copy().ravel()
    if np.linalg.norm(vector) <= TINY:
        return (float('nan'), float('nan'))
    if vector[2] > 0.0:
        vector = -vector
    vector /= np.linalg.norm(vector)
    plunge = math.degrees(math.asin(float(-vector[2])))
    if plunge > 89.999999:
        return (0.0, float(plunge))
    azimuth = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
    if abs(plunge) < 1e-08 and azimuth >= 180.0:
        azimuth -= 180.0
    return (float(azimuth), float(plunge))


def canonicalize_basis(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis, dtype=np.float64).copy()
    for column in range(3):
        vector = basis[:, column]
        if vector[2] > 0.0:
            basis[:, column] = -vector
        elif abs(vector[2]) < TINY:
            azimuth, _ = vector_to_azimuth_plunge(vector)
            if azimuth >= 180.0:
                basis[:, column] = -vector
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0
    return basis


def tensor_to_basis_phi(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    tensor = np.asarray(tensor, dtype=np.float64)
    symmetric = 0.5 * (tensor + tensor.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    basis = canonicalize_basis(eigenvectors[:, order])
    denominator = float(eigenvalues[0] - eigenvalues[2])
    phi = 0.5 if abs(denominator) <= TINY else float((eigenvalues[1] - eigenvalues[2]) / denominator)
    return (basis, float(np.clip(phi, 0.0, 1.0)))


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def align_basis_signs(basis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    options = (np.diag([1.0, 1.0, 1.0]), np.diag([-1.0, -1.0, 1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([1.0, -1.0, -1.0]))
    best = np.asarray(basis, dtype=np.float64).copy()
    best_angle = float('inf')
    for signs in options:
        candidate = basis @ signs
        angle = rotation_angle_deg(candidate @ reference.T)
        if angle < best_angle:
            best_angle = angle
            best = candidate
    return best


def basis_rotation_angle_deg(basis: np.ndarray, reference: np.ndarray) -> float:
    aligned = align_basis_signs(basis, reference)
    return rotation_angle_deg(aligned @ reference.T)


def axis_angle_deg(axis: np.ndarray, reference: np.ndarray) -> float:
    cosine = abs(float(np.dot(axis, reference)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
