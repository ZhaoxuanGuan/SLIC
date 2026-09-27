from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .catalog import FocalMechanismCatalog, prepare_catalog
from .config import SLICConfig
from .geometry import basis_phi_to_tensor, strike_dip_rake_to_vectors, vector_to_azimuth_plunge
from .criticality import optimal_failure_point
from .inversion import _traction_for_normals
from .perturbation import make_perturbation_ensemble
from .results import SLICResult, BootstrapResult


def _configure_plots() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, name: str, output_dir: str | Path) -> None:
    base = (Path(output_dir) / name).with_suffix("")
    for suffix, image_format in ((".tif", "tiff"), (".pdf", "pdf"), (".eps", "eps")):
        fig.savefig(
            base.with_suffix(suffix),
            format=image_format,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def _disk_xy(azimuth_deg, plunge_deg) -> tuple[np.ndarray, np.ndarray]:

    azimuth = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    plunge = np.clip(np.asarray(plunge_deg, dtype=float), 0.0, 90.0)
    radius = np.sqrt(2.0) * np.sin(np.deg2rad(90.0 - plunge) / 2.0)
    return radius * np.sin(azimuth), radius * np.cos(azimuth)


def _draw_projection_disk(ax: plt.Axes) -> None:
    rim = plt.Circle((0.0, 0.0), 1.0, facecolor="#FAFAFA",
                     edgecolor="#222222", linewidth=1.1, zorder=0)
    ax.add_patch(rim)
    for plunge in (30.0, 60.0):
        radius = np.sqrt(2.0) * np.sin(np.deg2rad(90.0 - plunge) / 2.0)
        ax.add_patch(
            plt.Circle((0.0, 0.0), radius, fill=False,
                       edgecolor="#D9D9D9", linewidth=0.55, zorder=0)
        )
    for azimuth in (0.0, 45.0, 90.0, 135.0):
        angle = np.deg2rad(azimuth)
        dx, dy = np.sin(angle), np.cos(angle)
        ax.plot([-dx, dx], [-dy, dy], color="#E3E3E3", linewidth=0.5, zorder=0)
    ax.text(0.0, 1.055, "N", ha="center", va="bottom", fontsize=9)
    ax.text(1.055, 0.0, "E", ha="left", va="center", fontsize=9)
    ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), aspect="equal")
    ax.axis("off")


def plot_stress_axes(results: BootstrapResult, output_dir: str | Path) -> None:

    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    _draw_projection_disk(ax)
    colors = ("#C44E52", "#4C9F70", "#4C78A8")
    markers = ("o", "s", "^")
    labels = (r"$S_1$", r"$S_2$", r"$S_3$")
    runs = results.runs
    original = results.original
    for index, (color, marker, label) in enumerate(zip(colors, markers, labels), 1):
        x, y = _disk_xy(
            runs[f"sigma{index}_azimuth_deg"],
            runs[f"sigma{index}_plunge_deg"],
        )
        ax.scatter(
            x, y, s=24, marker=marker, color=color, alpha=0.28,
            edgecolors="none", rasterized=True,
        )
        x0, y0 = _disk_xy(
            getattr(original, f"sigma{index}_azimuth_deg"),
            getattr(original, f"sigma{index}_plunge_deg"),
        )
        ax.scatter(
            x0, y0, s=115, marker=marker, color=color,
            edgecolors="black", linewidths=0.8, zorder=5, label=label,
        )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3)
    _save_figure(fig, "SLIC_stress_axes.png", output_dir)


def plot_r_distribution(
    catalog: FocalMechanismCatalog,
    config: SLICConfig,
    solution: SLICResult,
    output_dir: str | Path,
) -> None:
    prepared = prepare_catalog(catalog, config)
    normal1, normal2, _, _ = make_perturbation_ensemble(
        prepared, config, seed=solution.perturbation_seed
    )
    tensor = basis_phi_to_tensor(solution.basis, solution.phi)
    _, _, shear1 = _traction_for_normals(normal1, tensor)
    _, _, shear2 = _traction_for_normals(normal2, tensor)
    r1 = np.linalg.norm(shear1.mean(axis=1), axis=1)
    r2 = np.linalg.norm(shear2.mean(axis=1), axis=1)

    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    colors = ("#4C9F70", "#4C78A8")
    parts = ax.violinplot([r1, r2], positions=[1, 2], widths=0.72,
                          showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.32)
    parts["cmedians"].set_color("#222222")
    parts["cmedians"].set_linewidth(1.4)
    rng = np.random.default_rng(config.random_seed)
    for position, values, color in zip((1, 2), (r1, r2), colors):
        jitter = rng.uniform(-0.11, 0.11, size=values.size)
        ax.scatter(position + jitter, values, s=21, color=color, alpha=0.72,
                   edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xticks([1, 2], ["NP1", "NP2"])
    ax.set_ylabel("Mean resultant length, R")
    lower = max(0.0, np.floor((min(r1.min(), r2.min()) - 0.05) * 10.0) / 10.0)
    ax.set_ylim(lower, 1.01)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, "SLIC_R_distribution.png", output_dir)


def _downward_axis(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    vector = vector / np.linalg.norm(vector)
    return -vector if vector[2] < 0.0 else vector


def plot_focal_mechanism_pt_axes(catalog: FocalMechanismCatalog, output_dir: str | Path) -> None:
    p_axes, t_axes = [], []
    for strike, dip, rake in zip(catalog.s1, catalog.d1, catalog.r1):
        normal, slip = strike_dip_rake_to_vectors(strike, dip, rake)
        p_axes.append(_downward_axis((normal - slip) / np.sqrt(2.0)))
        t_axes.append(_downward_axis((normal + slip) / np.sqrt(2.0)))

    def project(vectors):
        directions = [vector_to_azimuth_plunge(vector) for vector in vectors]
        return _disk_xy(
            [value[0] for value in directions],
            [value[1] for value in directions],
        )

    px, py = project(p_axes)
    tx, ty = project(t_axes)
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    _draw_projection_disk(ax)
    ax.scatter(px, py, s=34, marker="^", color="#C44E52", alpha=0.72,
               edgecolor="white", linewidth=0.4, label="P axis")
    ax.scatter(tx, ty, s=32, marker="o", color="#4C78A8", alpha=0.72,
               edgecolor="white", linewidth=0.4, label="T axis")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2)
    _save_figure(fig, "SLIC_focal_mechanism_PT_axes.png", output_dir)


def plot_mohr_perturbations(
    catalog: FocalMechanismCatalog,
    config: SLICConfig,
    solution: SLICResult,
    output_dir: str | Path,
) -> None:
    prepared = prepare_catalog(catalog, config)
    normal1, normal2, _, _ = make_perturbation_ensemble(
        prepared, config, seed=solution.perturbation_seed
    )
    tensor = basis_phi_to_tensor(solution.basis, solution.phi)
    sigma_n1, tau1, _ = _traction_for_normals(normal1, tensor)
    sigma_n2, tau2, _ = _traction_for_normals(normal2, tensor)

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    principal = (-1.0, 2.0 * solution.phi - 1.0, 1.0)
    theta = np.linspace(0.0, np.pi, 500)
    for low, high, width in (
        (principal[0], principal[2], 1.25),
        (principal[0], principal[1], 0.85),
        (principal[1], principal[2], 0.85),
    ):
        center = 0.5 * (low + high)
        radius = 0.5 * (high - low)
        ax.plot(center + radius * np.cos(theta), radius * np.sin(theta),
                color="#444444", linewidth=width, zorder=1)
    ax.scatter(sigma_n1.ravel(), tau1.ravel(), s=11, color="#4C9F70",
               alpha=0.20, edgecolors="none", rasterized=True, label="NP1")
    ax.scatter(sigma_n2.ravel(), tau2.ravel(), s=11, color="#4C78A8",
               alpha=0.20, edgecolors="none", rasterized=True, label="NP2")
    sigma_c, tau_c = optimal_failure_point(solution.friction_mu)
    ax.scatter(sigma_c, tau_c, marker="*", s=125, color="#D55E00",
               edgecolor="black", linewidth=0.55, zorder=5,
               label="Optimal failure point")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel(r"Normalized normal stress, $\sigma_n$")
    ax.set_ylabel(r"Normalized shear stress, $\tau$")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-0.02, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, "SLIC_Mohr_plane_perturbations.png", output_dir)
