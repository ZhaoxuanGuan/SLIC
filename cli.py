from __future__ import annotations

import argparse
from pathlib import Path
from .config import SLICConfig
from .catalog import read_input_dat
from .bootstrap import bootstrap
from .results import SLICResult, _result_table
from .plotting import _configure_plots, plot_stress_axes, plot_r_distribution, plot_focal_mechanism_pt_axes, plot_mohr_perturbations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Invert every focal mechanism in input.dat with SLIC; no spatial "
            "or temporal partitioning is performed."
        )
    )
    parser.add_argument(
        "--uncertainty-deg", type=float, default=None
    )
    parser.add_argument("--friction-mu", type=float, default=0.6)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20251201)
    parser.add_argument("--numba-threads", type=int, choices=[1], default=1)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--workers", type=int, choices=[1], default=1)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "input.dat")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def make_config(args: argparse.Namespace) -> SLICConfig:
    return SLICConfig(
        friction_mu=args.friction_mu,
        perturbation_draws=args.draws,
        random_seed=args.seed,
        mechanism_uncertainty_deg=args.uncertainty_deg,
        default_mechanism_uncertainty_deg=10.0,
        search_mode="two_stage",
        coarse_angle_step_deg=5.0,
        coarse_phi_step=0.1,
        refine_angle_half_width_deg=5.0,
        refine_phi_half_width=0.1,
        refine_angle_step_deg=1.0,
        refine_phi_step=0.01,
        numba_threads=args.numba_threads,
    ).validate()


def print_solution(solution: SLICResult) -> None:
    print("\nBest-fitting stress state")
    print("-------------------------")
    print(
        f"S1: azimuth {solution.sigma1_azimuth_deg:.2f} deg, "
        f"plunge {solution.sigma1_plunge_deg:.2f} deg"
    )
    print(
        f"S2: azimuth {solution.sigma2_azimuth_deg:.2f} deg, "
        f"plunge {solution.sigma2_plunge_deg:.2f} deg"
    )
    print(
        f"S3: azimuth {solution.sigma3_azimuth_deg:.2f} deg, "
        f"plunge {solution.sigma3_plunge_deg:.2f} deg"
    )
    print(f"Phi: {solution.phi:.3f}")
    print(f"Objective: {solution.objective:.6f}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be at least 1.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    config = make_config(args)
    catalog = read_input_dat(args.input)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = bootstrap(
        catalog, config, repeats=args.bootstrap, workers=args.workers
    )
    solution = results.original
    table = _result_table(results)
    table.to_csv(output_dir / "SLIC_results.csv", index=False)

    _configure_plots()
    plot_stress_axes(results, output_dir)
    plot_r_distribution(catalog, config, solution, output_dir)
    plot_focal_mechanism_pt_axes(catalog, output_dir)
    plot_mohr_perturbations(catalog, config, solution, output_dir)

    print_solution(solution)
    print(f"\nResults written to: {output_dir}")
    return 0
