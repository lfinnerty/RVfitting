#!/usr/bin/env python3
"""Generate synthetic ExoArchive-style RV data from a saved system fit."""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.time import Time

import rv_io
import plot_rvs


def find_parameters(system: str) -> tuple[Path, dict]:
	"""Find and load the saved fit parameters for a system."""
	target = rv_io.normalize_identifier(system)
	for path in plot_rvs.PLOTS_DIRECTORY.glob("*_rv_fit_parameters.json"):
		saved_name = path.name.removesuffix("_rv_fit_parameters.json")
		if rv_io.normalize_identifier(saved_name) == target:
			return path, json.loads(path.read_text(encoding="utf-8"))
	raise FileNotFoundError(
		f"No saved fit parameters found for {system!r} in {plot_rvs.PLOTS_DIRECTORY}"
	)


def parse_date(value: str) -> float:
	"""Parse a BJD number or an ISO date into a BJD-like Julian date."""
	try:
		return float(value)
	except ValueError:
		return float(Time(value, format="isot", scale="utc").jd)


def load_reference_observations(
	system: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Load observations for noise estimates, preferring HIRES and HARPS."""
	datasets = rv_io.load_datasets(system, ("Teklu", "HARPS", "HARPS2020")) or rv_io.load_datasets(
		system, ("ExoArchive", "CLS", "SOPHIE", "Hebrard")
	)
	if not datasets:
		raise ValueError(f"No RV observations found for {system!r}")
	return rv_io.combine_rv_data(datasets)


def draw_synthetic_points(
	parameters: dict,
	dates: np.ndarray,
	reference_times: np.ndarray,
	reference_velocities: np.ndarray,
	reference_errors: np.ndarray,
	reference_sources: np.ndarray,
	rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
	"""Draw fit parameters and synthetic RVs using empirical source noise.

	The period, semiamplitude, and conjunction are drawn from their uncertainties;
	the eccentricity and argument of periastron are held at the adopted values.
	"""
	period = rng.normal(parameters["period_days"], parameters["period_uncertainty_days"])
	semiamplitude = max(0.0, rng.normal(
		parameters["semiamplitude_m_per_s"], parameters["semiamplitude_uncertainty_m_per_s"]
	))
	conjunction = rng.normal(
		parameters["conjunction_bjd"], parameters["conjunction_uncertainty_days"]
	)
	orbit = plot_rvs.saved_orbit(parameters, period, semiamplitude, conjunction)
	orbit[plot_rvs.ORBIT_GAMMA] = 0.0

	outer, outer_parameters = plot_rvs.saved_outer_signal(parameters)

	def model(times: np.ndarray) -> np.ndarray:
		rv = plot_rvs.orbit_rv(times, orbit, np.empty((len(times), 0)))
		return rv + outer.evaluate(times, outer_parameters) if outer is not None else rv

	reference_model = model(reference_times)
	for source, offset in parameters.get("source_offsets_m_per_s", {}).items():
		reference_model[reference_sources == source] += offset
	residuals = reference_velocities - reference_model
	for source in np.unique(reference_sources):
		mask = reference_sources == source
		residuals[mask] -= np.median(residuals[mask])
	scatter = max(
		float(1.4826 * np.median(np.abs(residuals - np.median(residuals)))),
		float(np.median(reference_errors)),
	)
	synthetic_errors = rng.choice(reference_errors, size=len(dates), replace=True)
	synthetic_sigmas = np.sqrt(synthetic_errors**2 + scatter**2)
	synthetic_velocities = model(dates) + rng.normal(0.0, synthetic_sigmas)
	drawn_parameters = {
		"period_days": float(period),
		"semiamplitude_m_per_s": float(semiamplitude),
		"conjunction_bjd": float(conjunction),
		"empirical_scatter_m_per_s": float(scatter),
	}
	return synthetic_velocities, synthetic_errors, drawn_parameters


def write_exoarchive_file(
	path: Path,
	system: str,
	dates: np.ndarray,
	velocities: np.ndarray,
	errors: np.ndarray,
	drawn_parameters: dict[str, float],
) -> None:
	"""Write synthetic points using the local ExoArchive table convention."""
	lines = [
		f'\\STAR_ID = "{system}"',
		'\\DATA_CATEGORY = "Synthetic Planet Radial Velocity Curve"',
		f'\\NUMBER_OF_POINTS = "{len(dates)}"',
		'\\TIME_REFERENCE_FRAME = "BJD-TDB"',
		'\\DATE_UNITS = "days"',
		'\\VALUE_UNITS = "m/s"',
		'\\COLUMN_RADIAL_VELOCITY = "Radial velocity relative to barycenter"',
		'\\COLUMN_RADIAL_VELOCITY_UNCERTAINTY = "Synthetic measurement uncertainty"',
		f'\\SYNTHETIC_PERIOD_DAYS = "{drawn_parameters["period_days"]:.12g}"',
		f'\\SYNTHETIC_SEMIAMPLITUDE_M_S = "{drawn_parameters["semiamplitude_m_per_s"]:.12g}"',
		f'\\SYNTHETIC_CONJUNCTION_BJD = "{drawn_parameters["conjunction_bjd"]:.12f}"',
		f'\\SYNTHETIC_SCATTER_M_S = "{drawn_parameters["empirical_scatter_m_per_s"]:.12g}"',
		'|          BJD | Radial_Velocity | Radial_Velocity_Uncertainty |',
		'|        double |          double |                      double |',
		'|          days |             m/s |                         m/s |',
	]
	lines.extend(
		f"{date:15.8f} {velocity:16.8f} {error:28.8f}"
		for date, velocity, error in zip(dates, velocities, errors)
	)
	path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate synthetic ExoArchive-style RV points."
	)
	parser.add_argument("system", help="System name matching a saved fit JSON.")
	parser.add_argument(
		"dates",
		nargs="+",
		help="Observation dates as ISO strings or BJD values.",
	)
	parser.add_argument("--seed", type=int, default=20260924)
	args = parser.parse_args()

	_, parameters = find_parameters(args.system)
	dates = np.asarray([parse_date(value) for value in args.dates], dtype=float)
	velocities, errors, drawn_parameters = draw_synthetic_points(
		parameters,
		dates,
		*load_reference_observations(args.system),
		np.random.default_rng(args.seed),
	)
	rv_io.SYNTHETICS_DATABASE.mkdir(parents=True, exist_ok=True)
	output_path = rv_io.SYNTHETICS_DATABASE / f"{args.system}_synthetic_RV.tbl"
	write_exoarchive_file(
		output_path, args.system, dates, velocities, errors, drawn_parameters
	)
	print(f"Wrote {len(dates)} synthetic RV points to {output_path}")
	print(json.dumps(drawn_parameters, indent=2))


if __name__ == "__main__":
	main()
