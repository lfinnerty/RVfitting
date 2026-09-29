#!/usr/bin/env python3
"""Plot combined radial velocities for a star and fit its orbit."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import least_squares

import rv_io


PLOTS_DIRECTORY = Path(__file__).resolve().parent / "plots"
SOURCE_COLORS = {
	"Teklu": "tab:blue",
	"ExoArchive": "tab:orange",
	"Fulton": "tab:green",
	"Hebrard": "tab:red",
	"HARPS": "tab:purple",
	"Synthetic": "tab:brown",
}
OBSERVED_SOURCES = ("Teklu", "ExoArchive", "Fulton", "Hebrard", "HARPS")
PERIOD_RANGE = (1.2, np.nextafter(8.0, 1.2))  # days
FALSE_ALARM_PROBABILITY = 0.001
TARGET_DATE = "2027-07-01"
TARGET_BJD = 2_461_587.5  # 2027-07-01 00:00 UTC


def calculate_periodogram(
	bjd: np.ndarray, rv: np.ndarray, rv_error: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
	"""Return trial periods, weighted Lomb-Scargle powers, the peak period, and
	the Baluev power threshold for ``FALSE_ALARM_PROBABILITY`` over the same range."""
	periods = np.geomspace(*PERIOD_RANGE, 40_000)
	periodogram = LombScargle(bjd, rv, rv_error)
	power = periodogram.power(1 / periods)
	threshold = periodogram.false_alarm_level(
		FALSE_ALARM_PROBABILITY,
		minimum_frequency=1 / PERIOD_RANGE[1],
		maximum_frequency=1 / PERIOD_RANGE[0],
		method="baluev",
	)
	return periods, power, periods[np.argmax(power)], float(threshold)


def source_offset_design(source_labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
	"""Return fitted source offsets with Teklu or ExoArchive as the zero point."""
	sources = list(dict.fromkeys(source_labels.tolist()))
	if "Teklu" in sources:
		reference = "Teklu"
	elif "ExoArchive" in sources:
		reference = "ExoArchive"
	else:
		reference = sources[0]
	offset_sources = [source for source in sources if source != reference]
	design = np.column_stack(
		[(source_labels == source).astype(float) for source in offset_sources]
	) if offset_sources else np.empty((len(source_labels), 0))
	return design, offset_sources


def sinusoid_design(
	phase: np.ndarray, offset_design: np.ndarray | None = None
) -> np.ndarray:
	"""Return the [1, sin, cos, source offsets] design matrix for orbital phases."""
	angle = 2 * np.pi * phase
	columns = [np.ones_like(phase), np.sin(angle), np.cos(angle)]
	if offset_design is not None:
		columns.append(offset_design)
	return np.column_stack(columns)


def _fit_covariance(fit) -> tuple[np.ndarray, float]:
	"""Return the parameter covariance and reduced chi-squared of a least-squares fit."""
	reduced_chi_squared = np.sum(fit.fun**2) / (len(fit.fun) - len(fit.x))
	return reduced_chi_squared * np.linalg.pinv(fit.jac.T @ fit.jac), reduced_chi_squared


def fit_phase_curve(
	phase: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	offset_design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Fit a weighted sinusoid and return its coefficients and covariance."""
	design = sinusoid_design(phase, offset_design)
	weighted_design = design / rv_error[:, np.newaxis]
	coefficients = np.linalg.lstsq(weighted_design, rv / rv_error, rcond=None)[0]
	residuals = (rv - design @ coefficients) / rv_error
	residual_variance = np.sum(residuals**2) / (len(rv) - design.shape[1])
	covariance = residual_variance * np.linalg.pinv(weighted_design.T @ weighted_design)
	return coefficients, covariance


def fit_period(
	bjd: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	initial_period: float,
	offset_design: np.ndarray,
) -> tuple[float, float]:
	"""Refine the period and return its 1-sigma weighted-fit uncertainty."""
	elapsed = bjd - bjd[0]
	initial_coefficients, _ = fit_phase_curve(
		elapsed / initial_period, rv, rv_error, offset_design
	)

	def residuals(parameters: np.ndarray) -> np.ndarray:
		design = sinusoid_design(elapsed / parameters[-1], offset_design)
		return (rv - design @ parameters[:-1]) / rv_error

	coefficient_count = len(initial_coefficients)
	fit = least_squares(
		residuals,
		np.append(initial_coefficients, initial_period),
		bounds=(
			[-np.inf] * coefficient_count + [PERIOD_RANGE[0]],
			[np.inf] * coefficient_count + [PERIOD_RANGE[1]],
		),
		x_scale="jac",
	)
	covariance, _ = _fit_covariance(fit)
	return fit.x[-1], np.sqrt(max(covariance[-1, -1], 0.0))


def solve_kepler(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
	"""Solve Kepler's equation for eccentric anomaly."""
	eccentric_anomaly = mean_anomaly.copy()
	for _ in range(20):
		eccentric_anomaly -= (
			eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
		) / (1 - eccentricity * np.cos(eccentric_anomaly))
	return eccentric_anomaly


def calculate_keplerian_rv(
	phase: np.ndarray, parameters: np.ndarray, offset_design: np.ndarray
) -> np.ndarray:
	"""Evaluate the Keplerian RV model at the supplied orbital phases."""
	gamma, semiamplitude, eccentricity, omega, periapsis_phase = parameters[:5]
	mean_anomaly = 2 * np.pi * (phase - periapsis_phase)
	eccentric_anomaly = solve_kepler(mean_anomaly, eccentricity)
	true_anomaly = 2 * np.arctan2(
		np.sqrt(1 + eccentricity) * np.sin(eccentric_anomaly / 2),
		np.sqrt(1 - eccentricity) * np.cos(eccentric_anomaly / 2),
	)
	model = gamma + semiamplitude * (
		np.cos(true_anomaly + omega) + eccentricity * np.cos(omega)
	)
	return model + offset_design @ parameters[5:]


def fit_keplerian(
	phase: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	offset_design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
	"""Fit a Keplerian RV model; return parameters, covariance, and reduced chi2."""
	circular, _ = fit_phase_curve(phase, rv, rv_error, offset_design)
	initial_parameters = np.concatenate([
		[
			circular[0],
			np.hypot(circular[1], circular[2]),
			0.05,
			0.0,
			(np.arctan2(circular[1], circular[2]) / (2 * np.pi)) % 1.0,
		],
		circular[3:],
	])

	def residuals(parameters: np.ndarray) -> np.ndarray:
		return (rv - calculate_keplerian_rv(phase, parameters, offset_design)) / rv_error

	offset_count = offset_design.shape[1]
	fit = least_squares(
		residuals,
		initial_parameters,
		bounds=(
			[-np.inf, 0.0, 0.0, -np.pi, 0.0] + [-np.inf] * offset_count,
			[np.inf, np.inf, 0.95, np.pi, 1.0] + [np.inf] * offset_count,
		),
		x_scale="jac",
		max_nfev=2_000,
	)
	covariance, reduced_chi_squared = _fit_covariance(fit)
	return fit.x, covariance, reduced_chi_squared


def sinusoid_parameters(
	coefficients: np.ndarray,
	covariance: np.ndarray,
	reference_bjd: float,
	period: float,
) -> tuple[float, float, float, float]:
	"""Return semiamplitude, its uncertainty, and the decreasing RV zero-crossing
	(conjunction) BJD with its fit-only uncertainty."""
	sine, cosine = coefficients[1:3]
	var_sine, var_cosine, cov_sine_cosine = covariance[1, 1], covariance[2, 2], covariance[1, 2]
	amplitude_squared = sine**2 + cosine**2
	semiamplitude = np.sqrt(amplitude_squared)
	semiamplitude_uncertainty = np.sqrt(
		(sine**2 * var_sine + cosine**2 * var_cosine + 2 * sine * cosine * cov_sine_cosine)
		/ amplitude_squared
	)

	phase_angle = np.arctan2(cosine, sine)
	conjunction_bjd = reference_bjd + ((0.5 - phase_angle / (2 * np.pi)) % 1.0) * period
	phase_angle_variance = (
		cosine**2 * var_sine + sine**2 * var_cosine - 2 * sine * cosine * cov_sine_cosine
	) / amplitude_squared**2
	conjunction_uncertainty = period * np.sqrt(phase_angle_variance) / (2 * np.pi)
	return semiamplitude, semiamplitude_uncertainty, conjunction_bjd, conjunction_uncertainty


def calculate_phase_at_date(
	target_bjd: float,
	conjunction_bjd: float,
	conjunction_uncertainty: float,
	period: float,
	period_uncertainty: float,
) -> tuple[float, float]:
	"""Return orbital phase and propagated uncertainty at a target BJD."""
	elapsed_time = target_bjd - conjunction_bjd
	phase = (elapsed_time / period) % 1.0
	phase_uncertainty = np.sqrt(
		(conjunction_uncertainty / period) ** 2
		+ (elapsed_time * period_uncertainty / period**2) ** 2
	)
	return phase, phase_uncertainty


def exclude_orbit_fit_outliers(
	phase: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	offset_design: np.ndarray,
) -> np.ndarray:
	"""Return a mask excluding obvious 5-sigma residual outliers."""
	design = sinusoid_design(phase, offset_design)
	mask = np.ones(len(rv), dtype=bool)
	for _ in range(3):
		coefficients, _ = fit_phase_curve(
			phase[mask], rv[mask], rv_error[mask], offset_design[mask]
		)
		residuals = rv - design @ coefficients
		center = np.median(residuals[mask])
		robust_scale = 1.4826 * np.median(np.abs(residuals[mask] - center))
		scale = max(robust_scale, np.median(rv_error[mask]))
		new_mask = np.abs(residuals - center) <= 5 * scale
		if np.array_equal(new_mask, mask):
			break
		mask = new_mask
	return mask


@dataclass
class SystemFit:
	"""Fit summary (JSON-ready ``parameters``) plus arrays needed for plotting."""

	parameters: dict
	periods: np.ndarray
	power: np.ndarray
	phase: np.ndarray
	offset_corrected_rv: np.ndarray
	orbit_fit_mask: np.ndarray
	sinusoid_coefficients: np.ndarray


def fit_system(
	bjd: np.ndarray, rv: np.ndarray, rv_error: np.ndarray, source_labels: np.ndarray
) -> SystemFit:
	"""Find the period and fit sinusoidal and Keplerian orbits to combined RVs.

	Orbital phases are measured from ``bjd[0]``. Offsets are defined once from all
	source labels so masked fits keep the same offset columns.
	"""
	offset_design, offset_sources = source_offset_design(source_labels)
	periods, power, peak_period, false_alarm_threshold = calculate_periodogram(
		bjd, rv, rv_error
	)
	period, period_uncertainty = fit_period(bjd, rv, rv_error, peak_period, offset_design)
	phase = ((bjd - bjd[0]) / period) % 1.0

	mask = exclude_orbit_fit_outliers(phase, rv, rv_error, offset_design)
	coefficients, covariance = fit_phase_curve(
		phase[mask], rv[mask], rv_error[mask], offset_design[mask]
	)
	semiamplitude, semiamplitude_uncertainty, conjunction_bjd, conjunction_uncertainty = (
		sinusoid_parameters(coefficients, covariance, bjd[0], period)
	)
	target_phase, target_phase_uncertainty = calculate_phase_at_date(
		TARGET_BJD, conjunction_bjd, conjunction_uncertainty, period, period_uncertainty
	)

	keplerian, keplerian_covariance, reduced_chi_squared = fit_keplerian(
		phase[mask], rv[mask], rv_error[mask], offset_design[mask]
	)
	eccentricity = keplerian[2]
	eccentricity_uncertainty = np.sqrt(keplerian_covariance[2, 2])
	omega_degrees = np.degrees(keplerian[3])
	omega_uncertainty_degrees = np.degrees(np.sqrt(keplerian_covariance[3, 3]))
	periapsis_bjd = bjd[0] + keplerian[4] * period
	periapsis_uncertainty = period * np.sqrt(keplerian_covariance[4, 4])
	if eccentricity < eccentricity_uncertainty:
		eccentricity = 0.0
		omega_degrees = omega_uncertainty_degrees = np.nan
		periapsis_bjd = periapsis_uncertainty = np.nan

	parameters = {
		"measurement_count": len(bjd),
		"orbit_fit_excluded_count": np.count_nonzero(~mask),
		"periodogram_peak_period_days": peak_period,
		"periodogram_false_alarm_probability": FALSE_ALARM_PROBABILITY,
		"periodogram_false_alarm_threshold": false_alarm_threshold,
		"period_days": period,
		"period_uncertainty_days": period_uncertainty,
		"sinusoidal_gamma_m_per_s": coefficients[0],
		"sinusoidal_semiamplitude_m_per_s": semiamplitude,
		"sinusoidal_semiamplitude_uncertainty_m_per_s": semiamplitude_uncertainty,
		"source_offsets_m_per_s": dict(zip(offset_sources, coefficients[3:])),
		"conjunction_bjd": conjunction_bjd,
		"conjunction_uncertainty_days": conjunction_uncertainty,
		f"phase_on_{TARGET_DATE}": target_phase,
		f"phase_uncertainty_on_{TARGET_DATE}": target_phase_uncertainty,
		f"phase_uncertainty_hours_on_{TARGET_DATE}": target_phase_uncertainty * period * 24,
		"keplerian_gamma_m_per_s": keplerian[0],
		"keplerian_semiamplitude_m_per_s": keplerian[1],
		"keplerian_eccentricity": eccentricity,
		"keplerian_eccentricity_uncertainty": eccentricity_uncertainty,
		"keplerian_omega_degrees": omega_degrees,
		"keplerian_omega_uncertainty_degrees": omega_uncertainty_degrees,
		"keplerian_periapsis_bjd": periapsis_bjd,
		"keplerian_periapsis_uncertainty_days": periapsis_uncertainty,
		"orbit_fit_reduced_chi_squared": reduced_chi_squared,
	}
	return SystemFit(
		parameters=_to_json(parameters),
		periods=periods,
		power=power,
		phase=phase,
		offset_corrected_rv=rv - offset_design @ coefficients[3:],
		orbit_fit_mask=mask,
		sinusoid_coefficients=coefficients,
	)


def _to_json(value):
	"""Convert numpy scalars to JSON types, mapping NaN to None."""
	if isinstance(value, dict):
		return {key: _to_json(item) for key, item in value.items()}
	if isinstance(value, np.integer):
		return int(value)
	if isinstance(value, (float, np.floating)):
		return None if np.isnan(value) else float(value)
	return value


def _errorbar_by_source(
	axis, x, y, yerr, source_labels, mask=None, label_suffix="", **style
) -> None:
	"""Draw one colored errorbar series per RV source."""
	for source, color in SOURCE_COLORS.items():
		source_mask = source_labels == source
		if mask is not None:
			source_mask &= mask
		if np.any(source_mask):
			axis.errorbar(
				x[source_mask],
				y[source_mask],
				yerr=yerr[source_mask],
				fmt="o",
				capsize=2,
				color=color,
				label=f"{source}{label_suffix}",
				**style,
			)


def plot_fit(
	star: str,
	bjd: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	source_labels: np.ndarray,
	fit: SystemFit,
) -> plt.Figure:
	"""Plot the RV time series, periodogram, and phase-folded fit."""
	parameters = fit.parameters
	period = parameters["period_days"]
	period_label = f"{period:.6g} +/- {parameters['period_uncertainty_days']:.6g}"
	semiamplitude = parameters["sinusoidal_semiamplitude_m_per_s"]
	semiamplitude_uncertainty = parameters["sinusoidal_semiamplitude_uncertainty_m_per_s"]

	figure, axes = plt.subplots(3, 1, sharex=False, figsize=(8, 10))
	_errorbar_by_source(axes[0], bjd - 2_450_000, rv, rv_error, source_labels)
	axes[0].set_xlabel("BJD - 2450000")
	axes[0].set_ylabel("Radial velocity (m/s)")
	axes[0].set_title(f"{star} ({len(bjd)} measurements)")
	axes[0].grid(alpha=0.3)
	axes[0].legend()
	jd_to_matplotlib_date = 2_450_000 - 2_440_587.5
	calendar_axis = axes[0].secondary_xaxis(
		"top",
		functions=(
			lambda x: x + jd_to_matplotlib_date,
			lambda date_number: date_number - jd_to_matplotlib_date,
		),
	)
	calendar_axis.set_xlabel("Calendar date (UTC)")
	calendar_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

	axes[1].plot(fit.periods, fit.power)
	axes[1].axvline(period, color="tab:red", linestyle="--")
	axes[1].axhline(
		parameters["periodogram_false_alarm_threshold"],
		color="tab:purple",
		linestyle=":",
		label=f"{FALSE_ALARM_PROBABILITY:.1%} false-alarm threshold",
	)
	axes[1].set_xscale("log")
	axes[1].set_xlabel("Period (days)")
	axes[1].set_ylabel("Lomb-Scargle power")
	axes[1].set_title(f"Best-fit period = {period_label} days")
	axes[1].grid(alpha=0.3)
	axes[1].legend()

	mask = fit.orbit_fit_mask
	_errorbar_by_source(
		axes[2], fit.phase, fit.offset_corrected_rv, rv_error, source_labels, mask
	)
	_errorbar_by_source(
		axes[2], fit.phase, fit.offset_corrected_rv, rv_error, source_labels, ~mask,
		label_suffix=" (excluded)", alpha=0.25,
	)
	fit_phase = np.linspace(0, 1, 500)
	axes[2].plot(
		fit_phase,
		sinusoid_design(fit_phase) @ fit.sinusoid_coefficients[:3],
		color="tab:red",
		label=(
			f"Sinusoidal fit (K = {semiamplitude:.4g} +/- "
			f"{semiamplitude_uncertainty:.3g} m/s)"
		),
	)
	axes[2].axvline(
		((parameters["conjunction_bjd"] - bjd[0]) / period) % 1.0,
		color="tab:green",
		linestyle="--",
		label="Best-fit conjunction",
	)
	axes[2].set_xlabel("Orbital phase")
	axes[2].set_ylabel("Radial velocity (m/s)")
	axes[2].set_title(
		f"RV folded on {period_label}-day period; "
		f"K = {semiamplitude:.6g} +/- {semiamplitude_uncertainty:.6g} m/s"
	)
	axes[2].set_xlim(0, 1)
	axes[2].grid(alpha=0.3)
	axes[2].legend()
	figure.tight_layout()
	return figure


def print_fit_summary(star: str, parameters: dict) -> None:
	"""Print the main fitted quantities."""
	def value(key: str, digits: str = ".6g") -> str:
		number = parameters[key]
		return "nan" if number is None else format(number, digits)

	if parameters["orbit_fit_excluded_count"]:
		print(
			f"{star}: excluding {parameters['orbit_fit_excluded_count']}"
			" obvious outlier(s) from orbit fit"
		)
	lines = (
		f"best-fit period = {value('period_days')} +/- {value('period_uncertainty_days')} days",
		f"best-fit RV semiamplitude = {value('sinusoidal_semiamplitude_m_per_s')}"
		f" +/- {value('sinusoidal_semiamplitude_uncertainty_m_per_s')} m/s",
		f"primary-transit conjunction = BJD {value('conjunction_bjd', '.6f')}"
		f" +/- {value('conjunction_uncertainty_days', '.6f')}",
		f"orbital phase on {TARGET_DATE} = {value(f'phase_on_{TARGET_DATE}', '.6f')}"
		f" +/- {value(f'phase_uncertainty_on_{TARGET_DATE}', '.6f')} cycles"
		f" (+/- {value(f'phase_uncertainty_hours_on_{TARGET_DATE}', '.6f')} hours)",
		f"eccentricity = {value('keplerian_eccentricity')}"
		f" +/- {value('keplerian_eccentricity_uncertainty')}",
		f"longitude of periastron = {value('keplerian_omega_degrees')}"
		f" +/- {value('keplerian_omega_uncertainty_degrees')} degrees",
		f"time of periapsis passage = BJD {value('keplerian_periapsis_bjd', '.6f')}"
		f" +/- {value('keplerian_periapsis_uncertainty_days', '.6f')}",
		f"orbit-fit reduced chi2 = {value('orbit_fit_reduced_chi_squared')}",
	)
	for line in lines:
		print(f"{star}: {line}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot combined radial velocities for a star and fit its orbit."
	)
	parser.add_argument("star", help="Catalog or SIMBAD name, for example HD10700")
	parser.add_argument(
		"--database",
		type=Path,
		default=rv_io.TEKLU_DATABASE,
		help=f"Path to tablea1_Teklu.dat (default: {rv_io.TEKLU_DATABASE})",
	)
	source_names = {source.casefold(): source for source in OBSERVED_SOURCES}
	parser.add_argument(
		"--source",
		choices=("all", *source_names),
		default="all",
		help="RV source to load (default: all)",
	)
	parser.add_argument(
		"--synthetics",
		action="store_true",
		help="Also load matching synthetic RV points from RVdatabases/Synthetics/.",
	)
	args = parser.parse_args()

	sources = list(OBSERVED_SOURCES) if args.source == "all" else [source_names[args.source]]
	if "Teklu" in sources and not args.database.is_file():
		parser.error(f"Database file not found: {args.database}")
	if args.synthetics:
		sources.append("Synthetic")

	datasets = rv_io.load_datasets(args.star, sources, args.database)
	raw_measurement_count = sum(len(data[0]) for _, data in datasets)
	try:
		bjd, rv, rv_error, source_labels = rv_io.combine_rv_data(datasets)
	except ValueError as error:
		parser.error(str(error))
	print(
		f"{args.star}: loaded {len(bjd)} RV measurements from {args.source} source(s)"
		f" ({raw_measurement_count - len(bjd)} duplicate(s) removed)"
	)

	fit = fit_system(bjd, rv, rv_error, source_labels)
	print_fit_summary(args.star, fit.parameters)
	fit_parameters = {"input_star": args.star, "source": args.source, **fit.parameters}

	figure = plot_fit(args.star, bjd, rv, rv_error, source_labels, fit)
	PLOTS_DIRECTORY.mkdir(exist_ok=True)
	output_path = PLOTS_DIRECTORY / f"{args.star}_rv_fit_plot.png"
	figure.savefig(str(output_path), dpi=150)
	json_path = PLOTS_DIRECTORY / f"{args.star}_rv_fit_parameters.json"
	json_path.write_text(json.dumps(fit_parameters, indent=2) + "\n", encoding="utf-8")
	print(f"Saved plot to {output_path}")
	print(f"Saved fit parameters to {json_path}")
	plt.show()


if __name__ == "__main__":
	main()
