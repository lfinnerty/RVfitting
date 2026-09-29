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
from scipy.optimize import brentq, least_squares

import rv_io


PLOTS_DIRECTORY = Path(__file__).resolve().parent / "plots"
SYNTHETIC_SUFFIX = "_with_synthetics"
OBSERVED_SOURCES = ("Teklu", "ExoArchive", "CLS", "HARPS", "HARPS2020", "SOPHIE", "Hebrard")
PERIOD_RANGE = (1.2, np.nextafter(8.0, 1.2))  # days
FALSE_ALARM_PROBABILITY = 0.001
TARGET_DATE = "2027-07-01"
TARGET_BJD = 2_461_587.5  # 2027-07-01 00:00 UTC
# Orbit parameter layout used by orbit_rv; source offsets follow.
ORBIT_GAMMA, ORBIT_SEMIAMPLITUDE, ORBIT_PERIOD, ORBIT_CONJUNCTION, ORBIT_H, ORBIT_K = range(6)
ORBIT_OFFSETS = 6
MAX_ECCENTRICITY = 0.95
ECCENTRICITY_SIGNIFICANCE = 2.45  # Lucy & Sweeney (1971) 5% false-alarm level
JITTER_ITERATIONS = 8
MIN_JITTER_POINTS = 8  # smaller offset groups take the median jitter of the others


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


def solve_kepler(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
	"""Solve Kepler's equation for the eccentric anomaly (Newton, Danby start)."""
	mean_anomaly = np.mod(mean_anomaly, 2 * np.pi)
	eccentric_anomaly = mean_anomaly + 0.85 * eccentricity * np.sign(np.sin(mean_anomaly))
	for _ in range(30):
		eccentric_anomaly -= (
			eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
		) / (1 - eccentricity * np.cos(eccentric_anomaly))
	return eccentric_anomaly


def eccentricity_and_omega(parameters: np.ndarray) -> tuple[float, float]:
	"""Return (e, omega) from the sqrt(e)cos(omega), sqrt(e)sin(omega) parameters."""
	h, k = parameters[ORBIT_H], parameters[ORBIT_K]
	return min(h * h + k * k, MAX_ECCENTRICITY), float(np.arctan2(k, h))


def periapsis_time(conjunction: float, period: float, eccentricity: float, omega: float) -> float:
	"""Return the periastron time preceding a transit conjunction (f = pi/2 - omega)."""
	true_anomaly = np.pi / 2 - omega
	eccentric_anomaly = 2 * np.arctan(
		np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(true_anomaly / 2)
	)
	mean_anomaly = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)
	return conjunction - mean_anomaly * period / (2 * np.pi)


def orbit_rv(bjd: np.ndarray, parameters: np.ndarray, offset_design: np.ndarray) -> np.ndarray:
	"""Evaluate the Keplerian RV model; e = 0 gives -K sin(2 pi (t - Tc) / P).

	``parameters`` are [gamma, K, P, Tc, sqrt(e)cos(omega), sqrt(e)sin(omega),
	source offsets...], where Tc is the transit conjunction (f + omega = pi/2).
	"""
	gamma, semiamplitude, period, conjunction = parameters[:4]
	eccentricity, omega = eccentricity_and_omega(parameters)
	mean_anomaly = 2 * np.pi * (bjd - periapsis_time(conjunction, period, eccentricity, omega)) / period
	eccentric_anomaly = solve_kepler(mean_anomaly, eccentricity)
	true_anomaly = 2 * np.arctan2(
		np.sqrt(1 + eccentricity) * np.sin(eccentric_anomaly / 2),
		np.sqrt(1 - eccentricity) * np.cos(eccentric_anomaly / 2),
	)
	model = gamma + semiamplitude * (np.cos(true_anomaly + omega) + eccentricity * np.cos(omega))
	return model + offset_design @ parameters[ORBIT_OFFSETS:]


def saved_orbit(
	parameters: dict,
	period: float | None = None,
	semiamplitude: float | None = None,
	conjunction: float | None = None,
) -> np.ndarray:
	"""Return orbit_rv parameters (no offsets) for a saved fit's adopted orbit.

	``period``, ``semiamplitude`` and ``conjunction`` override the saved values.
	"""
	eccentricity = parameters["eccentricity"] or 0.0
	omega = np.radians(parameters["omega_degrees"] or 0.0)
	return np.array([
		parameters["gamma_m_per_s"],
		parameters["semiamplitude_m_per_s"] if semiamplitude is None else semiamplitude,
		parameters["period_days"] if period is None else period,
		parameters["conjunction_bjd"] if conjunction is None else conjunction,
		np.sqrt(eccentricity) * np.cos(omega),
		np.sqrt(eccentricity) * np.sin(omega),
	])


def estimate_jitter(
	residuals: np.ndarray, rv_error: np.ndarray, source_labels: np.ndarray
) -> dict[str, float]:
	"""Per-label jitter s with mean(r^2 / (err^2 + s^2)) = 1 (0 if already <= 1).

	Labels with fewer than ``MIN_JITTER_POINTS`` points take the median of the others.
	"""
	jitter = {}
	for label in dict.fromkeys(source_labels):
		in_label = source_labels == label
		if in_label.sum() < MIN_JITTER_POINTS:
			continue
		r2, e2 = residuals[in_label] ** 2, rv_error[in_label] ** 2
		excess = lambda s: np.mean(r2 / (e2 + s * s)) - 1
		jitter[label] = 0.0 if excess(0.0) <= 0 else brentq(excess, 0.0, np.sqrt(r2.max()) + 1)
	fallback = float(np.median(list(jitter.values()))) if jitter else 0.0
	return {label: jitter.get(label, fallback) for label in dict.fromkeys(source_labels)}


def fit_orbit(
	bjd: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	source_labels: np.ndarray,
	offset_design: np.ndarray,
	initial: np.ndarray,
	eccentric: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float]:
	"""Fit the orbit with per-label jitter, iterating jitter and fit to convergence.

	Returns parameters, their covariance (fixed parameters have zero variance),
	the jitter by label, and the reduced chi-squared including jitter.
	"""
	# Fit times relative to the initial conjunction so finite-difference steps on
	# Tc are small; the model depends only on t - Tc.
	epoch = initial[ORBIT_CONJUNCTION]
	bjd = bjd - epoch
	initial = initial.copy()
	initial[ORBIT_CONJUNCTION] = 0.0
	free = np.ones(len(initial), dtype=bool)
	if not eccentric:
		free[[ORBIT_H, ORBIT_K]] = False
	root_max_e = np.sqrt(MAX_ECCENTRICITY)
	lower = np.array([-np.inf, 0.0, PERIOD_RANGE[0], -np.inf, -root_max_e, -root_max_e]
		+ [-np.inf] * offset_design.shape[1])
	upper = np.array([np.inf, np.inf, PERIOD_RANGE[1], np.inf, root_max_e, root_max_e]
		+ [np.inf] * offset_design.shape[1])
	starts = [initial]
	if eccentric:  # a few periastron orientations at e = 0.09 avoid local minima
		starts = [
			np.concatenate([initial[:ORBIT_H], [0.3 * np.cos(angle), 0.3 * np.sin(angle)], initial[ORBIT_OFFSETS:]])
			for angle in np.radians([45, 135, 225, 315])
		]
	jitter = {label: 0.0 for label in dict.fromkeys(source_labels)}
	parameters = starts[0]
	for iteration in range(JITTER_ITERATIONS):
		sigma = np.hypot(rv_error, np.array([jitter[label] for label in source_labels]))

		def residuals(x: np.ndarray) -> np.ndarray:
			full = parameters_template.copy()
			full[free] = x
			return (rv - orbit_rv(bjd, full, offset_design)) / sigma

		best = None
		for start in (starts if iteration == 0 else [parameters]):
			parameters_template = start.copy()
			fit = least_squares(
				residuals, np.clip(start[free], lower[free], upper[free]),
				bounds=(lower[free], upper[free]), x_scale="jac", max_nfev=5_000,
			)
			if best is None or fit.cost < best[0].cost:
				best = (fit, start.copy())
		fit, parameters_template = best
		parameters = parameters_template.copy()
		parameters[free] = fit.x
		new_jitter = estimate_jitter(
			rv - orbit_rv(bjd, parameters, offset_design), rv_error, source_labels
		)
		converged = all(abs(new_jitter[l] - jitter[l]) < 0.01 * (jitter[l] + 0.1) for l in jitter)
		jitter = new_jitter
		if converged:
			break
	sigma = np.hypot(rv_error, np.array([jitter[label] for label in source_labels]))
	parameters_template = parameters.copy()
	final_residuals = residuals(parameters[free])
	jacobian = _central_jacobian(residuals, parameters[free])
	reduced_chi_squared = np.sum(final_residuals**2) / (len(rv) - free.sum())
	covariance = np.zeros((len(parameters), len(parameters)))
	covariance[np.ix_(free, free)] = reduced_chi_squared * np.linalg.pinv(jacobian.T @ jacobian)
	parameters[ORBIT_CONJUNCTION] += epoch
	return parameters, covariance, jitter, reduced_chi_squared


def _central_jacobian(function, x: np.ndarray) -> np.ndarray:
	"""Central-difference Jacobian of a vector function at ``x``."""
	columns = []
	for index in range(len(x)):
		step = 1e-6 * max(abs(x[index]), 1e-3)
		up, down = x.copy(), x.copy()
		up[index] += step
		down[index] -= step
		columns.append((function(up) - function(down)) / (2 * step))
	return np.column_stack(columns)


def _propagate(function, parameters: np.ndarray, covariance: np.ndarray) -> tuple[float, float]:
	"""Return f(parameters) and its linearly propagated 1-sigma uncertainty."""
	value = function(parameters)
	gradient = np.zeros(len(parameters))
	for index in np.flatnonzero(np.diag(covariance) > 0):
		step = 1e-6 * max(abs(parameters[index]), 1.0) if index != ORBIT_CONJUNCTION else 1e-6
		shifted = parameters.copy()
		shifted[index] += step
		gradient[index] = (function(shifted) - value) / step
	return value, float(np.sqrt(max(gradient @ covariance @ gradient, 0.0)))


def summarize_orbit(
	parameters: np.ndarray, covariance: np.ndarray, pivot_bjd: float
) -> tuple[np.ndarray, np.ndarray, dict]:
	"""Move Tc to the cycle nearest ``pivot_bjd`` and derive reported quantities.

	Shifting Tc by whole periods is an exact reparametrization, so the covariance is
	transformed rather than refitted. The target-date phase uses the full Tc-P covariance.
	"""
	period = parameters[ORBIT_PERIOD]
	cycles = np.round((pivot_bjd - parameters[ORBIT_CONJUNCTION]) / period)
	transform = np.eye(len(parameters))
	transform[ORBIT_CONJUNCTION, ORBIT_PERIOD] = cycles
	parameters = transform @ parameters
	covariance = transform @ covariance @ transform.T

	def target_phase(p):
		return (TARGET_BJD - p[ORBIT_CONJUNCTION]) / p[ORBIT_PERIOD]

	phase, phase_uncertainty = _propagate(target_phase, parameters, covariance)
	eccentricity, e_uncertainty = _propagate(lambda p: eccentricity_and_omega(p)[0], parameters, covariance)
	_, omega = eccentricity_and_omega(parameters)
	eccentric = e_uncertainty > 0
	if eccentric:
		# Wrap-safe omega uncertainty: propagate the angle relative to its best value.
		_, omega_uncertainty = _propagate(
			lambda p: (eccentricity_and_omega(p)[1] - omega + np.pi) % (2 * np.pi) - np.pi,
			parameters, covariance,
		)
		periapsis, periapsis_uncertainty = _propagate(
			lambda p: periapsis_time(p[ORBIT_CONJUNCTION], p[ORBIT_PERIOD], *eccentricity_and_omega(p)),
			parameters, covariance,
		)
	summary = {
		"period_days": period,
		"period_uncertainty_days": np.sqrt(covariance[ORBIT_PERIOD, ORBIT_PERIOD]),
		"conjunction_bjd": parameters[ORBIT_CONJUNCTION],
		"conjunction_uncertainty_days": np.sqrt(covariance[ORBIT_CONJUNCTION, ORBIT_CONJUNCTION]),
		"semiamplitude_m_per_s": parameters[ORBIT_SEMIAMPLITUDE],
		"semiamplitude_uncertainty_m_per_s": np.sqrt(covariance[ORBIT_SEMIAMPLITUDE, ORBIT_SEMIAMPLITUDE]),
		"gamma_m_per_s": parameters[ORBIT_GAMMA],
		"eccentricity": eccentricity,
		"eccentricity_uncertainty": e_uncertainty if eccentric else np.nan,
		"omega_degrees": np.degrees(omega) if eccentric else np.nan,
		"omega_uncertainty_degrees": np.degrees(omega_uncertainty) if eccentric else np.nan,
		"periapsis_bjd": periapsis if eccentric else np.nan,
		"periapsis_uncertainty_days": periapsis_uncertainty if eccentric else np.nan,
		f"phase_on_{TARGET_DATE}": phase % 1.0,
		f"phase_uncertainty_on_{TARGET_DATE}": phase_uncertainty,
		f"phase_uncertainty_hours_on_{TARGET_DATE}": phase_uncertainty * period * 24,
	}
	return parameters, covariance, summary


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
	orbit_parameters: np.ndarray  # adopted model, see orbit_rv
	offset_corrected_rv: np.ndarray
	orbit_fit_mask: np.ndarray


def fit_system(
	bjd: np.ndarray, rv: np.ndarray, rv_error: np.ndarray, source_labels: np.ndarray
) -> SystemFit:
	"""Find the period and fit circular and eccentric orbits with per-label jitter.

	A circular sinusoid (periodogram peak, then refined period) initializes both
	fits and flags outliers. The period, K, transit conjunction, and offsets are then
	fitted jointly, once with e = 0 and once with e free. The eccentric fit is
	adopted only if e exceeds ``ECCENTRICITY_SIGNIFICANCE`` times its uncertainty
	(Lucy & Sweeney 1971). Offsets are defined once from all source labels so
	masked fits keep the same offset columns.
	"""
	offset_design, offset_sources = source_offset_design(source_labels)
	periods, power, peak_period, false_alarm_threshold = calculate_periodogram(
		bjd, rv, rv_error
	)
	period, _ = fit_period(bjd, rv, rv_error, peak_period, offset_design)
	mask = exclude_orbit_fit_outliers(((bjd - bjd[0]) / period) % 1.0, rv, rv_error, offset_design)
	coefficients, covariance = fit_phase_curve(
		((bjd[mask] - bjd[0]) / period) % 1.0, rv[mask], rv_error[mask], offset_design[mask]
	)
	semiamplitude, _, conjunction, _ = sinusoid_parameters(coefficients, covariance, bjd[0], period)
	initial = np.concatenate([[coefficients[0], semiamplitude, period, conjunction, 0.0, 0.0], coefficients[3:]])

	fits = {}
	for model in ("circular", "keplerian"):
		parameters, covariance, jitter, reduced_chi_squared = fit_orbit(
			bjd[mask], rv[mask], rv_error[mask], source_labels[mask], offset_design[mask],
			initial, eccentric=model == "keplerian",
		)
		sigma = np.hypot(rv_error[mask], [jitter[label] for label in source_labels[mask]])
		pivot = np.average(bjd[mask], weights=sigma**-2)
		parameters, covariance, summary = summarize_orbit(parameters, covariance, pivot)
		summary["reduced_chi_squared"] = reduced_chi_squared
		summary["jitter_m_per_s"] = jitter
		fits[model] = (parameters, summary)
	eccentric_summary = fits["keplerian"][1]
	significance = eccentric_summary["eccentricity"] / eccentric_summary["eccentricity_uncertainty"]
	adopted = "keplerian" if significance > ECCENTRICITY_SIGNIFICANCE else "circular"
	adopted_parameters, adopted_summary = fits[adopted]

	parameters = {
		"measurement_count": len(bjd),
		"orbit_fit_excluded_count": np.count_nonzero(~mask),
		"periodogram_peak_period_days": peak_period,
		"periodogram_false_alarm_probability": FALSE_ALARM_PROBABILITY,
		"periodogram_false_alarm_threshold": false_alarm_threshold,
		"orbit_model": adopted,
		"eccentricity_significance": significance,
		**adopted_summary,
		"source_offsets_m_per_s": dict(zip(offset_sources, adopted_parameters[ORBIT_OFFSETS:])),
		"circular_fit": {key: value for key, value in fits["circular"][1].items()
			if not key.startswith(("eccentricity", "omega", "periapsis"))},
		"keplerian_fit": fits["keplerian"][1],
	}
	return SystemFit(
		parameters=_to_json(parameters),
		periods=periods,
		power=power,
		orbit_parameters=adopted_parameters,
		offset_corrected_rv=rv - offset_design @ adopted_parameters[ORBIT_OFFSETS:],
		orbit_fit_mask=mask,
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
	"""Draw one colored errorbar series per RV offset group."""
	for index, source in enumerate(sorted(set(source_labels))):
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
				color=f"C{index % 10}",
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
	semiamplitude = parameters["semiamplitude_m_per_s"]
	semiamplitude_uncertainty = parameters["semiamplitude_uncertainty_m_per_s"]
	conjunction = parameters["conjunction_bjd"]
	model_label = (
		f"Keplerian (e = {parameters['eccentricity']:.3f} +/- {parameters['eccentricity_uncertainty']:.3f})"
		if parameters["orbit_model"] == "keplerian" else "Circular orbit"
	)

	figure, axes = plt.subplots(3, 1, sharex=False, figsize=(8, 11))
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

	# Fold with the transit conjunction at phase 0.
	phase = ((bjd - conjunction) / period + 0.25) % 1.0 - 0.25
	mask = fit.orbit_fit_mask
	_errorbar_by_source(
		axes[2], phase, fit.offset_corrected_rv, rv_error, source_labels, mask
	)
	_errorbar_by_source(
		axes[2], phase, fit.offset_corrected_rv, rv_error, source_labels, ~mask,
		label_suffix=" (excluded)", alpha=0.25,
	)
	model_phase = np.linspace(-0.25, 0.75, 500)
	orbit = fit.orbit_parameters[:ORBIT_OFFSETS]
	axes[2].plot(
		model_phase,
		orbit_rv(conjunction + model_phase * period, orbit, np.empty((len(model_phase), 0))),
		color="black",
		linewidth=2,
		zorder=5,
		label=f"{model_label}, K = {semiamplitude:.4g} +/- {semiamplitude_uncertainty:.3g} m/s",
	)
	axes[2].axvline(0.0, color="tab:green", linestyle="--", label="Transit conjunction")
	axes[2].set_xlabel("Orbital phase from transit conjunction")
	axes[2].set_ylabel("Radial velocity (m/s)")
	axes[2].set_title(
		f"RV folded on {period_label}-day period; "
		f"K = {semiamplitude:.6g} +/- {semiamplitude_uncertainty:.6g} m/s"
	)
	axes[2].set_xlim(-0.25, 0.75)
	axes[2].grid(alpha=0.3)
	axes[2].legend(fontsize="small", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
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
	jitter = ", ".join(f"{label} {value:.1f}" for label, value in parameters["jitter_m_per_s"].items())
	lines = (
		f"adopted {parameters['orbit_model']} orbit"
		f" (e/sigma_e = {value('eccentricity_significance', '.2f')})",
		f"best-fit period = {value('period_days')} +/- {value('period_uncertainty_days')} days",
		f"RV semiamplitude = {value('semiamplitude_m_per_s')}"
		f" +/- {value('semiamplitude_uncertainty_m_per_s')} m/s",
		f"transit conjunction = BJD {value('conjunction_bjd', '.6f')}"
		f" +/- {value('conjunction_uncertainty_days', '.6f')}",
		f"orbital phase on {TARGET_DATE} = {value(f'phase_on_{TARGET_DATE}', '.6f')}"
		f" +/- {value(f'phase_uncertainty_on_{TARGET_DATE}', '.6f')} cycles"
		f" (+/- {value(f'phase_uncertainty_hours_on_{TARGET_DATE}', '.6f')} hours)",
		f"eccentricity = {value('eccentricity')} +/- {value('eccentricity_uncertainty')}",
		f"argument of periastron = {value('omega_degrees')}"
		f" +/- {value('omega_uncertainty_degrees')} degrees",
		f"time of periastron = BJD {value('periapsis_bjd', '.6f')}"
		f" +/- {value('periapsis_uncertainty_days', '.6f')}",
		f"jitter (m/s): {jitter}",
		f"reduced chi2 (with jitter) = {value('reduced_chi_squared')}",
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
		help=(
			"Also load matching synthetic RV points from RVdatabases/Synthetics/;"
			f" outputs from fits that include them get a {SYNTHETIC_SUFFIX!r} suffix."
		),
	)
	args = parser.parse_args()

	sources = list(OBSERVED_SOURCES) if args.source == "all" else [source_names[args.source]]
	if "Teklu" in sources and not args.database.is_file():
		parser.error(f"Database file not found: {args.database}")
	if args.synthetics:
		sources.append("Synthetic")

	datasets = rv_io.load_datasets(args.star, sources, args.database)
	raw_measurement_count = sum(len(data.time) for data in datasets)
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
	includes_synthetics = bool(np.any(source_labels == "Synthetic"))
	fit_parameters = {
		"input_star": args.star,
		"source": args.source,
		"includes_synthetics": includes_synthetics,
		**fit.parameters,
	}

	figure = plot_fit(args.star, bjd, rv, rv_error, source_labels, fit)
	PLOTS_DIRECTORY.mkdir(exist_ok=True)
	# Tag synthetic-inclusive fits so they never overwrite real-data fits, which
	# generate_synthetic_rvs reads back by star name.
	output_stem = f"{args.star}{SYNTHETIC_SUFFIX if includes_synthetics else ''}"
	output_path = PLOTS_DIRECTORY / f"{output_stem}_rv_fit_plot.png"
	figure.savefig(str(output_path), dpi=150)
	json_path = PLOTS_DIRECTORY / f"{output_stem}_rv_fit_parameters.json"
	json_path.write_text(json.dumps(fit_parameters, indent=2) + "\n", encoding="utf-8")
	print(f"Saved plot to {output_path}")
	print(f"Saved fit parameters to {json_path}")
	plt.show()


if __name__ == "__main__":
	main()
