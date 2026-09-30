#!/usr/bin/env python3
"""Plot combined radial velocities for a star and fit its orbit."""

import argparse
import json
import textwrap
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
OBSERVED_SOURCES = ("Teklu", "ExoArchive", "CLS", "HARPS", "HARPS2020", "SOPHIE", "Hebrard", "NEID", "ESPRESSO", "NeveuVanMalle", "Literature", "HARPSDRS", "ELODIE")
PERIOD_RANGE = (1.2, np.nextafter(8.0, 1.2))  # days
# Fractional window around the periodogram peak rescanned with instrument offsets
# (covers the cycle-count aliases of gaps longer than ~200 orbits).
ALIAS_SCAN_WINDOW = 0.005
FALSE_ALARM_PROBABILITY = 0.001
TARGET_DATE = "2027-07-01"
TARGET_BJD = 2_461_587.5  # 2027-07-01 00:00 UTC
# Orbit parameter layout used by orbit_rv; source offsets follow.
ORBIT_GAMMA, ORBIT_SEMIAMPLITUDE, ORBIT_PERIOD, ORBIT_CONJUNCTION, ORBIT_H, ORBIT_K = range(6)
ORBIT_OFFSETS = 6
MAX_ECCENTRICITY = 0.95
ECCENTRICITY_SIGNIFICANCE = 2.45  # Lucy & Sweeney (1971) 5% false-alarm level
# An eccentric outer orbit whose K runs away from the circular solution (an unsampled
# periastron spike) or whose e is extreme is not adopted.
OUTER_MAX_ECCENTRICITY = 0.8
OUTER_MAX_ECCENTRIC_K_RATIO = 2.0
JITTER_ITERATIONS = 8
MIN_JITTER_POINTS = 8  # smaller offset groups take the median jitter of the others
# Outer-signal detection on the one-planet residuals (see detect_outer_signal).
SEASON_DAYS = 60.0
ACTIVITY_MAX_PERIOD_DAYS = 60.0  # shorter residual peaks are treated as possible activity
MIN_RESIDUAL_PERIOD_DAYS = 10.0
OUTER_FALSE_ALARM_PROBABILITY = 1e-3
OUTER_MIN_AMPLITUDE_SIGNIFICANCE = 10.0  # K / sigma_K
RESIDUAL_PERIOD_GRID = 3000
# Vetting of a detected outer signal after fitting it jointly with the inner orbit.
OUTER_MIN_FITTED_SIGNIFICANCE = 20.0  # K / sigma_K (sinusoid) or max slope/curvature significance (trend)
YEARLY_ALIAS_TOLERANCE = 0.05  # reject periods within 5% of 1, 1/2, 1/3 yr (seasonal sampling)
TREND_MIN_AMPLITUDE_TO_NOISE = 3.0  # trend's implied minimum K relative to the median per-point noise
# Vetting uses one point per instrument per night, so dense single-night campaigns
# (e.g. transit sequences) cannot make a slow signal look significant.
NIGHT_GAP_DAYS = 0.25  # exposures closer than this (same instrument) belong to one night
OUTER_MIN_NIGHTS = 10
OUTER_MIN_SEASONS = 5  # a sinusoid can fit a few season means whatever the signal
SEASON_GAP_DAYS = 120.0  # observing seasons are separated by gaps longer than this
DAYS_PER_YEAR = 365.25
K_JUPITER_1YR_M_PER_S = 28.4329  # K of a Jupiter-mass planet on a 1-yr orbit around 1 Msun
TREND_MIN_MASS_MJUP_PER_AU2 = 0.0056  # M sin i >= this * |dv/dt| [m/s/yr] * a[AU]^2


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


def refine_period_with_offsets(
	bjd: np.ndarray, rv: np.ndarray, rv_error: np.ndarray, offset_design: np.ndarray, period: float
) -> float:
	"""Return the best circular-orbit period near ``period`` with source offsets fitted.

	The Lomb-Scargle periodogram fits one mean, so a small dataset whose own median
	is a poor zero point (e.g. a few points on one side of the orbit) can move its
	peak to a neighbouring cycle-count alias across a long gap. Rescanning at a
	step of 1/10 of the finest alias spacing (P^2 / baseline) with the offsets as
	free linear terms picks the right one.
	"""
	step = 0.1 * period**2 / np.ptp(bjd)
	periods = np.arange(period * (1 - ALIAS_SCAN_WINDOW), period * (1 + ALIAS_SCAN_WINDOW), step)
	weights = 1 / rv_error
	chi_squared = []
	for trial in periods:
		design = sinusoid_design((bjd - bjd[0]) / trial, offset_design) * weights[:, None]
		coefficients, *_ = np.linalg.lstsq(design, rv * weights, rcond=None)
		chi_squared.append(np.sum((rv * weights - design @ coefficients) ** 2))
	return float(periods[np.argmin(chi_squared)])


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


def keplerian_rv(
	bjd: np.ndarray, semiamplitude: float, period: float, conjunction: float,
	eccentricity: float, omega: float,
) -> np.ndarray:
	"""Keplerian RV (no systemic velocity) with transit conjunction ``conjunction``."""
	mean_anomaly = 2 * np.pi * (bjd - periapsis_time(conjunction, period, eccentricity, omega)) / period
	eccentric_anomaly = solve_kepler(mean_anomaly, eccentricity)
	true_anomaly = 2 * np.arctan2(
		np.sqrt(1 + eccentricity) * np.sin(eccentric_anomaly / 2),
		np.sqrt(1 - eccentricity) * np.cos(eccentric_anomaly / 2),
	)
	return semiamplitude * (np.cos(true_anomaly + omega) + eccentricity * np.cos(omega))


def orbit_rv(
	bjd: np.ndarray,
	parameters: np.ndarray,
	offset_design: np.ndarray,
	outer: "OuterSignal | None" = None,
) -> np.ndarray:
	"""Evaluate the inner Keplerian orbit, source offsets, and optional outer signal.

	``parameters`` are [gamma, K, P, Tc, sqrt(e)cos(omega), sqrt(e)sin(omega),
	source offsets..., outer-signal parameters...], where Tc is the transit
	conjunction (f + omega = pi/2); e = 0 gives -K sin(2 pi (t - Tc) / P).
	"""
	gamma, semiamplitude, period, conjunction = parameters[:4]
	offsets_end = ORBIT_OFFSETS + offset_design.shape[1]
	model = gamma + keplerian_rv(bjd, semiamplitude, period, conjunction, *eccentricity_and_omega(parameters))
	model = model + offset_design @ parameters[ORBIT_OFFSETS:offsets_end]
	if outer is not None:
		model = model + outer.evaluate(bjd, parameters[offsets_end:])
	return model


class OuterSignal:
	"""An extra RV term (e.g. an outer planet) fitted jointly with the inner orbit.

	Subclasses define the parameters, their starting values and bounds, how to
	evaluate the term, and what to report. ``origin`` gives values subtracted
	during the fit so that finite-difference steps are well scaled (e.g. times).
	"""

	name = ""
	parameter_names: tuple[str, ...] = ()

	def evaluate(self, bjd: np.ndarray, parameters: np.ndarray) -> np.ndarray:
		raise NotImplementedError

	def initial(self) -> np.ndarray:
		raise NotImplementedError

	def bounds(self) -> tuple[list[float], list[float]]:
		count = len(self.parameter_names)
		return [-np.inf] * count, [np.inf] * count

	def origin(self) -> np.ndarray:
		return np.zeros(len(self.parameter_names))

	def starts(self, parameters: np.ndarray) -> list[np.ndarray]:
		"""Starting points tried by fit_orbit (one by default)."""
		return [parameters]

	def describe(self) -> str:
		"""Short model name for plot titles."""
		return self.name

	def summarize(self, parameters: np.ndarray, covariance: np.ndarray, baseline_days: float) -> dict:
		summary = {"model": self.name}
		for index, name in enumerate(self.parameter_names):
			summary[name] = parameters[index]
			summary[f"{name}_uncertainty"] = np.sqrt(max(covariance[index, index], 0.0))
		return summary


class Sinusoid(OuterSignal):
	"""Circular outer orbit, -K sin(2 pi (t - Tc) / P), for periods within ~2x the baseline."""

	name = "sinusoid"
	parameter_names = ("semiamplitude_m_per_s", "period_days", "conjunction_bjd")

	def __init__(self, semiamplitude: float, period: float, conjunction: float):
		self.start = np.array([semiamplitude, period, conjunction])

	def evaluate(self, bjd, parameters):
		semiamplitude, period, conjunction = parameters
		return -semiamplitude * np.sin(2 * np.pi * (bjd - conjunction) / period)

	def initial(self):
		return self.start.copy()

	def bounds(self):
		return [0.0, ACTIVITY_MAX_PERIOD_DAYS, -np.inf], [np.inf, np.inf, np.inf]

	def origin(self):
		return np.array([0.0, 0.0, self.start[2]])

	def describe(self):
		return "circular outer planet"

	def summarize(self, parameters, covariance, baseline_days):
		summary = {
			"model": self.name,
			"semiamplitude_m_per_s": parameters[0],
			"semiamplitude_uncertainty_m_per_s": np.sqrt(covariance[0, 0]),
			"period_days": parameters[1],
			"period_uncertainty_days": np.sqrt(covariance[1, 1]),
			"conjunction_bjd": parameters[2],
			"conjunction_uncertainty_days": np.sqrt(covariance[2, 2]),
			"period_to_baseline": parameters[1] / baseline_days,
			# With less than ~1.5 cycles observed, P and K depend on the sinusoid
			# assumption and are effectively lower limits despite small formal errors.
			"orbit_coverage": "multiple_cycles" if parameters[1] <= baseline_days / 1.5 else "partial_orbit",
		}
		# m sin i = this value * (M_star / Msun)^(2/3); circular orbit, m << M_star.
		scale, scale_uncertainty = _propagate(
			lambda p: p[0] / K_JUPITER_1YR_M_PER_S * (p[1] / DAYS_PER_YEAR) ** (1 / 3),
			parameters, covariance,
		)
		summary["msini_mjup_per_mstar_msun_2_3"] = scale
		summary["msini_mjup_per_mstar_msun_2_3_uncertainty"] = scale_uncertainty
		return summary


class Trend(OuterSignal):
	"""Quadratic trend about ``reference_bjd`` for signals longer than the baseline."""

	name = "trend"
	parameter_names = ("slope_m_per_s_per_yr", "curvature_m_per_s_per_yr2")

	def __init__(self, reference_bjd: float, slope: float, curvature: float):
		self.reference_bjd = reference_bjd
		self.start = np.array([slope, curvature])

	def evaluate(self, bjd, parameters):
		years = (bjd - self.reference_bjd) / DAYS_PER_YEAR
		return parameters[0] * years + parameters[1] * years**2

	def initial(self):
		return self.start.copy()

	def describe(self):
		return "quadratic trend"

	def summarize(self, parameters, covariance, baseline_days):
		span_years = baseline_days / DAYS_PER_YEAR
		half_range = 0.5 * np.ptp(self.evaluate(
			self.reference_bjd + np.linspace(-0.5, 0.5, 201) * baseline_days, parameters
		))
		return {
			"model": self.name,
			"reference_bjd": self.reference_bjd,
			"slope_m_per_s_per_yr": parameters[0],
			"slope_uncertainty_m_per_s_per_yr": np.sqrt(covariance[0, 0]),
			"curvature_m_per_s_per_yr2": parameters[1],
			"curvature_uncertainty_m_per_s_per_yr2": np.sqrt(covariance[1, 1]),
			# Unresolved companion: only lower limits on P and K, and a mass-separation relation.
			"period_lower_limit_days": baseline_days,
			"semiamplitude_lower_limit_m_per_s": half_range,
			"min_mass_mjup_per_au2": TREND_MIN_MASS_MJUP_PER_AU2 * abs(parameters[0]),
			"baseline_years": span_years,
		}


class KeplerianCompanion(OuterSignal):
	"""Eccentric outer orbit, parameterized like the inner planet.

	Parameters are K, P, the transit conjunction Tc (f + omega = pi/2), and
	sqrt(e)cos(omega), sqrt(e)sin(omega). Fits start from a circular solution
	(e.g. the vetted Sinusoid) and from four periastron orientations at e = 0.09.
	"""

	name = "keplerian"
	parameter_names = (
		"semiamplitude_m_per_s", "period_days", "conjunction_bjd",
		"sqrt_e_cos_omega", "sqrt_e_sin_omega",
	)

	def __init__(self, semiamplitude: float, period: float, conjunction: float, h: float = 0.0, k: float = 0.0):
		self.start = np.array([semiamplitude, period, conjunction, h, k])

	@staticmethod
	def elements(parameters: np.ndarray) -> tuple[float, float]:
		h, k = parameters[3], parameters[4]
		return min(h * h + k * k, MAX_ECCENTRICITY), float(np.arctan2(k, h))

	def evaluate(self, bjd, parameters):
		semiamplitude, period, conjunction = parameters[:3]
		return keplerian_rv(bjd, semiamplitude, period, conjunction, *self.elements(parameters))

	def initial(self):
		return self.start.copy()

	def starts(self, parameters):
		return [parameters] + [
			np.concatenate([parameters[:3], _orientation_start(angle)]) for angle in PERIASTRON_STARTS_DEGREES
		]

	def bounds(self):
		root_max_e = np.sqrt(MAX_ECCENTRICITY)
		return (
			[0.0, ACTIVITY_MAX_PERIOD_DAYS, -np.inf, -root_max_e, -root_max_e],
			[np.inf, np.inf, np.inf, root_max_e, root_max_e],
		)

	def origin(self):
		return np.array([0.0, 0.0, self.start[2], 0.0, 0.0])

	def describe(self):
		return "eccentric Keplerian outer planet"

	def summarize(self, parameters, covariance, baseline_days):
		eccentricity, e_uncertainty = _propagate(lambda p: self.elements(p)[0], parameters, covariance)
		omega = self.elements(parameters)[1]
		_, omega_uncertainty = _propagate(
			lambda p: (self.elements(p)[1] - omega + np.pi) % (2 * np.pi) - np.pi, parameters, covariance
		)
		periapsis, periapsis_uncertainty = _propagate(
			lambda p: periapsis_time(p[2], p[1], *self.elements(p)), parameters, covariance
		)
		# m sin i = this value * (M_star / Msun)^(2/3), for m << M_star.
		scale, scale_uncertainty = _propagate(
			lambda p: p[0] * np.sqrt(1 - self.elements(p)[0] ** 2) / K_JUPITER_1YR_M_PER_S
			* (p[1] / DAYS_PER_YEAR) ** (1 / 3),
			parameters, covariance,
		)
		return {
			"model": self.name,
			"semiamplitude_m_per_s": parameters[0],
			"semiamplitude_uncertainty_m_per_s": np.sqrt(covariance[0, 0]),
			"period_days": parameters[1],
			"period_uncertainty_days": np.sqrt(covariance[1, 1]),
			"conjunction_bjd": parameters[2],
			"conjunction_uncertainty_days": np.sqrt(covariance[2, 2]),
			"eccentricity": eccentricity,
			"eccentricity_uncertainty": e_uncertainty,
			"eccentricity_significance": eccentricity / e_uncertainty if e_uncertainty > 0 else np.nan,
			"omega_degrees": np.degrees(omega),
			"omega_uncertainty_degrees": np.degrees(omega_uncertainty),
			"periapsis_bjd": periapsis,
			"periapsis_uncertainty_days": periapsis_uncertainty,
			"msini_mjup_per_mstar_msun_2_3": scale,
			"msini_mjup_per_mstar_msun_2_3_uncertainty": scale_uncertainty,
			"period_to_baseline": parameters[1] / baseline_days,
			"orbit_coverage": "multiple_cycles" if parameters[1] <= baseline_days / 1.5 else "partial_orbit",
		}


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

	Labels with fewer than ``MIN_JITTER_POINTS`` points take the median of the others,
	or, if no label has enough points, one jitter pooled over all points.
	"""
	def solve(in_group):
		r2, e2 = residuals[in_group] ** 2, rv_error[in_group] ** 2
		excess = lambda s: np.mean(r2 / (e2 + s * s)) - 1
		return 0.0 if excess(0.0) <= 0 else brentq(excess, 0.0, np.sqrt(r2.max()) + 1)

	jitter = {}
	for label in dict.fromkeys(source_labels):
		in_label = source_labels == label
		if in_label.sum() >= MIN_JITTER_POINTS:
			jitter[label] = solve(in_label)
	fallback = float(np.median(list(jitter.values()))) if jitter else solve(np.ones(len(residuals), dtype=bool))
	return {label: jitter.get(label, fallback) for label in dict.fromkeys(source_labels)}


PERIASTRON_STARTS_DEGREES = (45, 135, 225, 315)


def _orientation_start(angle_degrees: float) -> np.ndarray:
	"""sqrt(e)cos(omega), sqrt(e)sin(omega) for e = 0.09 at the given omega."""
	angle = np.radians(angle_degrees)
	return np.array([0.3 * np.cos(angle), 0.3 * np.sin(angle)])


def fit_orbit(
	bjd: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	source_labels: np.ndarray,
	offset_design: np.ndarray,
	initial: np.ndarray,
	eccentric: bool,
	outer: OuterSignal | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float]:
	"""Fit the orbit (and optional outer signal) with per-label jitter.

	``initial`` holds the orbit, offset, and outer-signal parameters in orbit_rv's
	layout. Jitter and fit are iterated to convergence. Returns parameters, their
	covariance (fixed parameters have zero variance), the jitter by label, and
	the reduced chi-squared including jitter.
	"""
	outer_count = len(outer.parameter_names) if outer is not None else 0
	free = np.ones(len(initial), dtype=bool)
	if not eccentric:
		free[[ORBIT_H, ORBIT_K]] = False
	root_max_e = np.sqrt(MAX_ECCENTRICITY)
	outer_lower, outer_upper = outer.bounds() if outer is not None else ([], [])
	lower = np.array([-np.inf, 0.0, PERIOD_RANGE[0], -np.inf, -root_max_e, -root_max_e]
		+ [-np.inf] * offset_design.shape[1] + list(outer_lower))
	upper = np.array([np.inf, np.inf, PERIOD_RANGE[1], np.inf, root_max_e, root_max_e]
		+ [np.inf] * offset_design.shape[1] + list(outer_upper))
	# Fit parameters relative to an origin (times near their starting values) so
	# finite-difference steps are well scaled; the covariance is unchanged.
	origin = np.zeros(len(initial))
	origin[ORBIT_CONJUNCTION] = initial[ORBIT_CONJUNCTION]
	if outer_count:
		origin[-outer_count:] = outer.origin()
	initial = initial.copy()
	if not eccentric:
		initial[[ORBIT_H, ORBIT_K]] = 0.0  # a circular fit must not inherit a starting e
	inner_count = len(initial) - outer_count
	inner_starts = [initial[:inner_count]]
	if eccentric:  # a few periastron orientations at e = 0.09 avoid local minima
		inner_starts += [
			np.concatenate([initial[:ORBIT_H], _orientation_start(angle), initial[ORBIT_OFFSETS:inner_count]])
			for angle in PERIASTRON_STARTS_DEGREES
		]
	outer_starts = outer.starts(initial[inner_count:]) if outer_count else [np.empty(0)]
	starts = [np.concatenate([inner, outer_start]) for inner in inner_starts for outer_start in outer_starts]
	jitter = {label: 0.0 for label in dict.fromkeys(source_labels)}
	parameters = starts[0]

	def residuals(x: np.ndarray) -> np.ndarray:
		full = parameters_template.copy()
		full[free] = x + origin[free]
		return (rv - orbit_rv(bjd, full, offset_design, outer)) / sigma

	for iteration in range(JITTER_ITERATIONS):
		sigma = np.hypot(rv_error, np.array([jitter[label] for label in source_labels]))
		best = None
		for start in (starts if iteration == 0 else [parameters]):
			parameters_template = start.copy()
			x0 = np.clip(start[free], lower[free], upper[free]) - origin[free]
			fit = least_squares(
				residuals, x0, bounds=(lower[free] - origin[free], upper[free] - origin[free]),
				x_scale="jac", max_nfev=5_000,
			)
			if best is None or fit.cost < best[0].cost:
				best = (fit, start.copy())
		fit, parameters_template = best
		parameters = parameters_template.copy()
		parameters[free] = fit.x + origin[free]
		new_jitter = estimate_jitter(
			rv - orbit_rv(bjd, parameters, offset_design, outer), rv_error, source_labels
		)
		converged = all(abs(new_jitter[l] - jitter[l]) < 0.01 * (jitter[l] + 0.1) for l in jitter)
		jitter = new_jitter
		if converged:
			break
	sigma = np.hypot(rv_error, np.array([jitter[label] for label in source_labels]))
	parameters_template = parameters.copy()
	x = parameters[free] - origin[free]
	final_residuals = residuals(x)
	jacobian = _central_jacobian(residuals, x)
	reduced_chi_squared = np.sum(final_residuals**2) / (len(rv) - free.sum())
	covariance = np.zeros((len(parameters), len(parameters)))
	covariance[np.ix_(free, free)] = reduced_chi_squared * np.linalg.pinv(jacobian.T @ jacobian)
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


def short_term_jitter(times, residuals, errors, season_days=SEASON_DAYS) -> float:
	"""Robust within-season residual scatter, minus measurement errors in quadrature.

	Subtracting each season's median residual removes long-period signals (such as
	outer companions) that a single-planet orbit leaves in the residuals.
	"""
	season = np.floor((times - times.min()) / season_days)
	centered = []
	for value in np.unique(season):
		in_season = season == value
		if in_season.sum() >= 3:
			centered.extend(residuals[in_season] - np.median(residuals[in_season]))
	if len(centered) < 5:
		centered = residuals - np.median(residuals)
	centered = np.asarray(centered)
	scatter = 1.4826 * np.median(np.abs(centered - np.median(centered)))
	return float(np.sqrt(max(scatter**2 - np.median(errors) ** 2, 0.0)))


def _residual_periodogram(
	bjd: np.ndarray, residuals: np.ndarray, sigma: np.ndarray, groups: np.ndarray, periods: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Chi-squared reduction from a sinusoid at each period, with a free offset per group.

	Returns (delta chi2, sine coefficient, cosine coefficient) per period. Offsets
	are projected out once; each period then needs only a 2x2 solve.
	"""
	weights = 1 / sigma
	basis, _ = np.linalg.qr(groups * weights[:, np.newaxis])
	project = lambda matrix: matrix - (matrix @ basis) @ basis.T
	y = project((residuals * weights)[np.newaxis, :])[0]
	delta, sine, cosine = (np.empty(len(periods)) for _ in range(3))
	for start in range(0, len(periods), 256):
		angle = 2 * np.pi * bjd[np.newaxis, :] / periods[start:start + 256, np.newaxis]
		s_column = project(np.sin(angle) * weights)
		c_column = project(np.cos(angle) * weights)
		ss, sc, cc = (np.sum(a * b, axis=1) for a, b in ((s_column, s_column), (s_column, c_column), (c_column, c_column)))
		sy, cy = s_column @ y, c_column @ y
		determinant = ss * cc - sc**2
		a = (cc * sy - sc * cy) / determinant
		b = (ss * cy - sc * sy) / determinant
		delta[start:start + 256] = a * sy + b * cy
		sine[start:start + 256], cosine[start:start + 256] = a, b
	return delta, sine, cosine


def detect_outer_signal(
	bjd: np.ndarray, residuals: np.ndarray, rv_error: np.ndarray, source_labels: np.ndarray
) -> dict:
	"""Flag a possible outer planet from one-planet residuals (no catalogue lookups).

	Residuals are weighted by errors plus each group's within-season jitter, which
	excludes the slow signal being searched for. A sinusoid with free per-group
	offsets is scanned over ``ACTIVITY_MAX_PERIOD_DAYS``..3x baseline (outer) and
	``MIN_RESIDUAL_PERIOD_DAYS``..``ACTIVITY_MAX_PERIOD_DAYS`` (activity). Flags:

	- ``outer_planet``: long-period peak with FAP < ``OUTER_FALSE_ALARM_PROBABILITY``,
	  K / sigma_K > ``OUTER_MIN_AMPLITUDE_SIGNIFICANCE``, and P <= 2x baseline;
	- ``trend``: the same but P > 2x baseline (only lower limits on P and K);
	- ``possible_activity``: only a significant short-period (rotation-like) peak;
	- ``none``.

	fit_system then vets ``outer_planet``/``trend`` detections (vet_outer_signal);
	failures become ``outer_candidate``/``trend_candidate`` and get no fitted term.

	FAPs use a Bonferroni count of independent frequencies and assume white noise,
	so the amplitude-significance cut guards against correlated noise.
	"""
	labels = list(dict.fromkeys(source_labels))
	jitter = {}
	for label in labels:
		in_label = source_labels == label
		if in_label.sum() >= MIN_JITTER_POINTS:
			jitter[label] = short_term_jitter(bjd[in_label], residuals[in_label], rv_error[in_label])
	fallback = float(np.median(list(jitter.values()))) if jitter else 0.0
	sigma = np.hypot(rv_error, [jitter.get(label, fallback) for label in source_labels])
	groups = np.column_stack([(source_labels == label).astype(float) for label in labels])
	baseline = float(np.ptp(bjd))
	amplitude_sigma = float(np.sqrt(2 / np.sum(sigma**-2)))

	def best_peak(minimum, maximum):
		if maximum <= minimum:
			return None
		periods = np.geomspace(minimum, maximum, RESIDUAL_PERIOD_GRID)
		delta, sine, cosine = _residual_periodogram(bjd, residuals, sigma, groups, periods)
		best = int(np.argmax(delta))
		independent = max(baseline * (1 / minimum - 1 / maximum), 1.0)
		semiamplitude = float(np.hypot(sine[best], cosine[best]))
		# -K sin(w (t - Tc)) = S sin(w t) + C cos(w t)  =>  w Tc = atan2(C, -S)
		conjunction = float(np.arctan2(cosine[best], -sine[best]) / (2 * np.pi) * periods[best])
		return {
			"period_days": float(periods[best]),
			"semiamplitude_m_per_s": semiamplitude,
			"amplitude_significance": semiamplitude / amplitude_sigma,
			"delta_chi_squared": float(delta[best]),
			"false_alarm_probability": float(min(1.0, independent * np.exp(-delta[best] / 2))),
			"conjunction_bjd": conjunction,
		}

	def significant(peak):
		return (
			peak is not None
			and peak["false_alarm_probability"] < OUTER_FALSE_ALARM_PROBABILITY
			and peak["amplitude_significance"] > OUTER_MIN_AMPLITUDE_SIGNIFICANCE
		)

	long_peak = best_peak(ACTIVITY_MAX_PERIOD_DAYS, 3 * baseline)
	short_peak = best_peak(MIN_RESIDUAL_PERIOD_DAYS, ACTIVITY_MAX_PERIOD_DAYS)
	if significant(long_peak):
		flag = "outer_planet" if long_peak["period_days"] <= 2 * baseline else "trend"
	elif significant(short_peak):
		flag = "possible_activity"
	else:
		flag = "none"

	# Quadratic trend about the weighted mean time, for the trend model's start.
	reference = float(np.average(bjd, weights=sigma**-2))
	years = (bjd - reference) / DAYS_PER_YEAR
	design = np.column_stack([groups, years, years**2]) / sigma[:, np.newaxis]
	coefficients = np.linalg.lstsq(design, residuals / sigma, rcond=None)[0]
	return {
		"flag": flag,
		"baseline_days": baseline,
		"median_noise_m_per_s": float(np.median(sigma)),
		"short_term_jitter_m_per_s": {label: jitter.get(label, fallback) for label in labels},
		"amplitude_uncertainty_m_per_s": amplitude_sigma,
		"long_period_peak": long_peak,
		"short_period_peak": short_peak,
		"trend_reference_bjd": reference,
		"trend_slope_m_per_s_per_yr": float(coefficients[-2]),
		"trend_curvature_m_per_s_per_yr2": float(coefficients[-1]),
	}


OUTER_MODELS = ("auto", "none", "sinusoid", "trend", "keplerian")


def choose_outer_signal(detection: dict, outer_model: str, pivot_bjd: float) -> OuterSignal | None:
	"""Pick the outer-signal term: by the detection flag ("auto") or as requested."""
	if outer_model == "auto":
		outer_model = {"outer_planet": "sinusoid", "trend": "trend"}.get(detection["flag"], "none")
	if outer_model == "none":
		return None
	if outer_model == "trend":
		return Trend(
			detection["trend_reference_bjd"],
			detection["trend_slope_m_per_s_per_yr"],
			detection["trend_curvature_m_per_s_per_yr2"],
		)
	peak = detection["long_period_peak"]
	if peak is None:
		raise ValueError("No long-period residual peak to start an outer orbit from")
	period = peak["period_days"]
	conjunction = peak["conjunction_bjd"] + np.round((pivot_bjd - peak["conjunction_bjd"]) / period) * period
	if outer_model == "keplerian":
		return KeplerianCompanion(peak["semiamplitude_m_per_s"], period, conjunction)
	return Sinusoid(peak["semiamplitude_m_per_s"], period, conjunction)


def bin_by_night(
	bjd: np.ndarray, values: np.ndarray, rv_error: np.ndarray, source_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Inverse-variance average each instrument's exposures within a night.

	Nights are runs of same-label exposures separated by < ``NIGHT_GAP_DAYS``, which
	works for any observatory longitude (a fixed UT-day boundary splits Hawaii nights).
	"""
	times, means, errors, labels = [], [], [], []
	for label in dict.fromkeys(source_labels):
		in_label = np.flatnonzero(source_labels == label)
		in_label = in_label[np.argsort(bjd[in_label])]
		breaks = np.flatnonzero(np.diff(bjd[in_label]) > NIGHT_GAP_DAYS) + 1
		for night in np.split(in_label, breaks):
			weights = rv_error[night] ** -2
			times.append(np.average(bjd[night], weights=weights))
			means.append(np.average(values[night], weights=weights))
			errors.append(1 / np.sqrt(weights.sum()))
			labels.append(label)
	return np.array(times), np.array(means), np.array(errors), np.array(labels, dtype=object)


def _binned_outer_fit(
	outer: OuterSignal, start: np.ndarray, bjd: np.ndarray, residuals: np.ndarray,
	rv_error: np.ndarray, source_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
	"""Fit the outer term plus free per-label offsets to nightly-binned residuals.

	Returns the outer parameters, their covariance (per-label jitter included), and
	the number of independent nights.
	"""
	times, means, errors, labels = bin_by_night(bjd, residuals, rv_error, source_labels)
	groups = np.column_stack([(labels == label).astype(float) for label in dict.fromkeys(labels)])
	count = len(outer.parameter_names)
	origin = np.concatenate([outer.origin(), np.zeros(groups.shape[1])])
	lower, upper = outer.bounds()
	lower = np.array(list(lower) + [-np.inf] * groups.shape[1]) - origin
	upper = np.array(list(upper) + [np.inf] * groups.shape[1]) - origin
	x = np.clip(np.concatenate([start, np.zeros(groups.shape[1])]) - origin, lower, upper)
	jitter = {label: 0.0 for label in dict.fromkeys(labels)}

	def residual(x):
		full = x + origin
		return (means - outer.evaluate(times, full[:count]) - groups @ full[count:]) / sigma

	for _ in range(JITTER_ITERATIONS):
		sigma = np.hypot(errors, [jitter[label] for label in labels])
		x = least_squares(residual, x, bounds=(lower, upper), x_scale="jac", max_nfev=5_000).x
		jitter = estimate_jitter(residual(x) * sigma, errors, labels)
	sigma = np.hypot(errors, [jitter[label] for label in labels])
	jacobian = _central_jacobian(residual, x)
	dof = max(len(means) - len(x), 1)
	covariance = np.sum(residual(x) ** 2) / dof * np.linalg.pinv(jacobian.T @ jacobian)
	return (x + origin)[:count], covariance[:count, :count], len(means)


def vet_outer_signal(
	outer: OuterSignal, detection: dict, data: tuple, one_planet: np.ndarray, eccentric: bool
) -> tuple[bool, dict]:
	"""Fit a detected outer term jointly with the adopted inner orbit and vet it.

	Residual scans with hundreds of points pass white-noise tests easily, so the
	signal must also be strong (``OUTER_MIN_FITTED_SIGNIFICANCE``) in a refit of the
	outer term to nightly-binned residuals from the joint fit, over at least
	``OUTER_MIN_NIGHTS`` independent nights and ``OUTER_MIN_SEASONS`` seasons.
	Sinusoids near 1, 1/2 or 1/3 yr are rejected as seasonal-sampling systematics;
	trends must imply an amplitude of at least ``TREND_MIN_AMPLITUDE_TO_NOISE``
	times the typical noise, since slow drifts are nearly degenerate with
	instrument offsets.
	"""
	parameters, covariance, _, _ = fit_orbit(
		*data, np.concatenate([one_planet, outer.initial()]), eccentric=eccentric, outer=outer
	)
	count = len(outer.parameter_names)
	exposure_errors = np.sqrt(np.diag(covariance)[-count:])
	bjd, rv, rv_error, source_labels, offset_design = data
	# Residuals from the inner orbit and offsets, with the outer signal left in.
	residuals = rv - orbit_rv(bjd, parameters[:-count], offset_design)
	values, binned_covariance, nights = _binned_outer_fit(
		outer, parameters[-count:], bjd, residuals, rv_error, source_labels
	)
	errors = np.sqrt(np.diag(binned_covariance))
	covariance = np.zeros_like(covariance)
	covariance[-count:, -count:] = binned_covariance
	seasons = 1 + int(np.sum(np.diff(np.sort(bjd)) > SEASON_GAP_DAYS))
	reasons = []
	if nights < OUTER_MIN_NIGHTS:
		reasons.append(f"only {nights} independent nights")
	if seasons < OUTER_MIN_SEASONS:
		reasons.append(f"only {seasons} observing seasons")
	if outer.name == "sinusoid":
		significance = values[0] / errors[0]
		period = values[1]
		alias = min(abs(period - DAYS_PER_YEAR / harmonic) / period for harmonic in (1, 2, 3))
		if alias < YEARLY_ALIAS_TOLERANCE:
			reasons.append(f"period within {alias:.1%} of a yearly harmonic")
		vetting = {
			"fitted_significance": significance,
			"exposure_level_significance": parameters[-count] / exposure_errors[0],
			"yearly_alias_offset": alias,
		}
	else:
		significance = float(np.max(np.abs(values) / errors))
		half_range = outer.summarize(values, covariance[-count:, -count:], detection["baseline_days"])[
			"semiamplitude_lower_limit_m_per_s"
		]
		amplitude_to_noise = half_range / detection["median_noise_m_per_s"]
		if amplitude_to_noise < TREND_MIN_AMPLITUDE_TO_NOISE:
			reasons.append(f"implied amplitude only {amplitude_to_noise:.1f}x the per-point noise")
		vetting = {
			"fitted_significance": significance,
			"exposure_level_significance": float(np.max(np.abs(parameters[-count:]) / exposure_errors)),
			"amplitude_to_noise": amplitude_to_noise,
		}
	if significance < OUTER_MIN_FITTED_SIGNIFICANCE:
		reasons.append(f"fitted significance {significance:.1f} < {OUTER_MIN_FITTED_SIGNIFICANCE:g}")
	vetting["independent_nights"] = nights
	vetting["observing_seasons"] = seasons
	vetting["rejection_reasons"] = reasons
	return not reasons, vetting


def saved_outer_signal(parameters: dict) -> tuple[OuterSignal | None, np.ndarray]:
	"""Return the fitted outer-signal term and its parameters from a saved fit."""
	fit = (parameters.get("outer_signal") or {}).get("fit")
	if not fit:
		return None, np.empty(0)
	if fit["model"] == "sinusoid":
		values = [fit["semiamplitude_m_per_s"], fit["period_days"], fit["conjunction_bjd"]]
		return Sinusoid(*values), np.array(values)
	if fit["model"] == "trend":
		values = [fit["slope_m_per_s_per_yr"], fit["curvature_m_per_s_per_yr2"]]
		return Trend(fit["reference_bjd"], *values), np.array(values)
	if fit["model"] == "keplerian":
		root_e, omega = np.sqrt(fit["eccentricity"]), np.radians(fit["omega_degrees"])
		values = [
			fit["semiamplitude_m_per_s"], fit["period_days"], fit["conjunction_bjd"],
			root_e * np.cos(omega), root_e * np.sin(omega),
		]
		return KeplerianCompanion(*values), np.array(values)
	raise ValueError(f"Unknown saved outer-signal model {fit['model']!r}")


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
	orbit_parameters: np.ndarray  # adopted inner orbit only, see orbit_rv
	offset_corrected_rv: np.ndarray  # offsets and outer signal removed
	orbit_fit_mask: np.ndarray
	outer: OuterSignal | None = None
	outer_parameters: np.ndarray | None = None


def fit_system(
	bjd: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	source_labels: np.ndarray,
	outer_model: str = "auto",
	initial_period: float | None = None,
) -> SystemFit:
	"""Find the period and fit circular and eccentric orbits with per-label jitter.

	A circular sinusoid (periodogram peak, then refined period) initializes the
	fits and flags outliers. A first circular fit's residuals are searched for an
	outer signal (detect_outer_signal); ``outer_model`` "auto" adds a sinusoid or
	quadratic trend when one is flagged ("none", "sinusoid", "trend", and the
	"keplerian" hook can be forced). The period, K, transit conjunction, offsets,
	and outer term are then fitted jointly, once with e = 0 and once with e free.
	The eccentric fit is adopted only if e exceeds ``ECCENTRICITY_SIGNIFICANCE``
	times its uncertainty (Lucy & Sweeney 1971). Offsets are defined once from all
	source labels so masked fits keep the same offset columns. ``initial_period``
	replaces the periodogram peak as the starting inner period when given.
	"""
	offset_design, offset_sources = source_offset_design(source_labels)
	offsets_end = ORBIT_OFFSETS + offset_design.shape[1]
	periods, power, peak_period, false_alarm_threshold = calculate_periodogram(
		bjd, rv, rv_error
	)
	# A known period (e.g. from transits) avoids periodogram aliases in sparse data.
	start = initial_period or refine_period_with_offsets(bjd, rv, rv_error, offset_design, peak_period)
	period, _ = fit_period(bjd, rv, rv_error, start, offset_design)
	mask = exclude_orbit_fit_outliers(((bjd - bjd[0]) / period) % 1.0, rv, rv_error, offset_design)
	coefficients, covariance = fit_phase_curve(
		((bjd[mask] - bjd[0]) / period) % 1.0, rv[mask], rv_error[mask], offset_design[mask]
	)
	semiamplitude, _, conjunction, _ = sinusoid_parameters(coefficients, covariance, bjd[0], period)
	initial = np.concatenate([[coefficients[0], semiamplitude, period, conjunction, 0.0, 0.0], coefficients[3:]])
	data = (bjd[mask], rv[mask], rv_error[mask], source_labels[mask], offset_design[mask])

	def fit_both(start: np.ndarray, outer: OuterSignal | None):
		"""Circular and eccentric fits, and the model adopted by the e significance test."""
		fits = {}
		for model in ("circular", "keplerian"):
			parameters, covariance, jitter, reduced_chi_squared = fit_orbit(
				*data, start, eccentric=model == "keplerian", outer=outer
			)
			sigma = np.hypot(rv_error[mask], [jitter[label] for label in source_labels[mask]])
			pivot = np.average(bjd[mask], weights=sigma**-2)
			parameters, covariance, summary = summarize_orbit(parameters, covariance, pivot)
			summary["reduced_chi_squared"] = reduced_chi_squared
			summary["jitter_m_per_s"] = jitter
			fits[model] = (parameters, covariance, summary, pivot)
		eccentric_summary = fits["keplerian"][2]
		significance = eccentric_summary["eccentricity"] / eccentric_summary["eccentricity_uncertainty"]
		return fits, significance, "keplerian" if significance > ECCENTRICITY_SIGNIFICANCE else "circular"

	fits, significance, adopted = fit_both(initial, None)
	one_planet, _, _, pivot = fits[adopted]
	detection = detect_outer_signal(
		bjd[mask], rv[mask] - orbit_rv(bjd[mask], one_planet, offset_design[mask]),
		rv_error[mask], source_labels[mask],
	)
	outer = choose_outer_signal(detection, outer_model, pivot)
	# Vet detections even when a model is forced, so the flag stays honest; only
	# "auto" drops a term that fails.
	if outer is not None and detection["flag"] in ("outer_planet", "trend"):
		accepted, detection["vetting"] = vet_outer_signal(
			outer, detection, data, one_planet, eccentric=adopted == "keplerian"
		)
		if not accepted:
			detection["flag"] = {"outer_planet": "outer_candidate", "trend": "trend_candidate"}[detection["flag"]]
			if outer_model == "auto":
				outer = None
	if outer is not None:
		fits, significance, adopted = fit_both(np.concatenate([one_planet, outer.initial()]), outer)

	def outer_summary(fits, adopted, outer):
		parameters, covariance = fits[adopted][:2]
		return outer.summarize(
			parameters[offsets_end:], covariance[offsets_end:, offsets_end:], detection["baseline_days"]
		)

	# A vetted circular outer planet is refitted as a full Keplerian; the eccentric
	# outer orbit is adopted only if its e passes the same significance test as the
	# inner planet's. Partial orbits cannot constrain e, so they stay circular.
	if isinstance(outer, Sinusoid) and outer_model == "auto":
		circular_outer = outer_summary(fits, adopted, outer)
		comparison = {"circular": circular_outer}
		if circular_outer["orbit_coverage"] == "multiple_cycles":
			companion = KeplerianCompanion(*fits[adopted][0][offsets_end:])
			eccentric_fits, eccentric_significance, eccentric_adopted = fit_both(
				np.concatenate([fits[adopted][0][:offsets_end], companion.initial()]), companion
			)
			eccentric_outer = outer_summary(eccentric_fits, eccentric_adopted, companion)
			comparison["keplerian"] = eccentric_outer
			plausible = (
				eccentric_outer["eccentricity"] <= OUTER_MAX_ECCENTRICITY
				and eccentric_outer["semiamplitude_m_per_s"]
				<= OUTER_MAX_ECCENTRIC_K_RATIO * circular_outer["semiamplitude_m_per_s"]
			)
			comparison["keplerian_plausible"] = plausible
			if plausible and eccentric_outer["eccentricity_significance"] > ECCENTRICITY_SIGNIFICANCE:
				outer, fits, significance, adopted = companion, eccentric_fits, eccentric_significance, eccentric_adopted
		else:
			comparison["keplerian"] = "not attempted: partial orbit"
		detection["outer_model_comparison"] = comparison
	adopted_parameters, adopted_covariance, adopted_summary, _ = fits[adopted]
	outer_parameters = adopted_parameters[offsets_end:] if outer is not None else None
	if outer is not None:
		detection["fit"] = outer.summarize(
			outer_parameters, adopted_covariance[offsets_end:, offsets_end:], detection["baseline_days"]
		)
	detection["model"] = outer.name if outer is not None else None

	parameters = {
		"measurement_count": len(bjd),
		"orbit_fit_excluded_count": np.count_nonzero(~mask),
		"periodogram_peak_period_days": peak_period,
		"periodogram_false_alarm_probability": FALSE_ALARM_PROBABILITY,
		"periodogram_false_alarm_threshold": false_alarm_threshold,
		"orbit_model": adopted,
		"eccentricity_significance": significance,
		**adopted_summary,
		"source_offsets_m_per_s": dict(zip(offset_sources, adopted_parameters[ORBIT_OFFSETS:offsets_end])),
		"outer_signal": detection,
		"circular_fit": {key: value for key, value in fits["circular"][2].items()
			if not key.startswith(("eccentricity", "omega", "periapsis"))},
		"keplerian_fit": fits["keplerian"][2],
	}
	outer_rv = outer.evaluate(bjd, outer_parameters) if outer is not None else 0.0
	return SystemFit(
		parameters=_to_json(parameters),
		periods=periods,
		power=power,
		orbit_parameters=adopted_parameters[:ORBIT_OFFSETS],
		offset_corrected_rv=rv - offset_design @ adopted_parameters[ORBIT_OFFSETS:offsets_end] - outer_rv,
		orbit_fit_mask=mask,
		outer=outer,
		outer_parameters=outer_parameters,
	)


def _to_json(value):
	"""Convert numpy scalars to JSON types, mapping NaN to None."""
	if isinstance(value, dict):
		return {key: _to_json(item) for key, item in value.items()}
	if isinstance(value, (bool, np.bool_)):
		return bool(value)
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


def describe_model(parameters: dict, outer: OuterSignal | None) -> str:
	"""Plain-language description of the adopted RV model, for figure titles."""
	inner = "eccentric Keplerian" if parameters["orbit_model"] == "keplerian" else "circular"
	parts = [f"{inner} inner planet"]
	if outer is not None:
		forced = parameters["outer_signal"]["flag"] not in ("outer_planet", "trend")
		parts.append(outer.describe() + (" (candidate, forced fit)" if forced else ""))
	offsets = len(parameters["source_offsets_m_per_s"])
	parts.append(f"{offsets} instrument offset{'s' if offsets != 1 else ''} and per-instrument jitter")
	return " + ".join(parts)


def _robust_ylim(axis, values: np.ndarray, errors: np.ndarray) -> None:
	"""Limit the y-range to the bulk of the fitted points so failed measurements
	with huge error bars do not flatten the plot."""
	low, high = np.percentile(values, [0.5, 99.5])
	pad = 0.1 * (high - low) + 2 * np.median(errors)
	axis.set_ylim(low - pad, high + pad)


def _season_means(bjd, values, sigma):
	"""Inverse-variance weighted means (and errors) in SEASON_DAYS bins with >= 3 points."""
	season = np.floor((bjd - bjd.min()) / SEASON_DAYS)
	rows = []
	for value in np.unique(season):
		in_season = season == value
		if in_season.sum() >= 3:
			weights = sigma[in_season] ** -2
			rows.append((
				np.average(bjd[in_season], weights=weights),
				np.average(values[in_season], weights=weights),
				1 / np.sqrt(weights.sum()),
			))
	return np.array(rows).reshape(-1, 3)


def plot_fit(
	star: str,
	bjd: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	source_labels: np.ndarray,
	fit: SystemFit,
) -> plt.Figure:
	"""Plot the RVs, periodogram, inner-planet fold, and (if any) the outer signal.

	The fourth panel shows a fitted outer planet folded on its period, a fitted
	trend against time, or, for outer/trend candidates that are not in the fit,
	the inner-orbit residuals with seasonal means (and the candidate trend).
	"""
	parameters = fit.parameters
	outer_signal = parameters["outer_signal"]
	flag = outer_signal["flag"]
	period = parameters["period_days"]
	period_label = f"{period:.6g} +/- {parameters['period_uncertainty_days']:.2g}"
	semiamplitude = parameters["semiamplitude_m_per_s"]
	semiamplitude_uncertainty = parameters["semiamplitude_uncertainty_m_per_s"]
	conjunction = parameters["conjunction_bjd"]
	inner_name = (
		f"eccentric Keplerian, e = {parameters['eccentricity']:.3f} +/- {parameters['eccentricity_uncertainty']:.3f}"
		if parameters["orbit_model"] == "keplerian" else "circular orbit"
	)
	mask = fit.orbit_fit_mask
	no_offsets = np.empty((len(bjd), 0))
	outer_rv = fit.outer.evaluate(bjd, fit.outer_parameters) if fit.outer is not None else 0.0
	inner_residuals = fit.offset_corrected_rv + outer_rv - orbit_rv(bjd, fit.orbit_parameters, no_offsets)
	sigma = np.hypot(rv_error, [parameters["jitter_m_per_s"].get(label, 0.0) for label in source_labels])
	show_outer_panel = fit.outer is not None or flag in ("outer_candidate", "trend_candidate")

	panels = 4 if show_outer_panel else 3
	figure, axes = plt.subplots(panels, 1, sharex=False, figsize=(8, 3.9 * panels + 0.6), layout="constrained")
	figure.suptitle(
		f"{star}\n" + textwrap.fill(f"Model: {describe_model(parameters, fit.outer)}", 90), fontsize="medium"
	)

	_errorbar_by_source(axes[0], bjd - 2_450_000, rv, rv_error, source_labels)
	axes[0].set_xlabel("BJD - 2450000")
	axes[0].set_ylabel("Radial velocity (m/s)")
	axes[0].set_title(f"Combined RVs as published ({len(bjd)} measurements, offsets not removed)")
	axes[0].grid(alpha=0.3)
	axes[0].legend(fontsize="small")
	_robust_ylim(axes[0], rv[mask], rv_error[mask])
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
	axes[1].axvline(period, color="tab:red", linestyle="--", label=f"Inner planet, P = {period:.6g} d")
	axes[1].axhline(
		parameters["periodogram_false_alarm_threshold"],
		color="tab:purple",
		linestyle=":",
		label=f"{FALSE_ALARM_PROBABILITY:.1%} false-alarm threshold",
	)
	axes[1].set_xscale("log")
	axes[1].set_xlabel("Period (days)")
	axes[1].set_ylabel("Lomb-Scargle power")
	axes[1].set_title("Error-weighted Lomb-Scargle periodogram of the combined RVs")
	axes[1].grid(alpha=0.3)
	axes[1].legend(fontsize="small")

	# Fold with the transit conjunction at phase 0.
	phase = ((bjd - conjunction) / period + 0.25) % 1.0 - 0.25
	_errorbar_by_source(axes[2], phase, fit.offset_corrected_rv, rv_error, source_labels, mask)
	_errorbar_by_source(
		axes[2], phase, fit.offset_corrected_rv, rv_error, source_labels, ~mask,
		label_suffix=" (excluded)", alpha=0.25,
	)
	model_phase = np.linspace(-0.25, 0.75, 500)
	axes[2].plot(
		model_phase,
		orbit_rv(conjunction + model_phase * period, fit.orbit_parameters, np.empty((len(model_phase), 0))),
		color="black", linewidth=2, zorder=5,
		label="Inner-planet model",
	)
	axes[2].axvline(0.0, color="tab:green", linestyle="--", label="Transit conjunction")
	axes[2].set_xlabel("Inner-planet orbital phase from transit conjunction")
	axes[2].set_ylabel("Radial velocity (m/s)")
	removed = "offsets and outer signal removed" if fit.outer is not None else "offsets removed"
	axes[2].set_title(
		f"Inner planet: {inner_name}, K = {semiamplitude:.4g} +/- {semiamplitude_uncertainty:.2g} m/s"
		f"\nfolded on P = {period_label} d; {removed}"
	)
	axes[2].set_xlim(-0.25, 0.75)
	_robust_ylim(axes[2], fit.offset_corrected_rv[mask], rv_error[mask])
	axes[2].grid(alpha=0.3)
	axes[2].legend(fontsize="small", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))

	if show_outer_panel:
		axis = axes[3]
		outer = outer_signal.get("fit")
		if fit.outer is not None and outer["model"] in ("sinusoid", "keplerian"):
			outer_period, outer_conjunction = outer["period_days"], outer["conjunction_bjd"]
			outer_phase = ((bjd - outer_conjunction) / outer_period + 0.25) % 1.0 - 0.25
			_errorbar_by_source(axis, outer_phase, inner_residuals, rv_error, source_labels, mask)
			grid = np.linspace(-0.25, 0.75, 500)
			if outer["model"] == "keplerian":
				outer_name = (
					f"eccentric Keplerian, e = {outer['eccentricity']:.2f} +/- {outer['eccentricity_uncertainty']:.2f}"
				)
			else:
				outer_name = "circular orbit"
			axis.plot(
				grid, fit.outer.evaluate(outer_conjunction + grid * outer_period, fit.outer_parameters),
				color="black", linewidth=2, zorder=5,
				label="Outer-planet model",
			)
			partial = "\npartial orbit: P and K are effectively lower limits" if outer["orbit_coverage"] == "partial_orbit" else ""
			axis.set_xlim(-0.25, 0.75)
			axis.set_xlabel("Outer-planet orbital phase from conjunction")
			candidate = "Candidate outer planet (forced fit, not vetted)" if flag not in ("outer_planet", "trend") else "Outer planet"
			axis.set_title(
				f"{candidate}: {outer_name}, K = {outer['semiamplitude_m_per_s']:.3g}"
				f" +/- {outer['semiamplitude_uncertainty_m_per_s']:.2g} m/s\nfolded on"
				f" P = {outer_period:.5g} +/- {outer['period_uncertainty_days']:.2g} d; inner orbit removed{partial}"
			)
		else:
			_errorbar_by_source(axis, bjd - 2_450_000, inner_residuals, rv_error, source_labels, mask, alpha=0.5)
			seasons = _season_means(bjd[mask], inner_residuals[mask], sigma[mask])
			axis.errorbar(
				seasons[:, 0] - 2_450_000, seasons[:, 1], yerr=seasons[:, 2], fmt="s", color="black",
				markersize=4, capsize=2, zorder=6, label=f"{SEASON_DAYS:.0f}-day seasonal means",
			)
			grid = np.linspace(bjd.min(), bjd.max(), 1000)
			if fit.outer is not None:  # fitted trend
				axis.plot(grid - 2_450_000, fit.outer.evaluate(grid, fit.outer_parameters),
					color="black", linewidth=2, zorder=5, label="Fitted trend")
				trend_kind = "Candidate quadratic trend (forced fit, not vetted)" if flag != "trend" else "Quadratic trend"
				axis.set_title(
					f"{trend_kind}, fitted jointly with the inner orbit: {outer['slope_m_per_s_per_yr']:.3g}"
					f" +/- {outer['slope_uncertainty_m_per_s_per_yr']:.2g} m/s/yr,\ncurvature"
					f" {outer['curvature_m_per_s_per_yr2']:.3g} +/- {outer['curvature_uncertainty_m_per_s_per_yr2']:.2g} m/s/yr^2;"
					" residuals from the inner orbit"
				)
			else:
				reasons = "; ".join(outer_signal.get("vetting", {}).get("rejection_reasons", []))
				peak = outer_signal["long_period_peak"]
				if flag == "trend_candidate":
					# Display only: quadratic through the residuals shown (the scan's own
					# trend was fitted with separate group offsets).
					years = (bjd[mask] - bjd[mask].mean()) / DAYS_PER_YEAR
					coefficients = np.polyfit(years, inner_residuals[mask], 2, w=1 / sigma[mask])
					axis.plot(
						grid - 2_450_000, np.polyval(coefficients, (grid - bjd[mask].mean()) / DAYS_PER_YEAR),
						color="black", linestyle="--", linewidth=2, zorder=5,
						label="Quadratic through these residuals (not in the orbit fit)",
					)
				axis.set_title(
					f"{flag.replace('_', ' ').capitalize()}, not included in the fit: residual peak"
					f" P = {peak['period_days']:.0f} d, K = {peak['semiamplitude_m_per_s']:.1f} m/s\n({reasons})",
					fontsize="medium",
				)
			axis.set_xlabel("BJD - 2450000")
		axis.set_ylabel("Residual from inner orbit (m/s)")
		_robust_ylim(axis, inner_residuals[mask], rv_error[mask])
		axis.grid(alpha=0.3)
		axis.legend(fontsize="small", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
	return figure


def _outer_summary(outer: dict) -> str:
	"""One-line description of the outer-signal flag and fitted term."""
	text = f"outer-signal flag = {outer['flag']}"
	fit = outer.get("fit")
	if fit and fit["model"] in ("sinusoid", "keplerian"):
		shape = (
			f"eccentric Keplerian (e = {fit['eccentricity']:.3f} +/- {fit['eccentricity_uncertainty']:.3f},"
			f" omega = {fit['omega_degrees']:.0f} deg)" if fit["model"] == "keplerian" else "circular orbit"
		)
		text += (
			f"; fitted outer planet, {shape}: P = {fit['period_days']:.0f} +/- {fit['period_uncertainty_days']:.0f} d,"
			f" K = {fit['semiamplitude_m_per_s']:.1f} +/- {fit['semiamplitude_uncertainty_m_per_s']:.1f} m/s,"
			f" m sin i = {fit['msini_mjup_per_mstar_msun_2_3']:.2f} Mjup (M*/Msun)^(2/3)"
			+ (" [partial orbit: P and K effectively lower limits]" if fit["orbit_coverage"] == "partial_orbit" else "")
		)
		comparison = outer.get("outer_model_comparison", {}).get("keplerian")
		if isinstance(comparison, dict) and fit["model"] == "sinusoid":
			text += f" (eccentric outer fit not preferred: e/sigma_e = {comparison['eccentricity_significance']:.1f})"
	elif fit and fit["model"] == "trend":
		text += (
			f"; fitted trend {fit['slope_m_per_s_per_yr']:.2f} +/- {fit['slope_uncertainty_m_per_s_per_yr']:.2f} m/s/yr"
			f" (P > {fit['period_lower_limit_days']:.0f} d, K > {fit['semiamplitude_lower_limit_m_per_s']:.0f} m/s,"
			f" m sin i > {fit['min_mass_mjup_per_au2']:.2f} Mjup (a/AU)^2)"
		)
	elif outer["flag"] == "possible_activity":
		text += f" (residual peak at {outer['short_period_peak']['period_days']:.1f} d)"
	elif outer["flag"] in ("outer_candidate", "trend_candidate"):
		peak = outer["long_period_peak"]
		text += (
			f" (residual peak P = {peak['period_days']:.0f} d, K = {peak['semiamplitude_m_per_s']:.1f} m/s;"
			f" not fitted: {'; '.join(outer['vetting']['rejection_reasons'])})"
		)
	return text


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
		_outer_summary(parameters["outer_signal"]),
		f"reduced chi2 (with jitter) = {value('reduced_chi_squared')}",
	)
	for line in lines:
		print(f"{star}: {line}")


def save_fit(
	star: str,
	source: str,
	bjd: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	source_labels: np.ndarray,
	fit: SystemFit,
) -> tuple[Path, Path]:
	"""Save the fit plot and parameter JSON to plots/; return (plot, JSON) paths."""
	includes_synthetics = bool(np.any(source_labels == "Synthetic"))
	fit_parameters = {
		"input_star": star,
		"source": source,
		"includes_synthetics": includes_synthetics,
		**fit.parameters,
	}
	figure = plot_fit(star, bjd, rv, rv_error, source_labels, fit)
	PLOTS_DIRECTORY.mkdir(exist_ok=True)
	# Tag synthetic-inclusive fits so they never overwrite real-data fits, which
	# generate_synthetic_rvs reads back by star name.
	output_stem = f"{star}{SYNTHETIC_SUFFIX if includes_synthetics else ''}"
	plot_path = PLOTS_DIRECTORY / f"{output_stem}_rv_fit_plot.png"
	figure.savefig(str(plot_path), dpi=150)
	json_path = PLOTS_DIRECTORY / f"{output_stem}_rv_fit_parameters.json"
	json_path.write_text(json.dumps(fit_parameters, indent=2) + "\n", encoding="utf-8")
	return plot_path, json_path


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
		"--outer-model",
		choices=OUTER_MODELS,
		default="auto",
		help="Outer-signal term: added when flagged (auto), never (none), or forced"
		" (sinusoid = circular outer planet, keplerian = eccentric outer planet, trend).",
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

	fit = fit_system(bjd, rv, rv_error, source_labels, args.outer_model)
	print_fit_summary(args.star, fit.parameters)
	for path in save_fit(args.star, args.source, bjd, rv, rv_error, source_labels, fit):
		print(f"Saved {'plot' if path.suffix == '.png' else 'fit parameters'} to {path}")
	plt.show()


if __name__ == "__main__":
	main()
