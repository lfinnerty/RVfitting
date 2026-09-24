#!/usr/bin/env python3
"""Plot combined HIRES radial velocities for a star."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astroquery.simbad import Simbad
from scipy.optimize import least_squares
from scipy.signal import lombscargle


DATABASE = Path(__file__).resolve().parent / "RVdatabases" / "tablea1_Teklu.dat"
EXOARCHIVE_DATABASE = DATABASE.parent / "exoarchive"
FULTON_DATABASE = DATABASE.parent / "rv_data_fulton"
EXOARCHIVE_COORDINATES = DATABASE.parent / "exoarchive_host_coordinates.txt"
SOURCE_COLORS = {
	"Teklu": "tab:blue",
	"ExoArchive": "tab:orange",
	"Fulton": "tab:green",
}


def normalize_identifier(identifier: str) -> str:
	"""Normalize catalog identifiers for case- and whitespace-insensitive matching."""
	return "".join(identifier.casefold().split())


def read_star_rvs(database: Path, star: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return BJD, NZP-corrected RV, and RV error for ``star``."""
	target = normalize_identifier(star)
	times = []
	velocities = []
	errors = []

	with database.open(encoding="ascii") as data_file:
		for line in data_file:
			name = line[0:14].strip()
			simbad_name = line[15:45].strip()
			if target not in (normalize_identifier(name), normalize_identifier(simbad_name)):
				continue

			# The ReadMe says km/s, but the catalog RV values are in m/s.
			# The byte ranges follow the tablea1_Teklu.dat description in ReadMe.
			bjd_text = line[82:95].strip()
			rv_text = line[132:148].strip()
			error_text = line[149:156].strip()
			if "-" in (bjd_text, rv_text, error_text):
				continue

			times.append(float(bjd_text))
			velocities.append(float(rv_text))
			errors.append(float(error_text))

	if not times:
		raise ValueError(f"No usable RV measurements found for {star!r}")

	order = np.argsort(times)
	return (
		np.asarray(times)[order],
		np.asarray(velocities)[order],
		np.asarray(errors)[order],
	)


def read_exoarchive_rvs(
	database: Path, star: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return ExoArchive RV points in Teklu's systemic-velocity convention."""
	target = normalize_identifier(star)
	times = []
	velocities = []
	errors = []
	skipped_time_frame = 0
	skipped_site = 0
	target_coordinates = None

	for path in sorted(database.glob("*.tbl")):
		text = path.read_text(encoding="utf-8")
		match = re.search(r"\\STAR_ID\s*=\s*[\"']([^\"']+)[\"']", text)
		if match is None or normalize_identifier(match.group(1)) != target:
			continue

		velocity_definition = _exoarchive_header_value(
			text, "COLUMN_RADIAL_VELOCITY"
		)
		if velocity_definition is None:
			continue
		already_barycentric = "relative to barycenter" in velocity_definition.casefold()

		date_units = _exoarchive_header_value(text, "DATE_UNITS")
		time_frame = _exoarchive_header_value(text, "TIME_REFERENCE_FRAME")
		if date_units is None or date_units.casefold() != "days" or time_frame is None:
			skipped_time_frame += 1
			continue
		time_frame = time_frame.upper()
		if time_frame not in {"BJD", "BJD-UTC", "BJD-TDB", "JD", "JD-UTC", "HJD", "HJD-UTC", "HJD-TBD", "FCJD", "MJD"}:
			skipped_time_frame += 1
			continue

		if target_coordinates is None:
			target_coordinates = find_simbad_coordinates(star)
			if target_coordinates is None:
				raise ValueError(f"Could not resolve coordinates for ExoArchive target {star!r}")
		observatory_site = _exoarchive_header_value(text, "OBSERVATORY_SITE")
		location = exoarchive_location(observatory_site)
		if location is None:
			skipped_site += 1
			continue

		file_times = []
		file_velocities = []
		file_errors = []
		for line in text.splitlines():
			if not line.strip() or line.lstrip().startswith(("\\", "|")):
				continue
			fields = line.split()
			if len(fields) < 3:
				continue
			try:
				observation_time, velocity, error = map(float, fields[:3])
			except ValueError:
				continue
			file_times.append(_exoarchive_time_to_bjd(observation_time, time_frame))
			file_velocities.append(velocity)
			file_errors.append(error)

		if not already_barycentric:
			observation_times = Time(
				file_times,
				format="jd",
				scale=_exoarchive_time_scale(time_frame),
			)
			correction = target_coordinates.radial_velocity_correction(
				obstime=observation_times,
				location=location,
			).to_value(u.m / u.s)
			file_velocities = (
				np.asarray(file_velocities) + correction
			).tolist()

		# Teklu reports velocities relative to the host systemic velocity.
		file_velocities = (
			np.asarray(file_velocities) - np.median(file_velocities)
		).tolist()
		times.extend(file_times)
		velocities.extend(file_velocities)
		errors.extend(file_errors)

	if skipped_time_frame or skipped_site:
		print(
			f"{star}: skipped {skipped_time_frame} ExoArchive file(s)"
			f" with unsupported date frames and {skipped_site} with unknown sites"
		)

	return _sort_rv_arrays(times, velocities, errors)


def _exoarchive_header_value(text: str, field: str) -> str | None:
	"""Return one quoted ExoArchive header value."""
	match = re.search(
		rf"^\\{field}\s*=\s*[\"']?([^\"'\n]+)", text, re.MULTILINE
	)
	return match.group(1).strip() if match is not None else None


def _exoarchive_time_to_bjd(observation_time: float, time_frame: str) -> float:
	"""Convert a supported ExoArchive Julian-date value to BJD scale."""
	if time_frame.startswith("MJD") and observation_time < 1_000_000:
		return observation_time + 2_400_000.5
	return observation_time


def _exoarchive_time_scale(time_frame: str) -> str:
	"""Return the Astropy time scale indicated by an ExoArchive frame."""
	if time_frame.endswith("UTC"):
		return "utc"
	return "tdb"


def find_simbad_coordinates(star: str) -> SkyCoord | None:
	"""Return sky coordinates for an ExoArchive host star."""
	coordinates = load_cached_coordinates()
	cached_coordinates = coordinates.get(normalize_identifier(star))
	if cached_coordinates is not None:
		return cached_coordinates

	try:
		result = Simbad().query_object(star)
	except Exception:
		return None
	if result is None or len(result) == 0:
		return None
	coordinates = SkyCoord(result["ra"][0], result["dec"][0], unit=u.deg)
	with EXOARCHIVE_COORDINATES.open("a", encoding="utf-8", newline="") as cache:
		if EXOARCHIVE_COORDINATES.stat().st_size == 0:
			cache.write("host\tra_deg\tdec_deg\n")
		cache.write(f"{star}\t{coordinates.ra.deg:.12f}\t{coordinates.dec.deg:.12f}\n")
	return coordinates


def load_cached_coordinates() -> dict[str, SkyCoord]:
	"""Load cached ExoArchive host coordinates keyed by normalized name."""
	if not EXOARCHIVE_COORDINATES.is_file():
		return {}
	coordinates = {}
	with EXOARCHIVE_COORDINATES.open(encoding="utf-8") as cache:
		for line in cache:
			fields = line.rstrip().split("\t")
			if len(fields) != 3 or fields[0] == "host":
				continue
			try:
				coordinates[normalize_identifier(fields[0])] = SkyCoord(
					float(fields[1]), float(fields[2]), unit=u.deg
				)
			except ValueError:
				continue
	return coordinates


def exoarchive_location(site: str | None) -> EarthLocation | None:
	"""Resolve common ExoArchive observatory labels to Earth locations."""
	if not site:
		return None
	name = site.casefold()
	aliases = (
		("mauna kea", "keck"),
		("manua kea", "keck"),
		("maun kea", "keck"),
		("lick", "lick observatory"),
		("la silla", "La Silla Observatory"),
		("las campanas", "Las Campanas Observatory"),
		("mcdonald", "McDonald Observatory"),
		("apache point", "Apache Point Observatory"),
		("siding spring", "Siding Spring Observatory"),
		("siding springs", "Siding Spring Observatory"),
		("okayama", "Okayama Astrophysical Observatory"),
		("paranal", "Cerro Paranal"),
		("roque de los muchachos", "Roque de los Muchachos"),
		("la palma", "Roque de los Muchachos"),
		("whipple", "Whipple Observatory"),
		("cal ar alto", "Observatorio de Calar Alto"),
		("kitt peak", "Kitt Peak National Observatory"),
		("xinglong", "Beijing XingLong Observatory"),
	)
	for fragment, location_name in aliases:
		if fragment in name:
			try:
				return EarthLocation.of_site(location_name)
			except Exception:
				return None
	try:
		return EarthLocation.of_site(site)
	except Exception:
		return None


def read_fulton_rvs(
	database: Path, star: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return Fulton RV points for ``star``."""
	target = normalize_identifier(star)
	times = []
	velocities = []
	errors = []

	for path in sorted(database.glob("*_rv.csv")):
		with path.open(encoding="utf-8", newline="") as data_file:
			first_line = data_file.readline()
			match = re.fullmatch(r"# star HD number,\s*(\d+)\s*\n?", first_line)
			if match is None or normalize_identifier(f"HD {match.group(1)}") != target:
				continue

			reader = csv.reader(data_file)
			for row in reader:
				if len(row) < 3:
					continue
				try:
					observation_time, velocity, error = map(float, row[:3])
				except ValueError:
					continue
				times.append(observation_time + 2_440_000)
				velocities.append(velocity)
				errors.append(error)

	return _sort_rv_arrays(times, velocities, errors)


def _sort_rv_arrays(
	times: list[float], velocities: list[float], errors: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Convert RV lists to time-sorted arrays."""
	if not times:
		return np.array([]), np.array([]), np.array([])
	order = np.argsort(times)
	return (
		np.asarray(times)[order],
		np.asarray(velocities)[order],
		np.asarray(errors)[order],
	)


def combine_rv_data(
	datasets: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
	duplicate_tolerance: float = 300 / 86_400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Combine source datasets, dropping observations sharing a timestamp."""
	points = []
	for source, (times, velocities, errors) in datasets:
		points.extend(
			(time, velocity, error, source)
			for time, velocity, error in zip(times, velocities, errors)
		)

	points.sort(key=lambda point: point[0])
	unique_points = []
	for point in points:
		if unique_points and point[0] - unique_points[-1][0] <= duplicate_tolerance:
			continue
		unique_points.append(point)

	if not unique_points:
		raise ValueError("No RV measurements found in any database")
	return (
		np.asarray([point[0] for point in unique_points]),
		np.asarray([point[1] for point in unique_points]),
		np.asarray([point[2] for point in unique_points]),
		np.asarray([point[3] for point in unique_points]),
	)


def find_simbad_database_id(database: Path, star: str) -> str | None:
	"""Return a database ID matching one of the star's SIMBAD identifiers."""
	database_ids = {}
	with database.open(encoding="ascii") as data_file:
		for line in data_file:
			name = line[0:14].strip()
			simbad_name = line[15:45].strip()
			if name:
				database_ids[normalize_identifier(name)] = name
			if simbad_name:
				database_ids[normalize_identifier(simbad_name)] = name

	try:
		simbad = Simbad()
		simbad.add_votable_fields("ids")
		result = simbad.query_object(star)
	except Exception:
		return None
	if result is None or len(result) == 0 or result["ids"][0] is None:
		return None

	identifiers = [identifier.strip() for identifier in str(result["ids"][0]).split("|")]
	print(f"{star}: SIMBAD alternative IDs: {identifiers}")
	for identifier in identifiers:
		match = database_ids.get(normalize_identifier(identifier))
		if match is not None:
			return match
	return None


def calculate_periodogram(
	bjd: np.ndarray, rv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
	"""Return trial periods, Lomb-Scargle powers, and the peak period in days."""
	minimum_period = 1.0
	maximum_period = np.nextafter(8.0, minimum_period)
	periods = np.geomspace(minimum_period, maximum_period, 40_000)
	angular_frequencies = 2 * np.pi / periods
	power = lombscargle(
		bjd,
		rv,
		angular_frequencies,
		precenter=True,
		normalize=True,
	)
	peak_period = periods[np.argmax(power)]
	return periods, power, peak_period


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


def fit_period(
	bjd: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	initial_period: float,
	source_labels: np.ndarray,
) -> tuple[float, float]:
	"""Refine the period and return its 1-sigma weighted-fit uncertainty."""
	reference_bjd = bjd[0]
	initial_phase = (bjd - reference_bjd) / initial_period
	phase_design = np.column_stack(
		[
			np.ones_like(initial_phase),
			np.sin(2 * np.pi * initial_phase),
			np.cos(2 * np.pi * initial_phase),
		]
	)
	offset_design, _ = source_offset_design(source_labels)
	initial_design = np.column_stack([phase_design, offset_design])
	initial_coefficients = np.linalg.lstsq(
		initial_design / rv_error[:, np.newaxis], rv / rv_error, rcond=None
	)[0]

	def residuals(parameters: np.ndarray) -> np.ndarray:
		period = parameters[-1]
		phase = (bjd - reference_bjd) / period
		phase_design = np.column_stack(
			[
				np.ones_like(phase),
				np.sin(2 * np.pi * phase),
				np.cos(2 * np.pi * phase),
			]
		)
		design = np.column_stack([phase_design, offset_design])
		return (rv - design @ parameters[:-1]) / rv_error

	minimum_period = 1.0
	maximum_period = np.nextafter(8.0, minimum_period)
	fit = least_squares(
		residuals,
		np.append(initial_coefficients, initial_period),
		bounds=(
			[-np.inf] * initial_design.shape[1] + [minimum_period],
			[np.inf] * initial_design.shape[1] + [maximum_period],
		),
		x_scale="jac",
	)
	degrees_of_freedom = len(rv) - len(fit.x)
	residual_variance = np.sum(fit.fun**2) / degrees_of_freedom
	covariance = residual_variance * np.linalg.inv(fit.jac.T @ fit.jac)
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
	bjd: np.ndarray,
	reference_bjd: float,
	period: float,
	parameters: np.ndarray,
	source_labels: np.ndarray,
) -> np.ndarray:
	"""Evaluate the Keplerian RV model for the supplied observation times."""
	phase = (bjd - reference_bjd) / period
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
	offset_design, _ = source_offset_design(source_labels)
	return model + offset_design @ parameters[5:]


def fit_keplerian(
	bjd: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	period: float,
	source_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Fit a Keplerian RV model and return parameters and covariance."""
	reference_bjd = bjd[0]
	phase = (bjd - reference_bjd) / period
	phase_design = np.column_stack(
		[np.ones_like(phase), np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
	)
	offset_design, _ = source_offset_design(source_labels)
	design = np.column_stack([phase_design, offset_design])
	circular_coefficients = np.linalg.lstsq(
		design / rv_error[:, np.newaxis], rv / rv_error, rcond=None
	)[0]
	initial_semiamplitude = np.hypot(circular_coefficients[1], circular_coefficients[2])
	initial_periapsis_phase = (
		np.arctan2(circular_coefficients[1], circular_coefficients[2]) / (2 * np.pi)
	) % 1.0
	initial_parameters = np.append(
		[
			circular_coefficients[0],
			initial_semiamplitude,
			0.05,
			0.0,
			initial_periapsis_phase,
		],
		np.zeros(offset_design.shape[1]),
	)

	def residuals(parameters: np.ndarray) -> np.ndarray:
		model = calculate_keplerian_rv(
			bjd, reference_bjd, period, parameters, source_labels
		)
		return (rv - model) / rv_error

	fit = least_squares(
		residuals,
		initial_parameters,
		bounds=(
			[-np.inf, 0.0, 0.0, -np.pi, 0.0]
			+ [-np.inf] * offset_design.shape[1],
			[np.inf, np.inf, 0.95, np.pi, 1.0]
			+ [np.inf] * offset_design.shape[1],
		),
		x_scale="jac",
		max_nfev=2_000,
	)
	degrees_of_freedom = len(rv) - len(fit.x)
	residual_variance = np.sum(fit.fun**2) / degrees_of_freedom
	covariance = residual_variance * np.linalg.pinv(fit.jac.T @ fit.jac)
	return fit.x, covariance


def fit_phase_curve(
	phase: np.ndarray,
	rv: np.ndarray,
	rv_error: np.ndarray,
	source_labels: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
	"""Fit a weighted sinusoid and return coefficients, amplitude, and covariance."""
	phase_design = np.column_stack(
		[np.ones_like(phase), np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
	)
	offset_design, _ = source_offset_design(source_labels)
	design = np.column_stack([phase_design, offset_design])
	weights = 1.0 / rv_error
	coefficients, _, _, _ = np.linalg.lstsq(
		design * weights[:, np.newaxis], rv * weights, rcond=None
	)
	weighted_design = design * weights[:, np.newaxis]
	residuals = rv - design @ coefficients
	degrees_of_freedom = len(rv) - design.shape[1]
	residual_variance = np.sum((residuals * weights) ** 2) / degrees_of_freedom
	coefficient_covariance = residual_variance * np.linalg.inv(
		weighted_design.T @ weighted_design
	)
	semiamplitude = np.hypot(coefficients[1], coefficients[2])
	return coefficients, semiamplitude, coefficient_covariance


def calculate_conjunction_time(
	bjd_reference: float,
	period: float,
	coefficients: np.ndarray,
	coefficient_covariance: np.ndarray,
) -> tuple[float, float]:
	"""Return decreasing RV zero-crossing BJD and its fit-only uncertainty."""
	sine_coefficient, cosine_coefficient = coefficients[1:3]
	phase_angle = np.arctan2(cosine_coefficient, sine_coefficient)
	conjunction_phase = (0.5 - phase_angle / (2 * np.pi)) % 1.0
	conjunction_bjd = bjd_reference + conjunction_phase * period

	amplitude_squared = sine_coefficient**2 + cosine_coefficient**2
	phase_angle_variance = (
		cosine_coefficient**2 * coefficient_covariance[1, 1]
		+ sine_coefficient**2 * coefficient_covariance[2, 2]
		- 2 * sine_coefficient * cosine_coefficient * coefficient_covariance[1, 2]
	) / amplitude_squared**2
	conjunction_uncertainty = period * np.sqrt(phase_angle_variance) / (2 * np.pi)
	return conjunction_bjd, conjunction_uncertainty


def calculate_semiamplitude_uncertainty(
	coefficients: np.ndarray, coefficient_covariance: np.ndarray
) -> float:
	"""Return the 1-sigma uncertainty on the sinusoidal semiamplitude."""
	sine_coefficient, cosine_coefficient = coefficients[1:3]
	semiamplitude = np.hypot(sine_coefficient, cosine_coefficient)
	variance = (
		sine_coefficient**2 * coefficient_covariance[1, 1]
		+ cosine_coefficient**2 * coefficient_covariance[2, 2]
		+ 2 * sine_coefficient * cosine_coefficient * coefficient_covariance[1, 2]
	) / semiamplitude**2
	return np.sqrt(variance)


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
	source_labels: np.ndarray,
) -> np.ndarray:
	"""Return a mask excluding obvious 5-sigma residual outliers."""
	mask = np.ones(len(rv), dtype=bool)
	for _ in range(3):
		coefficients, _, _ = fit_phase_curve(
			phase[mask], rv[mask], rv_error[mask], source_labels[mask]
		)
		model = coefficients[0] + coefficients[1] * np.sin(2 * np.pi * phase)
		model += coefficients[2] * np.cos(2 * np.pi * phase)
		offset_design, _ = source_offset_design(source_labels)
		model += offset_design @ coefficients[3:]
		residuals = rv - model
		center = np.median(residuals[mask])
		mad = np.median(np.abs(residuals[mask] - center))
		robust_scale = 1.4826 * mad
		scale = max(robust_scale, np.median(rv_error[mask]))
		new_mask = np.abs(residuals - center) <= 5 * scale
		if np.array_equal(new_mask, mask):
			break
		mask = new_mask
	return mask


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot combined HIRES radial velocities for a star."
	)
	parser.add_argument("star", help="Catalog or SIMBAD name, for example HD10700")
	parser.add_argument(
		"--database",
		type=Path,
		default=DATABASE,
		help=f"Path to tablea1_Teklu.dat (default: {DATABASE})",
	)
	parser.add_argument(
		"--source",
		choices=("all", "teklu", "exoarchive", "fulton"),
		default="all",
		help="RV source to load (default: all)",
	)
	args = parser.parse_args()

	if args.source in ("all", "teklu") and not args.database.is_file():
		parser.error(f"Database file not found: {args.database}")

	datasets = []
	database_star = args.star
	external_star = args.star
	if args.source in ("all", "teklu"):
		try:
			teklu_data = read_star_rvs(args.database, args.star)
		except ValueError as error:
			database_star = find_simbad_database_id(args.database, args.star)
			if database_star is None:
				if args.source == "teklu":
					parser.error(str(error))
				print(f"{args.star}: no matching Teklu RV data found")
				teklu_data = None
			else:
				print(f"{args.star}: SIMBAD identifier found in database as {database_star}")
				try:
					teklu_data = read_star_rvs(args.database, database_star)
				except ValueError:
					teklu_data = None
		if teklu_data is not None:
			datasets.append(("Teklu", teklu_data))

	if args.source in ("all", "exoarchive"):
		datasets.append(
		("ExoArchive", read_exoarchive_rvs(EXOARCHIVE_DATABASE, external_star))
	)
	if args.source in ("all", "fulton"):
		datasets.append(("Fulton", read_fulton_rvs(FULTON_DATABASE, external_star)))
	raw_measurement_count = sum(len(data[0]) for _, data in datasets)
	try:
		bjd, rv, rv_error, source_labels = combine_rv_data(datasets)
	except ValueError as error:
		parser.error(str(error))
	print(
		f"{args.star}: loaded {len(bjd)} RV measurements from {args.source} source(s)"
		f" ({raw_measurement_count - len(bjd)} duplicate(s) removed)"
	)

	periods, power, peak_period = calculate_periodogram(bjd, rv)
	period, period_uncertainty = fit_period(
		bjd, rv, rv_error, peak_period, source_labels
	)
	print(
		f"{args.star}: best-fit period = "
		f"{period:.6g} +/- {period_uncertainty:.6g} days"
	)

	phase = ((bjd - bjd[0]) / period) % 1.0
	orbit_fit_mask = exclude_orbit_fit_outliers(
		phase, rv, rv_error, source_labels
	)
	if not np.all(orbit_fit_mask):
		print(
			f"{args.star}: excluding "
			f"{np.count_nonzero(~orbit_fit_mask)} obvious outlier(s) from orbit fit"
		)
	fit_coefficients, semiamplitude, fit_covariance = fit_phase_curve(
		phase, rv, rv_error, source_labels
	)
	semiamplitude_uncertainty = calculate_semiamplitude_uncertainty(
		fit_coefficients, fit_covariance
	)
	conjunction_bjd, conjunction_uncertainty = calculate_conjunction_time(
		bjd[0], period, fit_coefficients, fit_covariance
	)
	fit_phase = np.linspace(0, 1, 500)
	fit_design = np.column_stack(
		[
			np.ones_like(fit_phase),
			np.sin(2 * np.pi * fit_phase),
			np.cos(2 * np.pi * fit_phase),
		]
	)
	fit_rv = fit_design @ fit_coefficients[:3]
	offset_design, _ = source_offset_design(source_labels)
	phase_plot_rv = rv - offset_design @ fit_coefficients[3:]
	print(
		f"{args.star}: best-fit RV semiamplitude = "
		f"{semiamplitude:.6g} +/- {semiamplitude_uncertainty:.6g} m/s"
	)
	print(
		f"{args.star}: primary-transit conjunction = "
		f"BJD {conjunction_bjd:.6f} +/- {conjunction_uncertainty:.6f}"
	)
	target_bjd = 2_461_587.5  # 2027-07-01 00:00 UTC
	target_phase, target_phase_uncertainty = calculate_phase_at_date(
		target_bjd,
		conjunction_bjd,
		conjunction_uncertainty,
		period,
		period_uncertainty,
	)
	print(
		f"{args.star}: orbital phase on 2027-07-01 = "
		f"{target_phase:.6f} +/- {target_phase_uncertainty:.6f} cycles"
	)
	keplerian_parameters, keplerian_covariance = fit_keplerian(
		bjd[orbit_fit_mask],
		rv[orbit_fit_mask],
		rv_error[orbit_fit_mask],
		period,
		source_labels[orbit_fit_mask],
	)
	orbit_residuals = (
		rv[orbit_fit_mask]
		- calculate_keplerian_rv(
			bjd[orbit_fit_mask],
			bjd[orbit_fit_mask][0],
			period,
			keplerian_parameters,
			source_labels[orbit_fit_mask],
		)
	) / rv_error[orbit_fit_mask]
	degrees_of_freedom = np.count_nonzero(orbit_fit_mask) - len(keplerian_parameters)
	reduced_chi_squared = np.sum(orbit_residuals**2) / degrees_of_freedom
	eccentricity = keplerian_parameters[2]
	eccentricity_uncertainty = np.sqrt(keplerian_covariance[2, 2])
	omega_degrees = np.degrees(keplerian_parameters[3])
	omega_uncertainty_degrees = np.degrees(
		np.sqrt(keplerian_covariance[3, 3])
	)
	periapsis_bjd = bjd[0] + keplerian_parameters[4] * period
	periapsis_uncertainty = period * np.sqrt(keplerian_covariance[4, 4])
	if eccentricity < eccentricity_uncertainty:
		eccentricity = 0.0
		omega_degrees = np.nan
		omega_uncertainty_degrees = np.nan
		periapsis_bjd = np.nan
		periapsis_uncertainty = np.nan
	print(
		f"{args.star}: eccentricity = "
		f"{eccentricity:.6g} +/- {eccentricity_uncertainty:.6g}"
	)
	print(
		f"{args.star}: longitude of periastron = "
		f"{omega_degrees:.6g} +/- {omega_uncertainty_degrees:.6g} degrees"
	)
	print(
		f"{args.star}: time of periapsis passage = "
		f"BJD {periapsis_bjd:.6f} +/- {periapsis_uncertainty:.6f}"
	)
	print(f"{args.star}: orbit-fit reduced chi2 = {reduced_chi_squared:.6g}")

	figure, axes = plt.subplots(3, 1, sharex=False, figsize=(8, 10))
	for source, color in SOURCE_COLORS.items():
		source_mask = source_labels == source
		if np.any(source_mask):
			axes[0].errorbar(
				bjd[source_mask] - 2_450_000,
				rv[source_mask],
				yerr=rv_error[source_mask],
				fmt="o",
				capsize=2,
				color=color,
				label=source,
			)
	axes[0].set_xlabel("BJD - 2450000")
	axes[0].set_ylabel("Radial velocity (m/s)")
	axes[0].set_title(f"{args.star} ({len(bjd)} measurements)")
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

	axes[1].plot(periods, power)
	axes[1].axvline(period, color="tab:red", linestyle="--")
	axes[1].set_xscale("log")
	axes[1].set_xlabel("Period (days)")
	axes[1].set_ylabel("Lomb-Scargle power")
	axes[1].set_title(
		f"Best-fit period = {period:.6g} +/- {period_uncertainty:.6g} days"
	)
	axes[1].grid(alpha=0.3)

	for source, color in SOURCE_COLORS.items():
		source_mask = source_labels == source
		included_mask = source_mask & orbit_fit_mask
		excluded_mask = source_mask & ~orbit_fit_mask
		if np.any(included_mask):
			axes[2].errorbar(
				phase[included_mask],
				phase_plot_rv[included_mask],
				yerr=rv_error[included_mask],
				fmt="o",
				capsize=2,
				color=color,
				label=source,
			)
		if np.any(excluded_mask):
			axes[2].errorbar(
				phase[excluded_mask],
				phase_plot_rv[excluded_mask],
				yerr=rv_error[excluded_mask],
				fmt="o",
				capsize=2,
				color=color,
				alpha=0.25,
				label=f"{source} (excluded)",
			)
	axes[2].plot(
		fit_phase,
		fit_rv,
		color="tab:red",
		label=(
			f"Sinusoidal fit (K = {semiamplitude:.4g} +/- "
			f"{semiamplitude_uncertainty:.3g} m/s)"
		),
	)
	axes[2].set_xlabel("Orbital phase")
	axes[2].set_ylabel("Radial velocity (m/s)")
	axes[2].set_title(
		f"RV folded on {period:.6g} +/- {period_uncertainty:.6g}-day period; "
		f"K = {semiamplitude:.6g} +/- {semiamplitude_uncertainty:.6g} m/s"
	)
	conjunction_phase = ((conjunction_bjd - bjd[0]) / period) % 1.0
	axes[2].axvline(
		conjunction_phase,
		color="tab:green",
		linestyle="--",
		label="Best-fit conjunction",
	)
	axes[2].set_xlim(0, 1)
	axes[2].grid(alpha=0.3)
	axes[2].legend()

	figure.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
