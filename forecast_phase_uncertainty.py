#!/usr/bin/env python3
"""Forecast how a few new HIRES RVs in 2027A change the orbital phase uncertainty.

For each target in plots/rv_target_summary.csv, synthetic HIRES points are drawn
on random observable 2027A nights from the adopted orbit fit (plus HIRES
errors and the star's jitter). They are appended to the real data, and the full
plot_rvs fit is rerun. Each realization reports the pipeline's uncertainties
and an analytic forecast that weights the new points by their actual noise.

Observability is analytic: Maunakea, altitude > 30 deg while the Sun is below
-12 deg, for at least one hour. Mean solar time is used, so windows are good to
about 15 minutes.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import plot_rvs
import rv_io


LATITUDE = np.radians(19.8264)
LONGITUDE_HOURS = -155.4747 / 15
SEMESTER_JD = (2_461_437.5, 2_461_617.5)  # 2027-02-01 to 2027-07-31 (UT dates)
MIN_ALTITUDE = np.radians(30)
SUN_ALTITUDE = np.radians(-12)
MIN_WINDOW_HOURS = 1.0
HIRES_UPGRADE_BJD = 2_453_236.5  # post-2004 CCD upgrade
SIDEREAL_RATE = 1.002_737_909


def _half_arc_hours(altitude: float, declination: np.ndarray) -> np.ndarray:
	"""Hour angle (h) at which an object at ``declination`` crosses ``altitude``."""
	cos_h = (np.sin(altitude) - np.sin(LATITUDE) * np.sin(declination)) / (
		np.cos(LATITUDE) * np.cos(declination)
	)
	return np.degrees(np.arccos(np.clip(cos_h, -1, 1))) / 15


def observable_windows(ra_deg: float, dec_deg: float) -> list[tuple[float, float]]:
	"""Return (start JD, end JD) of the target's usable window on each 2027A night."""
	dates = np.arange(*SEMESTER_JD)  # 0h UT of each date; the night starts that evening
	midnight = dates + (24 - LONGITUDE_HOURS) / 24 % 1  # local mean midnight, UT JD
	days = midnight - 2_451_545.0
	sun_longitude = np.radians(280.46 + 0.985_647_4 * days)  # mean longitude, good to ~2 deg
	sun_dec = np.arcsin(np.sin(np.radians(23.44)) * np.sin(sun_longitude))
	night_half = 12 - _half_arc_hours(SUN_ALTITUDE, sun_dec)  # hours either side of midnight
	gmst = (18.697_374_558 + 24.065_709_824_419_08 * days) % 24
	hour_angle = (gmst + LONGITUDE_HOURS - ra_deg / 15 + 12) % 24 - 12  # at midnight
	transit = -hour_angle / SIDEREAL_RATE  # hours from midnight
	up_half = _half_arc_hours(MIN_ALTITUDE, np.radians(dec_deg)) / SIDEREAL_RATE
	windows = []
	for mid, half, tr in zip(midnight, night_half, transit):
		start, end = max(-half, tr - up_half), min(half, tr + up_half)
		if end - start >= MIN_WINDOW_HOURS:
			windows.append((mid + start / 24, mid + end / 24))
	return windows


def saved_model(time, parameters, label):
	"""Evaluate the saved adopted orbit, outer signal, and the label's zero point."""
	time = np.atleast_1d(time)
	orbit = plot_rvs.saved_orbit(parameters)
	offset = parameters["source_offsets_m_per_s"].get(label, 0.0)
	model = plot_rvs.orbit_rv(time, orbit, np.empty((len(time), 0))) + offset
	outer, outer_parameters = plot_rvs.saved_outer_signal(parameters)
	if outer is not None:
		model = model + outer.evaluate(time, outer_parameters)
	return model


def analytic_phase_hours(parameters, new_times, new_sigma, free_offset):
	"""Combine the current phase uncertainty with the phase information from new points.

	With the orbit shape fixed, N points of noise sigma_i give
	1/var(phi) = sum_i (dRV_i/dphi / sigma_i)^2, where dRV/dphi = -P dRV/dTc is
	evaluated numerically for the adopted (possibly eccentric) orbit. A free zero
	point for the new points removes the weighted-mean part of the slopes.
	"""
	period = parameters["period_days"]
	step = 1e-4 * period
	orbit = plot_rvs.saved_orbit(parameters)
	no_offsets = np.empty((len(new_times), 0))
	later, earlier = orbit.copy(), orbit.copy()
	later[plot_rvs.ORBIT_CONJUNCTION] += step
	earlier[plot_rvs.ORBIT_CONJUNCTION] -= step
	slope = period * (
		plot_rvs.orbit_rv(new_times, later, no_offsets) - plot_rvs.orbit_rv(new_times, earlier, no_offsets)
	) / (2 * step)
	weights = np.broadcast_to(new_sigma, new_times.shape) ** -2.0
	information = np.sum(weights * slope**2)
	if free_offset:
		information -= np.sum(weights * slope) ** 2 / np.sum(weights)
	current = parameters[f"phase_uncertainty_on_{plot_rvs.TARGET_DATE}"]
	return period * 24 / np.sqrt(current**-2 + information)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--realizations", type=int, default=5, help="date sets per N")
	parser.add_argument("--counts", type=int, nargs="+", default=[3, 4, 5])
	parser.add_argument("--seed", type=int, default=2027)
	parser.add_argument("--output", type=Path, default=Path("plots/forecast_2027A_hires.csv"))
	parser.add_argument("--targets", nargs="*", help="default: rows of rv_target_summary.csv")
	args = parser.parse_args()

	rng = np.random.default_rng(args.seed)
	targets = args.targets or [
		row["target_name"] for row in csv.DictReader(open("plots/rv_target_summary.csv"))
	]
	rows = []
	for target in targets:
		parameters = json.loads(Path(f"plots/{target}_rv_fit_parameters.json").read_text())
		bjd, rv, rv_error, labels = rv_io.combine_rv_data(
			rv_io.load_datasets(target, plot_rvs.OBSERVED_SOURCES)
		)
		record = rv_io.simbad_record(target)
		windows = observable_windows(record["ra_deg"], record["dec_deg"])

		# HIRES noise model: the star's own modern HIRES errors and jitter when
		# available, otherwise the pooled errors below and the star's jitter.
		hires_labels = [label for label in ("Teklu", "CLS HIRES-j") if np.any(labels == label)]
		hires = np.isin(labels, hires_labels) & (bjd > HIRES_UPGRADE_BJD)
		noise_mask = hires if hires.any() else np.ones(len(bjd), dtype=bool)
		jitter = plot_rvs.short_term_jitter(
			bjd[noise_mask],
			np.array([
				rv[i] - saved_model(bjd[i], parameters, labels[i])[0]
				for i in np.flatnonzero(noise_mask)
			]),
			rv_error[noise_mask],
		)
		error_pool = rv_error[hires] if hires.any() else POOLED_HIRES_ERRORS

		cases = [("free offset", "HIRES 2027")]
		if hires_labels:
			cases.insert(0, ("shared zero point", hires_labels[0]))
		base_hours = parameters[f"phase_uncertainty_hours_on_{plot_rvs.TARGET_DATE}"]
		print(
			f"{target}: {len(windows)} observable 2027A nights, jitter {jitter:.1f} m/s,"
			f" HIRES error ~{np.median(error_pool):.2f} m/s, now {base_hours:.2f} h"
		)
		if len(windows) < max(args.counts):
			print(f"  only {len(windows)} observable nights; skipping larger N")
		for count in args.counts:
			if count > len(windows):
				continue
			for realization in range(args.realizations):
				nights = rng.choice(len(windows), size=count, replace=False)
				times = np.sort([rng.uniform(*windows[night]) for night in nights])
				errors = rng.choice(error_pool, size=count)
				sigma = np.hypot(errors, jitter)
				noise = rng.normal(0.0, sigma)
				for case, label in cases:
					new_rv = saved_model(times, parameters, label) + noise
					fit = plot_rvs.fit_system(
						np.concatenate([bjd, times]),
						np.concatenate([rv, new_rv]),
						np.concatenate([rv_error, errors]),
						np.concatenate([labels, np.full(count, label, dtype=object)]),
					).parameters
					rows.append({
						"target": target,
						"case": case,
						"n_new": count,
						"realization": realization,
						"dates_bjd": " ".join(f"{time:.3f}" for time in times),
						"orbit_model": fit["orbit_model"],
						"eccentricity": fit["eccentricity"],
						"period_days": fit["period_days"],
						"period_uncertainty_days": fit["period_uncertainty_days"],
						"conjunction_uncertainty_days": fit["conjunction_uncertainty_days"],
						"semiamplitude_uncertainty_m_per_s": fit["semiamplitude_uncertainty_m_per_s"],
						"phase_uncertainty_hours": fit[f"phase_uncertainty_hours_on_{plot_rvs.TARGET_DATE}"],
						"analytic_phase_uncertainty_hours": analytic_phase_hours(
							parameters, times, sigma, case == "free offset"
						),
						"baseline_phase_uncertainty_hours": base_hours,
						"baseline_period_uncertainty_days": parameters["period_uncertainty_days"],
						"baseline_conjunction_uncertainty_days": parameters["conjunction_uncertainty_days"],
						"baseline_semiamplitude_uncertainty_m_per_s":
							parameters["semiamplitude_uncertainty_m_per_s"],
						"pipeline_new_point_jitter_m_per_s": fit["jitter_m_per_s"][label],
						"jitter_m_per_s": jitter,
					})
			done = [row for row in rows if row["target"] == target and row["n_new"] == count]
			for case, _ in cases:
				hours = [row["phase_uncertainty_hours"] for row in done if row["case"] == case]
				analytic = [row["analytic_phase_uncertainty_hours"] for row in done if row["case"] == case]
				print(
					f"  N={count} {case:17s} pipeline {np.median(hours):6.2f} h"
					f" [{min(hours):.2f}-{max(hours):.2f}]  analytic {np.median(analytic):6.2f} h"
				)

	with args.output.open("w", newline="") as output:
		writer = csv.DictWriter(output, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)
	print(f"Wrote {len(rows)} realizations to {args.output}")


def _pooled_hires_errors() -> np.ndarray:
	"""Post-upgrade Teklu HIRES errors across all targets with HIRES data."""
	errors = []
	with rv_io.TEKLU_DATABASE.open(encoding="ascii") as data_file:
		for line in data_file:
			row = rv_io._first_three_floats([line[82:95], line[132:148], line[149:156]])
			if row is not None and row[0] > HIRES_UPGRADE_BJD:
				errors.append(row[2])
	errors = np.asarray(errors)
	return errors[errors < np.percentile(errors, 95)]


POOLED_HIRES_ERRORS = _pooled_hires_errors()


if __name__ == "__main__":
	main()
