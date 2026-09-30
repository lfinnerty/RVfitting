#!/usr/bin/env python3
"""Fit every RV-discovered hot Jupiter and tabulate its 2027B phase uncertainty.

Planets come from the cached NASA Exoplanet Archive ``pscomppars`` table:
discovered by radial velocity, P < 8 d, and M >= 0.3 MJup. Some Archive rows give
pl_bmassj in Earth masses, so a planet whose listed K implies M sin i below a
tenth of its listed mass is dropped, as is any host fainter than MAX_KMAG in 2MASS K
(cached per planet in KMAG_TABLE) and any planet in EXCLUDED.

A host that already has a fit in plots/ (matched by SIMBAD identifiers) keeps it.
Otherwise the host is fitted with plot_rvs.fit_system and saved like a plot_rvs
fit; if the fitted period misses the Archive's by more than 1%, it is refitted
starting from the Archive period. Each new fit is recorded in a cache keyed on a
hash of its RVs and seed, so an interrupted run resumes and a host is refitted
only when its data change. sigma_t on each date is propagated from the saved fit
(propagate_phase_uncertainty.sigma_t_hours).

Transit status comes from a cached exoplanet.eu table (the planet within 30" nearest in
period, if within 10%): a planet transits if its detection type includes "Primary Transit" or it
has a transit epoch together with a measured radius and inclination (an epoch alone
can be an RV conjunction time). A second table keeps planets north of ``--min-dec``
that are not known to transit (status "no" or "unknown").
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

import plot_rvs
import propagate_phase_uncertainty
import rv_io

ARCHIVE_TABLE = rv_io.ARCHIVE_TABLE
KMAG_TABLE = rv_io.DATABASE_ROOT / "twomass_kmag_rv_hot_jupiters_20260930.csv"
EU_TABLE = rv_io.DATABASE_ROOT / "exoplanet_eu_shortperiod_20260930.csv"
OUTPUT = plot_rvs.PLOTS_DIRECTORY / "phase_uncertainty_2027B_rv_hot_jupiters.csv"
FILTERED_OUTPUT = plot_rvs.PLOTS_DIRECTORY / "phase_uncertainty_2027B_rv_hot_jupiters_nontransiting.csv"
MIN_DEC_DEG = -24.0
EU_MATCH_ARCSEC = 30.0
EU_PERIOD_TOLERANCE = 0.1
CACHE = plot_rvs.PLOTS_DIRECTORY / "rv_hot_jupiter_fits.json"
MAX_PERIOD_DAYS = 8.0
MIN_MASS_MJUP = 0.3
MAX_KMAG = 10.0
MIN_POINTS = 8  # enough for a circular orbit, offset and jitter
# Planets left out on purpose: RVs dominated by stellar activity that a Keplerian fit
# without an activity model cannot separate from the planet.
EXCLUDED = {
	"HS Psc b": "young active star; published fit needs a GP activity model (Tran+2024)",
	"V830 Tau b": "spot RVs of ~1.5 km/s; planet disputed (Damasso+2020)",
}
PERIOD_TOLERANCE = 0.01  # fractional match to the Archive period
UNCONSTRAINED_PHASE_CYCLES = 0.2  # beyond this, first-order propagation of sigma_t breaks down
K_JUPITER_1YR_M_PER_S = 28.4329


def _float(value: str) -> float:
	return float(value) if value not in ("", None) else float("nan")


def host_kmags() -> dict[str, float]:
	"""2MASS K of each planet's host, by planet name (NaN if unmatched)."""
	if not KMAG_TABLE.is_file():
		return {}
	return {row["pl_name"]: _float(row["kmag"]) for row in csv.DictReader(KMAG_TABLE.open())}


def rv_hot_jupiters(max_kmag: float = MAX_KMAG) -> list[dict]:
	"""Archive planets discovered by RV with P < MAX_PERIOD_DAYS and M >= MIN_MASS_MJUP
	(consistent with K), around hosts with 2MASS K <= ``max_kmag`` (kept if unmatched)."""
	kmags = host_kmags()
	planets = []
	for row in csv.DictReader(ARCHIVE_TABLE.open()):
		if not (
			row["discoverymethod"] == "Radial Velocity"
			and _float(row["pl_orbper"]) < MAX_PERIOD_DAYS and _float(row["pl_bmassj"]) >= MIN_MASS_MJUP
			and mass_consistent_with_k(row)
		):
			continue
		row["kmag"] = kmags.get(row["pl_name"], float("nan"))
		if not row["kmag"] > max_kmag and row["pl_name"] not in EXCLUDED:
			planets.append(row)
	return sorted(planets, key=lambda row: _float(row["pl_orbper"]))


def mass_consistent_with_k(planet: dict) -> bool:
	"""False when the listed K implies M sin i (1 Msun, circular) below 10% of the listed mass."""
	implied = (
		_float(planet["pl_rvamp"]) / K_JUPITER_1YR_M_PER_S * (_float(planet["pl_orbper"]) / 365.25) ** (1 / 3)
	)
	return not implied < 0.1 * _float(planet["pl_bmassj"])


def transit_status(planet: dict, eu_rows: list[dict]) -> str:
	"""'yes', 'no', or 'unknown' (no exoplanet.eu planet at this position and period).

	Among the exoplanet.eu planets within EU_MATCH_ARCSEC, the one nearest in period is
	used if it is within EU_PERIOD_TOLERANCE (catalogue periods can disagree by a few %).
	"""
	ra, dec, period = _float(planet["ra"]), _float(planet["dec"]), _float(planet["pl_orbper"])
	candidates = []
	for row in eu_rows:
		d_ra = (_float(row["ra"]) - ra + 180) % 360 - 180
		if 3600 * np.hypot(d_ra * np.cos(np.radians(dec)), _float(row["dec"]) - dec) < EU_MATCH_ARCSEC:
			candidates.append((abs(_float(row["period"]) / period - 1), row))
	if not candidates:
		return "unknown"
	mismatch, row = min(candidates, key=lambda candidate: candidate[0])
	if not mismatch < EU_PERIOD_TOLERANCE:
		return "unknown"
	measured = row["tzero_tr"] and row["radius_error_min"] and row["inclination"]
	return "yes" if "Primary Transit" in row["detection_type"] or measured else "no"


def existing_fits() -> dict[str, dict]:
	"""Saved real-data fits in plots/, by input star name."""
	fits = {}
	for path in plot_rvs.PLOTS_DIRECTORY.glob("*_rv_fit_parameters.json"):
		parameters = json.loads(path.read_text())
		if not parameters.get("includes_synthetics") and parameters.get("source") == "all":
			fits[parameters["input_star"]] = parameters
	return fits


def input_hash(bjd, rv, rv_error, labels, seed) -> str:
	digest = hashlib.sha256()
	for array in (bjd, rv, rv_error):
		digest.update(np.ascontiguousarray(array, dtype=float).tobytes())
	digest.update("\0".join(map(str, labels)).encode())
	digest.update(repr(seed).encode())
	return digest.hexdigest()


def fit_host(host: str, archive_period: float, cache: dict) -> tuple[str, str]:
	"""Fit ``host`` (or reuse its cached fit); return (status, note)."""
	datasets = rv_io.load_datasets(host, plot_rvs.OBSERVED_SOURCES)
	if not datasets:
		return "no local RVs", ""
	bjd, rv, rv_error, labels = rv_io.combine_rv_data(datasets)
	if len(bjd) < MIN_POINTS:
		return f"too few RVs ({len(bjd)})", ""
	key = input_hash(bjd, rv, rv_error, labels, None)
	entry = cache.get(host)
	json_path = plot_rvs.PLOTS_DIRECTORY / f"{host}_rv_fit_parameters.json"
	if entry and entry["input_hash"] == key and json_path.is_file():
		return entry["status"], entry["note"]
	fit = plot_rvs.fit_system(bjd, rv, rv_error, labels)
	note = ""
	if abs(fit.parameters["period_days"] - archive_period) / archive_period > PERIOD_TOLERANCE:
		# Periodogram alias: start from the Archive period instead.
		note = f"seeded with Archive period (unseeded fit found {fit.parameters['period_days']:.5f} d)"
		fit = plot_rvs.fit_system(bjd, rv, rv_error, labels, initial_period=archive_period)
	matched = abs(fit.parameters["period_days"] - archive_period) / archive_period <= PERIOD_TOLERANCE
	status = "new fit" if matched else "new fit, period mismatch"
	plot_rvs.save_fit(host, "all", bjd, rv, rv_error, labels, fit)
	plt.close("all")
	cache[host] = {"input_hash": key, "status": status, "note": note}
	CACHE.write_text(json.dumps(cache, indent=1) + "\n")
	return status, note


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--dates", nargs="+", default=propagate_phase_uncertainty.DEFAULT_DATES)
	parser.add_argument("--output", type=Path, default=OUTPUT)
	parser.add_argument("--filtered-output", type=Path, default=FILTERED_OUTPUT)
	parser.add_argument("--min-dec", type=float, default=MIN_DEC_DEG, help="filtered table: Dec > this [deg]")
	parser.add_argument("--max-kmag", type=float, default=MAX_KMAG, help="drop hosts fainter than this 2MASS K")
	args = parser.parse_args()
	eu_rows = list(csv.DictReader(EU_TABLE.open(newline="")))

	planets = rv_hot_jupiters(args.max_kmag)
	fits = existing_fits()
	rv_io.simbad_records(sorted({p["hostname"] for p in planets} | set(fits)))  # one batched query
	fit_keys = {star: rv_io.identifier_keys(star) for star in fits}
	cache = json.loads(CACHE.read_text()) if CACHE.is_file() else {}
	jds = Time([f"{date}T00:00:00" for date in args.dates], scale="utc").jd
	middle = len(args.dates) // 2
	rows = []
	for planet in planets:
		host, archive_period = planet["hostname"], _float(planet["pl_orbper"])
		keys = rv_io.identifier_keys(host)
		existing = next((star for star, star_keys in fit_keys.items() if keys & star_keys and star not in cache), None)
		if existing:
			target, (status, note) = existing, ("existing fit", "")
		else:
			try:
				status, note = fit_host(host, archive_period, cache)
			except Exception as error:  # one bad host must not stop the run
				status, note = "fit failed", str(error)
			target = host if status.startswith("new fit") else ""
		print(f"{planet['pl_name']}: {status}{f' ({note})' if note else ''}", flush=True)
		row = {
			"planet": planet["pl_name"], "host": host, "fit_target": target, "status": status,
			"dec_deg": f"{_float(planet['dec']):.4f}", "kmag": f"{planet['kmag']:.2f}", "transiting": transit_status(planet, eu_rows),
			"archive_period_days": f"{archive_period:.7f}", "archive_mass_mjup": planet["pl_bmassj"],
			"archive_k_m_per_s": planet["pl_rvamp"],
		}
		if target:
			parameters = json.loads((plot_rvs.PLOTS_DIRECTORY / f"{target}_rv_fit_parameters.json").read_text())
			hours = propagate_phase_uncertainty.sigma_t_hours(parameters, jds)
			row.update({
				"n_rvs": parameters["measurement_count"],
				"fitted_period_days": f"{parameters['period_days']:.7f}",
				"orbit_model": parameters["orbit_model"],
				"eccentricity": f"{parameters['eccentricity']:.4f}",
				"reduced_chi2": f"{parameters['reduced_chi_squared']:.2f}",
				"outer_flag": parameters["outer_signal"]["flag"],
				**{f"sigma_t_h_{date}": f"{value:.3f}" for date, value in zip(args.dates, hours)},
				f"sigma_phase_cycles_{args.dates[middle]}": f"{hours[middle] / 24 / parameters['period_days']:.5f}",
			})
			if hours[middle] / 24 / parameters["period_days"] > UNCONSTRAINED_PHASE_CYCLES:
				note = "; ".join(filter(None, [note, "phase effectively unknown (linear sigma_t not meaningful)"]))
		row["note"] = note
		rows.append(row)
	fields = [
		"planet", "host", "fit_target", "status", "dec_deg", "kmag", "transiting", "archive_period_days", "fitted_period_days", "archive_mass_mjup",
		"archive_k_m_per_s", "n_rvs", "orbit_model", "eccentricity", "reduced_chi2", "outer_flag",
		*[f"sigma_t_h_{date}" for date in args.dates], f"sigma_phase_cycles_{args.dates[middle]}", "note",
	]
	filtered = [row for row in rows if float(row["dec_deg"]) > args.min_dec and row["transiting"] != "yes"]
	for path, table in ((args.output, rows), (args.filtered_output, filtered)):
		with path.open("w", newline="") as output:
			writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
			writer.writeheader()
			writer.writerows(table)
		print(f"Wrote {len(table)} planets to {path}")


if __name__ == "__main__":
	main()
