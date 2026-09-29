#!/usr/bin/env python3
"""Search known hot-Jupiter hosts' RVs for outer planets missing from the Exoplanet Archive.

Hot Jupiters (1.2 d <= P <= 8 d, and M > 0.3 MJup or R > 0.7 RJup) come from a
cached NASA Exoplanet Archive ``pscomppars`` table. Every host with at least
``MIN_POINTS`` local RVs spanning ``MIN_BASELINE_DAYS`` is run through
plot_rvs.fit_system, and its outer-signal flag is compared with the Archive's other
planets for that host. Results are appended per host, so an interrupted run resumes.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import plot_rvs
import rv_io


ARCHIVE_TABLE = rv_io.DATABASE_ROOT / "exoplanet_archive_pscomppars_20260929.csv"
OUTPUT = plot_rvs.PLOTS_DIRECTORY / "hot_jupiter_outer_survey.csv"
CANDIDATE_PLOTS = plot_rvs.PLOTS_DIRECTORY / "survey_candidates"
CANDIDATE_PLOT_MIN_SIGNIFICANCE = 4.0  # nightly-binned significance for plotting a candidate
MIN_POINTS = 20
MIN_BASELINE_DAYS = 365.0
INNER_PERIOD_TOLERANCE = 0.01  # the fitted inner period must match the Archive's
KNOWN_PERIOD_TOLERANCE = 0.25  # a detection within 25% of a listed planet's period is "known"
FIELDS = [
	"host", "hot_jupiter", "archive_period_days", "n_points", "baseline_days", "sources",
	"fitted_inner_period_days", "seeded_with_archive_period", "inner_matches_archive", "flag", "outer_model",
	"outer_period_days", "outer_period_uncertainty_days", "outer_semiamplitude_m_per_s",
	"outer_semiamplitude_uncertainty_m_per_s", "outer_eccentricity", "orbit_coverage",
	"msini_mjup_per_mstar_msun_2_3", "candidate_period_days", "candidate_semiamplitude_m_per_s",
	"candidate_fitted_significance", "rejection_reasons", "archive_other_planets", "status",
]


def _float(value: str) -> float:
	return float(value) if value not in ("", None) else float("nan")


def hot_jupiter_hosts() -> dict[str, dict]:
	"""Archive hot Jupiters by host, with the host's other listed planets."""
	rows = list(csv.DictReader(ARCHIVE_TABLE.open()))
	by_host: dict[str, list[dict]] = {}
	for row in rows:
		by_host.setdefault(row["hostname"], []).append(row)
	hosts = {}
	for host, planets in by_host.items():
		hot = [
			p for p in planets
			if plot_rvs.PERIOD_RANGE[0] <= _float(p["pl_orbper"]) <= plot_rvs.PERIOD_RANGE[1]
			and (_float(p["pl_bmassj"]) >= 0.3 or _float(p["pl_radj"]) >= 0.7)
		]
		if hot:
			hosts[host] = {"hot": hot[0], "others": [p for p in planets if p is not hot[0]]}
	return hosts


def classify(result: dict, others: list[dict]) -> str:
	"""Compare the pipeline's outer signal with the Archive's other planets."""
	if not result["inner_matches_archive"]:
		return "inner period mismatch (not classified)"
	flag = result["flag"]
	if flag in ("none", "possible_activity"):
		return "no outer signal"
	listed = [_float(p["pl_orbper"]) for p in others if _float(p["pl_orbper"]) > 10]
	kind = "planet" if flag == "outer_planet" else "trend" if flag == "trend" else "candidate"
	if flag.startswith("trend"):
		# A trend is explained by any listed companion too long to be resolved.
		if any(known > 0.67 * result["baseline_days"] for known in listed):
			return "matches listed planet"
		return f"new {kind} (not in Archive)"
	period = result["outer_period_days"] or result["candidate_period_days"]
	partial = result["orbit_coverage"] == "partial_orbit"
	for known in listed:
		if abs(period - known) / known < KNOWN_PERIOD_TOLERANCE or (partial and known > 0.67 * result["baseline_days"]):
			return "matches listed planet"
	return f"new {kind} (not in Archive)"


def survey_host(host: str, info: dict) -> dict | None:
	datasets = rv_io.load_datasets(host, plot_rvs.OBSERVED_SOURCES)
	if not datasets:
		return None
	bjd, rv, rv_error, labels = rv_io.combine_rv_data(datasets)
	if len(bjd) < MIN_POINTS or np.ptp(bjd) < MIN_BASELINE_DAYS:
		return None
	archive_period = _float(info["hot"]["pl_orbper"])
	parameters = plot_rvs.fit_system(bjd, rv, rv_error, labels).parameters
	seeded = abs(parameters["period_days"] - archive_period) / archive_period >= INNER_PERIOD_TOLERANCE
	if seeded:  # periodogram alias: refit starting from the known (transit) period
		parameters = plot_rvs.fit_system(bjd, rv, rv_error, labels, initial_period=archive_period).parameters
	outer = parameters["outer_signal"]
	fit = outer.get("fit") or {}
	peak = outer.get("long_period_peak") or {}
	result = {
		"host": host,
		"hot_jupiter": info["hot"]["pl_name"],
		"archive_period_days": archive_period,
		"n_points": len(bjd),
		"baseline_days": float(np.ptp(bjd)),
		"sources": " ".join(sorted({label.split()[0].split("-")[0] for label in labels})),
		"fitted_inner_period_days": parameters["period_days"],
		"seeded_with_archive_period": seeded,
		"inner_matches_archive": abs(parameters["period_days"] - archive_period) / archive_period < INNER_PERIOD_TOLERANCE,
		"flag": outer["flag"],
		"outer_model": outer.get("model"),
		"outer_period_days": fit.get("period_days"),
		"outer_period_uncertainty_days": fit.get("period_uncertainty_days"),
		"outer_semiamplitude_m_per_s": fit.get("semiamplitude_m_per_s"),
		"outer_semiamplitude_uncertainty_m_per_s": fit.get("semiamplitude_uncertainty_m_per_s"),
		"outer_eccentricity": fit.get("eccentricity"),
		"orbit_coverage": fit.get("orbit_coverage"),
		"msini_mjup_per_mstar_msun_2_3": fit.get("msini_mjup_per_mstar_msun_2_3"),
		"candidate_period_days": peak.get("period_days") if outer["flag"] in ("outer_candidate", "trend_candidate") else None,
		"candidate_semiamplitude_m_per_s": peak.get("semiamplitude_m_per_s") if outer["flag"] in ("outer_candidate", "trend_candidate") else None,
		"candidate_fitted_significance": (outer.get("vetting") or {}).get("fitted_significance"),
		"rejection_reasons": "; ".join((outer.get("vetting") or {}).get("rejection_reasons", [])),
		"archive_other_planets": "; ".join(f"{p['pl_name']} ({_float(p['pl_orbper']):.4g} d)" for p in info["others"]),
	}
	result["status"] = classify(result, info["others"])
	return result


def plot_candidates(output: Path, hosts: dict[str, dict], minimum: float) -> None:
	"""Plot each unlisted candidate with its outer term forced into the fit."""
	import matplotlib.pyplot as plt

	CANDIDATE_PLOTS.mkdir(exist_ok=True)
	for row in csv.DictReader(output.open()):
		significance = _float(row["candidate_fitted_significance"])
		if not row["status"].startswith("new") or not significance >= minimum:
			continue
		host = row["host"]
		bjd, rv, rv_error, labels = rv_io.combine_rv_data(rv_io.load_datasets(host, plot_rvs.OBSERVED_SOURCES))
		model = "trend" if row["flag"].startswith("trend") else "sinusoid"
		seed = _float(hosts[host]["hot"]["pl_orbper"]) if row["seeded_with_archive_period"] == "True" else None
		fit = plot_rvs.fit_system(bjd, rv, rv_error, labels, outer_model=model, initial_period=seed)
		figure = plot_rvs.plot_fit(host, bjd, rv, rv_error, labels, fit)
		path = CANDIDATE_PLOTS / f"{host}_candidate_fit_plot.png"
		figure.savefig(path, dpi=110)
		plt.close(figure)
		print(f"{host}: {row['flag']} ({significance:.1f} sigma nightly) -> {path}", flush=True)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--output", type=Path, default=OUTPUT)
	parser.add_argument(
		"--plot-candidates", action="store_true",
		help=f"Only plot unlisted candidates (forced outer term) into {CANDIDATE_PLOTS}.",
	)
	args = parser.parse_args()
	hosts = hot_jupiter_hosts()
	if args.plot_candidates:
		plot_candidates(args.output, hosts, CANDIDATE_PLOT_MIN_SIGNIFICANCE)
		return
	rv_io.simbad_records(hosts)  # one batched query for any uncached host names
	done = {row["host"] for row in csv.DictReader(args.output.open())} if args.output.is_file() else set()
	skipped_path = args.output.with_name(args.output.stem + "_skipped.txt")
	skipped = set(skipped_path.read_text().split("\n")) if skipped_path.is_file() else set()
	new_file = not args.output.is_file()
	with args.output.open("a", newline="") as output, skipped_path.open("a") as skip_log:
		writer = csv.DictWriter(output, fieldnames=FIELDS)
		if new_file:
			writer.writeheader()
		for index, (host, info) in enumerate(sorted(hosts.items()), 1):
			if host in done or host in skipped:
				continue
			try:
				result = survey_host(host, info)
			except Exception as error:  # keep surveying; record the failure
				print(f"[{index}/{len(hosts)}] {host}: failed ({error})", flush=True)
				continue
			if result is None:
				skip_log.write(host + "\n")
				skip_log.flush()
				continue
			writer.writerow(result)
			output.flush()
			print(f"[{index}/{len(hosts)}] {host}: {result['n_points']} RVs, {result['flag']} -> {result['status']}", flush=True)


if __name__ == "__main__":
	main()
