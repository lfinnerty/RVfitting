#!/usr/bin/env python3
"""Write an hrccs_planner (KPIC/ObsTools) target list from the saved RV fits.

One row per target in plots/rv_target_summary.csv, with the adopted orbit
(transit conjunction, period, e and the star's omega; ObsTools uses the same
conventions), the conjunction and period uncertainties (`T err (d)`,
`P err (d)`), and two sigma_t scenarios on 2027-07-01 for
`hrccs-plan phase-sensitivity --sigma-column`: the current fit, and the median
after 3 new HIRES RVs from plots/forecast_2027A_hires.csv.

Kp max = 2 pi a / P with a from Kepler's third law for a 1 Msun star, so it is
good to ~10% (a ~ M^(1/3)); S/N ratios from phase-sensitivity do not depend on
it, but its --delta-v-min cut does.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import plot_rvs
import rv_io

AU_KM = 1.495_978_707e8
DAYS_PER_YEAR = 365.25
COLUMNS = [
	"Name", "RA", "Dec", "Period (day)", "T transit/inf conj (JD)", "Kp max (km/s)", "Eccentric?", "e", "omega",
	"T err (d)", "P err (d)", "sigma_t now (h)", "sigma_t +3 RVs (h)",
]


def kp_max_km_s(period_days: float, stellar_mass_msun: float = 1.0) -> float:
	"""Total orbital velocity 2 pi a / P for a star of the given mass."""
	a_au = (stellar_mass_msun * (period_days / DAYS_PER_YEAR) ** 2) ** (1 / 3)
	return 2 * np.pi * a_au * AU_KM / (period_days * 86_400)


def sigma_after_three(forecast: list[dict], target: str) -> float:
	"""Median 2027-07-01 sigma_t [h] with 3 new HIRES points (shared zero point if available)."""
	cases = {r["case"] for r in forecast if r["target"] == target}
	case = "shared zero point" if "shared zero point" in cases else "free offset"
	values = [float(r["phase_uncertainty_hours"]) for r in forecast
		if r["target"] == target and r["case"] == case and r["n_new"] == "3"]
	return float(np.median(values)) if values else float("nan")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--output", type=Path, default=plot_rvs.PLOTS_DIRECTORY / "hrccs_targetlist.csv")
	args = parser.parse_args()

	names = [row["target_name"] for row in csv.DictReader(open(plot_rvs.PLOTS_DIRECTORY / "rv_target_summary.csv"))]
	forecast = list(csv.DictReader(open(plot_rvs.PLOTS_DIRECTORY / "forecast_2027A_hires.csv")))
	records = rv_io.simbad_records(names)
	with args.output.open("w", newline="") as output:
		writer = csv.writer(output, lineterminator="\n")
		writer.writerow(COLUMNS)
		for name in names:
			fit = json.loads((plot_rvs.PLOTS_DIRECTORY / f"{name}_rv_fit_parameters.json").read_text())
			eccentric = fit["orbit_model"] == "keplerian"
			writer.writerow([
				name, f"{records[name]['ra_deg']:.6f}", f"{records[name]['dec_deg']:.6f}",
				f"{fit['period_days']:.8f}", f"{fit['conjunction_bjd']:.6f}", f"{kp_max_km_s(fit['period_days']):.1f}",
				"Y" if eccentric else "N",
				f"{fit['eccentricity'] if eccentric else 0.0:.4f}", f"{fit['omega_degrees'] if eccentric else 90.0:.2f}",
				f"{fit['conjunction_uncertainty_days']:.6f}", f"{fit['period_uncertainty_days']:.3e}",
				f"{fit['phase_uncertainty_hours_on_2027-07-01']:.3f}", f"{sigma_after_three(forecast, name):.3f}",
			])
	print(f"Wrote {len(names)} targets to {args.output}")


if __name__ == "__main__":
	main()
