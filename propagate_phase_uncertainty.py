#!/usr/bin/env python3
"""Propagate the saved fits' conjunction-time uncertainties to later dates.

For each target in plots/rv_target_summary.csv, sigma_t on each date follows from
the first-order propagation the fit uses on plot_rvs.TARGET_DATE,

    sigma_t^2 = sigma_Tc^2 + n^2 sigma_P^2 + 2 n cov(Tc, P),   n = (t - Tc) / P.

The fits save sigma_Tc, sigma_P, and sigma_t on TARGET_DATE but not cov(Tc, P),
so the covariance is recovered from the saved sigma_t. No new data are assumed.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from astropy.time import Time

import plot_rvs

DEFAULT_DATES = ["2027-08-01", "2027-11-01", "2028-01-31"]  # 2027B start, middle, end


def sigma_t_hours(parameters: dict, jds: np.ndarray) -> np.ndarray:
	"""Conjunction-time uncertainty [h] of a saved fit at the given (UTC) JDs."""
	period = parameters["period_days"]
	period_error = parameters["period_uncertainty_days"]
	conjunction = parameters["conjunction_bjd"]
	conjunction_error = parameters["conjunction_uncertainty_days"]
	saved = parameters[f"phase_uncertainty_hours_on_{plot_rvs.TARGET_DATE}"] / 24
	saved_cycles = (plot_rvs.TARGET_BJD - conjunction) / period
	covariance = (saved**2 - conjunction_error**2 - saved_cycles**2 * period_error**2) / (2 * saved_cycles)
	cycles = (np.asarray(jds) - conjunction) / period
	return 24 * np.sqrt(conjunction_error**2 + cycles**2 * period_error**2 + 2 * cycles * covariance)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES, help="UTC dates (default: 2027B)")
	parser.add_argument(
		"--output", type=Path, default=plot_rvs.PLOTS_DIRECTORY / "phase_uncertainty_2027B.csv"
	)
	args = parser.parse_args()

	jds = Time([f"{date}T00:00:00" for date in args.dates], scale="utc").jd
	middle = len(args.dates) // 2
	rows = []
	for row in csv.DictReader(open(plot_rvs.PLOTS_DIRECTORY / "rv_target_summary.csv")):
		target = row["target_name"]
		parameters = json.loads((plot_rvs.PLOTS_DIRECTORY / f"{target}_rv_fit_parameters.json").read_text())
		hours = sigma_t_hours(parameters, jds)
		rows.append({
			"target": target,
			"period_days": f"{parameters['period_days']:.7f}",
			"orbit_model": parameters["orbit_model"],
			"eccentricity": f"{parameters['eccentricity']:.4f}",
			**{f"sigma_t_h_{date}": f"{value:.3f}" for date, value in zip(args.dates, hours)},
			f"sigma_phase_cycles_{args.dates[middle]}": f"{hours[middle] / 24 / parameters['period_days']:.5f}",
		})
	rows.sort(key=lambda row: float(row[f"sigma_t_h_{args.dates[middle]}"]))
	with args.output.open("w", newline="") as output:
		writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
		writer.writeheader()
		writer.writerows(rows)
	print(f"Wrote {len(rows)} targets to {args.output}")


if __name__ == "__main__":
	main()
