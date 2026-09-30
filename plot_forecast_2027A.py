#!/usr/bin/env python3
"""Charts of the 2027A HIRES forecast and its 2027B HRCCS S/N impact, plus copies of the fits.

Writes to ``--output-dir``:
  forecast_2027A_sigma_t.png   current sigma_t on 2027-07-01 and the median (range) with
                               3 and 5 new HIRES RVs, from forecast_phase_uncertainty.py
  forecast_2027B_hrccs_snr.png mean S/N relative to a perfect ephemeris for each target's
                               best 2027B windows, now and with 3 new RVs, from
                               `hrccs-plan phase-sensitivity`
  fits/<target>_rv_fit_plot.png the current real-data orbit fit plot of every target
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_rvs

NOW_COLOR, PLUS3_COLOR, PLUS5_COLOR = "#8a8a8a", "#7fb2ee", "#1a4f96"
INK, MUTED, GRID, SURFACE = "#1f1f1f", "#555555", "#e4e4e4", "#fcfcfb"
UNKNOWN_PHASE_CYCLES = 0.2


def _style(axis) -> None:
	axis.set_facecolor(SURFACE)
	for side in ("top", "right", "left"):
		axis.spines[side].set_visible(False)
	axis.spines["bottom"].set_color(GRID)
	axis.tick_params(colors=MUTED, length=0)
	axis.grid(axis="x", color=GRID, linewidth=1)
	axis.set_axisbelow(True)


def _titles(figure, title: str, subtitle: str) -> None:
	"""Left-aligned title and subtitle above the plot, leaving about 1 inch for them."""
	height = figure.get_figheight()
	figure.text(0.02, 1 - 0.25 / height, title, ha="left", va="top", fontsize=14, color=INK)
	figure.text(0.02, 1 - 0.58 / height, subtitle, ha="left", va="top", fontsize=9, color=MUTED)
	figure.tight_layout(rect=(0, 0, 1, 1 - 1.05 / height))


def _subtitle(parameters: dict, new_zero_point: bool) -> str:
	parts = ["eccentric" if parameters["orbit_model"] == "keplerian" else "circular"]
	if parameters["outer_signal"]["flag"] in ("outer_planet", "trend"):
		parts[0] += " + outer planet" if parameters["outer_signal"]["flag"] == "outer_planet" else " + trend"
	if new_zero_point:
		parts.append("new zero point")
	return "; ".join(parts)


def sigma_t_chart(forecast: list[dict], targets: list[tuple[str, str]], path: Path) -> None:
	"""One row per target: now (open circle), +3 (dot) and +5 (diamond) with their ranges."""
	rows = []
	for fit_name, label in targets:
		parameters = json.loads((plot_rvs.PLOTS_DIRECTORY / f"{fit_name}_rv_fit_parameters.json").read_text())
		mine = [r for r in forecast if r["target"] == fit_name]
		case = "shared zero point" if any(r["case"] == "shared zero point" for r in mine) else "free offset"
		values = {
			n: np.array([float(r["phase_uncertainty_hours"]) for r in mine if r["case"] == case and r["n_new"] == n])
			for n in ("3", "5")
		}
		now = parameters[f"phase_uncertainty_hours_on_{plot_rvs.TARGET_DATE}"]
		unknown = parameters[f"phase_uncertainty_on_{plot_rvs.TARGET_DATE}"] > UNKNOWN_PHASE_CYCLES
		rows.append((label, _subtitle(parameters, case == "free offset"), now, values, unknown))
	rows.sort(key=lambda row: row[2])

	figure, axis = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 2.4), facecolor=SURFACE)
	_style(axis)
	for y, (label, subtitle, now, values, unknown) in enumerate(rows):
		for n, offset, color, marker, size in (("3", 0.16, PLUS3_COLOR, "o", 55), ("5", -0.16, PLUS5_COLOR, "D", 45)):
			if not len(values[n]):
				continue
			axis.plot([now, np.median(values[n])], [y + offset] * 2, color=GRID, linewidth=2, zorder=1)
			axis.plot([values[n].min(), values[n].max()], [y + offset] * 2, color=color, linewidth=2, zorder=2)
			axis.scatter(np.median(values[n]), y + offset, s=size, marker=marker, color=color, edgecolor=SURFACE,
				linewidth=1.5, zorder=4, label=f"+{n} HIRES points (median; line = range)" if y == 0 else None)
		axis.scatter(now, y, s=85, facecolor=SURFACE, edgecolor=NOW_COLOR, linewidth=2.2, zorder=3,
			label="Now (current data)" if y == 0 else None)
		axis.text(-0.01, y + 0.1, label + (" ⚠" if unknown else ""), transform=axis.get_yaxis_transform(),
			ha="right", va="center", fontsize=10.5, color=INK)
		axis.text(-0.01, y - 0.25, subtitle, transform=axis.get_yaxis_transform(), ha="right", va="center",
			fontsize=8, color=MUTED)
	axis.set_xscale("log")
	axis.set_xlim(0.04, 600)
	axis.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500])
	axis.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100", "200", "500"])
	axis.set_yticks([])
	axis.set_ylim(-0.7, len(rows) - 0.3)
	axis.set_xlabel(f"Transit-time uncertainty on {plot_rvs.TARGET_DATE} (hours, log scale)", color=MUTED)
	handles, labels = axis.get_legend_handles_labels()
	order = [labels.index(text) for text in sorted(labels, key=lambda text: (not text.startswith("Now"), text))]
	axis.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", frameon=False)
	_titles(
		figure, "How much 3 or 5 HIRES RVs in 2027A tighten the 2027-07-01 transit time",
		"Full pipeline refits of synthetic points on observable 2027A nights; median of 5 date sets, line = range.\n"
		"⚠ phase unknown today: the forecast assumes the new points recover the right cycle count.",
	)
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def snr_chart(sensitivity: list[dict], labels: dict[str, str], path: Path) -> None:
	"""Mean HRCCS S/N (relative to a perfect ephemeris) now and with 3 new RVs, per target."""
	by_target: dict[str, dict] = {}
	for row in sensitivity:
		by_target.setdefault(row["target"], {})[row["scenario"]] = float(row["mean_snr_ratio"])
	rows = sorted(
		((labels.get(t, t), s["sigma_t now (h)"], s["sigma_t +3 RVs (h)"]) for t, s in by_target.items()),
		key=lambda row: row[1], reverse=True,
	)
	figure, axis = plt.subplots(figsize=(9.5, 0.36 * len(rows) + 2.4), facecolor=SURFACE)
	_style(axis)
	for y, (label, now, plus3) in enumerate(rows):
		axis.plot([now, plus3], [y, y], color=GRID, linewidth=2, zorder=1)
		axis.scatter(now, y, s=75, facecolor=SURFACE, edgecolor=NOW_COLOR, linewidth=2.2, zorder=3,
			label="Now (current data)" if y == 0 else None)
		axis.scatter(plus3, y, s=55, color=PLUS5_COLOR, edgecolor=SURFACE, linewidth=1.5, zorder=4,
			label="+3 HIRES RVs in 2027A" if y == 0 else None)
		gain = 100 * (plus3 / now - 1)
		if gain >= 0.5:
			axis.text(max(now, plus3) + 0.006, y, f"+{gain:.0f}%", va="center", fontsize=8.5, color=MUTED)
	axis.set_yticks(range(len(rows)))
	axis.set_yticklabels([row[0] for row in rows], fontsize=10, color=INK)
	axis.set_xlim(0.55, 1.03)
	axis.set_xlabel("Mean 2027B S/N relative to a perfectly known ephemeris (best 5 windows)", color=MUTED)
	axis.legend(loc="lower left", frameon=False)
	_titles(
		figure, "2027B HRCCS S/N lost to ephemeris uncertainty, now and with 3 more RVs",
		"Keck, 2027-08-01 to 2028-01-31; hrccs-plan phase-sensitivity with a refitted conjunction.\n"
		"Targets with no window meeting the 30 km/s velocity-change cut are not shown.",
	)
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--targets", type=Path, required=True, help="table with fit_target and planet columns")
	parser.add_argument("--forecast", type=Path, required=True)
	parser.add_argument("--sensitivity", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	args = parser.parse_args()

	table = [row for row in csv.DictReader(args.targets.open()) if row["fit_target"]]
	targets = [(row["fit_target"], row["planet"]) for row in table]
	args.output_dir.mkdir(parents=True, exist_ok=True)
	sigma_t_chart(list(csv.DictReader(args.forecast.open())), targets, args.output_dir / "forecast_2027A_sigma_t.png")
	snr_chart(list(csv.DictReader(args.sensitivity.open())), dict(targets), args.output_dir / "forecast_2027B_hrccs_snr.png")
	fits = args.output_dir / "fits"
	fits.mkdir(exist_ok=True)
	for fit_name, _ in targets:
		shutil.copy2(plot_rvs.PLOTS_DIRECTORY / f"{fit_name}_rv_fit_plot.png", fits / f"{fit_name}_rv_fit_plot.png")
	print(f"Wrote 2 charts and {len(targets)} fit plots to {args.output_dir}")


if __name__ == "__main__":
	main()
