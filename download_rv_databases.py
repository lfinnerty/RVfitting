#!/usr/bin/env python3
"""Download public RV databases into RVdatabases/, skipping anything already cached.

Files are written to ``<name>.part`` and renamed only once complete, so an
interrupted run resumes where it stopped and never leaves a truncated file in
place of a valid one.
"""

import argparse
import csv
import datetime
import gzip
import io
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

import rv_io


VIZIER_FTP = "https://cdsarc.cds.unistra.fr/ftp"
VIZIER_CATALOGS = {
	# directory under RVdatabases/: (catalog, {local name: remote name}).
	# A remote ".gz" saved under a local name without ".gz" is decompressed.
	".": ("J/A+A/702/A68", {"ReadMe_Teklu": "ReadMe", "tablea1_Teklu.dat": "tablea1.dat.gz"}),
	"HARPS_RVBank2": (
		"J/A+A/683/A125",
		{"ReadMe": "ReadMe", "table1.dat.gz": "table1.dat.gz", "table4.dat.gz": "table4.dat.gz"},
	),
	"HARPS": ("J/A+A/636/A74", {"Readme": "ReadMe", "rvbank.dat": "rvbank.dat.gz"}),
	"CLS": (
		"J/ApJS/255/8",
		{"ReadMe": "ReadMe", "table2.dat": "table2.dat", "table6.dat.gz": "table6.dat.gz"},
	),
	"Hebrard2016": ("J/A+A/588/A145", {"Readme": "ReadMe", "rvdata.dat": "table1.dat"}),
	"Neveu-VanMalle2014": (
		"J/A+A/572/A49",
		{"ReadMe": "ReadMe", "w94a_rv.dat": "w94a_rv.dat", "w94b_rv.dat": "w94b_rv.dat"},
	),
}
# The NASA Exoplanet Archive bulk-download script (tracked in git) lists every table.
EXOARCHIVE_WGET_SCRIPT = rv_io.EXOARCHIVE_DATABASE / "wget_exoarchive_20260924.bat"
NEID_TAP = "https://neid.ipac.caltech.edu/TAP/sync"
NEID_SEARCH_RADIUS_DEG = 60 / 3600  # headers record requested coordinates; allow proper motion
NEID_COLUMNS = "qobject, obsdate, obstype, obsmode, swversion, ccfjdsum, ccfrvmod, dvrms, l2filename, l2propint, program"
ESO_TAP = "https://archive.eso.org/tap_obs/sync"
ESO_DATALINK = "https://archive.eso.org/datalink/links"
ESO_SEARCH_HALF_BOX_DEG = 5 / 3600  # pointings are at the target; stays clear of 15" binary companions
ESPRESSO_CCF_DIRECTORY = rv_io.ESPRESSO_DATABASE.parent / "ccf"
ESPRESSO_COLUMNS = ["target", "dp_id", "file", "bjd", "rv_km_s", "rv_error_km_s", "mask", "mode", "pipeline", "program"]
SOPHIE_URL = "http://atlas.obs-hp.fr/sophie/sophie.cgi"
SOPHIE_FIELDS = "seq,objname,bjd,mask,ccf_offline,rv,err"
# The SOPHIE server drops connections after ~2 minutes, so query small RA bins.
SOPHIE_RA_BIN_MINUTES = 10
SOPHIE_END_MARKER = "#-- end of data table"


def download_file(url: str, destination: Path, attempts: int = 5) -> None:
	"""Download ``url`` to ``destination``, resuming a partial ``.part`` file.

	A ``.gz`` URL saved to a destination without ``.gz`` is decompressed.
	"""
	if destination.is_file():
		return
	decompress = url.endswith(".gz") and destination.suffix != ".gz"
	partial = destination.with_name(destination.name + (".gz" if decompress else "") + ".part")
	for attempt in range(1, attempts + 1):
		offset = partial.stat().st_size if partial.is_file() else 0
		headers = {"Range": f"bytes={offset}-"} if offset else {}
		try:
			with requests.get(url, headers=headers, stream=True, timeout=120) as response:
				if response.status_code == 416:  # already complete
					break
				response.raise_for_status()
				mode = "ab" if response.status_code == 206 else "wb"
				with partial.open(mode) as output:
					for chunk in response.iter_content(1 << 20):
						output.write(chunk)
			break
		except requests.RequestException as error:
			print(f"  {destination.name}: attempt {attempt} failed ({error})", file=sys.stderr)
			time.sleep(10)
	else:
		raise RuntimeError(f"Could not download {url}")
	if url.endswith(".gz"):
		# Reading the whole stream raises if the file is truncated or corrupt.
		with gzip.open(partial) as data:
			if decompress:
				unpacked = destination.with_name(destination.name + ".part")
				with unpacked.open("wb") as output:
					shutil.copyfileobj(data, output, 1 << 24)
				partial.unlink()
				partial = unpacked
			else:
				while data.read(1 << 24):
					pass
	partial.rename(destination)
	print(f"  downloaded {destination}")


def download_vizier() -> None:
	for directory, (catalog, files) in VIZIER_CATALOGS.items():
		target = rv_io.DATABASE_ROOT / directory
		target.mkdir(parents=True, exist_ok=True)
		print(f"VizieR {catalog} -> {target.resolve()}")
		for local_name, remote_name in files.items():
			download_file(f"{VIZIER_FTP}/{catalog}/{remote_name}", target / local_name)


def download_exoarchive(workers: int = 4) -> None:
	"""Fetch any ExoArchive RV tables listed in the bulk script but missing locally."""
	commands = re.findall(
		r"^wget -O (\S+) (\S+)", EXOARCHIVE_WGET_SCRIPT.read_text(encoding="utf-8"), re.MULTILINE
	)
	missing = [
		(url.replace("http://", "https://").replace(":80/", "/"), rv_io.EXOARCHIVE_DATABASE / name)
		for name, url in commands
		if not (rv_io.EXOARCHIVE_DATABASE / name).is_file()
	]
	print(f"ExoArchive: {len(missing)} of {len(commands)} tables to fetch")
	with ThreadPoolExecutor(workers) as pool:
		list(pool.map(lambda job: download_file(*job), missing))


def _sophie_bin(start_minute: int) -> str | None:
	"""Return one RA bin of the SOPHIE CCF table, or None if the reply was incomplete."""
	end_minute = start_minute + SOPHIE_RA_BIN_MINUTES

	def ra(minute: int) -> str:
		return f"{minute // 60:02d} {minute % 60:02d} 00"

	params = {
		"n": "sophiecc",
		"a": "csv",
		"ob": "ra,seq",
		"r": f"[{ra(start_minute)} -90],[{ra(end_minute) if end_minute < 1440 else '23 59 59.99'} +90]",
		"d": SOPHIE_FIELDS,
	}
	try:
		response = requests.get(SOPHIE_URL, params=params, timeout=300)
		response.raise_for_status()
	except requests.RequestException:
		return None
	return response.text if SOPHIE_END_MARKER in response.text else None


def _save_sophie_bin(start_minute: int) -> bool:
	"""Fetch and save one SOPHIE RA bin; return whether it succeeded."""
	text = _sophie_bin(start_minute)
	if text is None:
		return False
	(rv_io.SOPHIE_DATABASE / f"ccf_ra{start_minute:04d}.txt").write_text(text, encoding="utf-8")
	return True


def download_sophie(workers: int = 2, attempts: int = 4) -> None:
	"""Download the public SOPHIE CCF radial velocities in RA bins."""
	rv_io.SOPHIE_DATABASE.mkdir(parents=True, exist_ok=True)
	pending = [
		minute for minute in range(0, 1440, SOPHIE_RA_BIN_MINUTES)
		if not (rv_io.SOPHIE_DATABASE / f"ccf_ra{minute:04d}.txt").is_file()
	]
	print(f"SOPHIE archive: {len(pending)} of {1440 // SOPHIE_RA_BIN_MINUTES} RA bins to fetch")
	for attempt in range(1, attempts + 1):
		if not pending:
			break
		with ThreadPoolExecutor(workers) as pool:
			succeeded = list(pool.map(_save_sophie_bin, pending))
		failed = [minute for minute, ok in zip(pending, succeeded) if not ok]
		print(f"  pass {attempt}: {len(pending) - len(failed)} bins saved, {len(failed)} failed")
		pending = failed
	if pending:
		print(f"  {len(pending)} SOPHIE bins still missing; re-run to retry", file=sys.stderr)


def fitted_targets() -> list[str]:
	"""Stars with saved fits in plots/ (the default NEID target list)."""
	names = []
	for path in sorted((rv_io.DATABASE_ROOT.parent / "plots").glob("*_rv_fit_parameters.json")):
		names.append(json.loads(path.read_text())["input_star"])
	return names


def _public(row: dict, today: datetime.date) -> bool:
	observed = datetime.date.fromisoformat(row["obsdate"][:10])
	months = observed.month - 1 + int(float(row["l2propint"] or 0))
	return datetime.date(observed.year + months // 12, months % 12 + 1, min(observed.day, 28)) <= today


def download_neid(targets: list[str] | None = None, refresh: bool = False) -> None:
	"""Query NEID L2 CCF RVs (from the archive's metadata table) for ``targets``.

	The table carries the barycentric CCF RV and its uncertainty, so no FITS files
	are needed. Results for all targets are kept in one CSV; targets already
	queried are skipped unless ``refresh``.
	"""
	targets = targets or fitted_targets()
	path = rv_io.NEID_DATABASE
	path.parent.mkdir(parents=True, exist_ok=True)
	existing = list(csv.DictReader(path.open())) if path.is_file() else []
	queried_path = path.with_name("queried_targets.json")
	queried = json.loads(queried_path.read_text()) if queried_path.is_file() else {}
	pending = [t for t in targets if refresh or t not in queried]
	print(f"NEID: {len(pending)} of {len(targets)} targets to query")
	records = rv_io.simbad_records(pending)
	today = datetime.date.today()
	rows = [row for row in existing if row["target"] not in pending]
	for target in pending:
		record = records.get(target)
		if record is None:
			print(f"  {target}: no SIMBAD position, skipped", file=sys.stderr)
			continue
		query = (
			f"select {NEID_COLUMNS} from neidl2 where obstype = 'Sci' and contains(point('icrs', qrad, qdecd),"
			f" circle('icrs', {record['ra_deg']}, {record['dec_deg']}, {NEID_SEARCH_RADIUS_DEG})) = 1"
		)
		try:
			response = requests.get(NEID_TAP, params={"query": query, "format": "csv"}, timeout=180)
			response.raise_for_status()
		except requests.RequestException as error:
			print(f"  {target}: query failed ({error}); will retry next run", file=sys.stderr)
			continue
		found = [row for row in csv.DictReader(io.StringIO(response.text)) if _public(row, today)]
		rows.extend({"target": target, **row} for row in found)
		queried[target] = today.isoformat()
		print(f"  {target}: {len(found)} public NEID RVs")
	if rows:
		with path.open("w", newline="") as output:
			writer = csv.DictWriter(output, fieldnames=["target", *[c.strip() for c in NEID_COLUMNS.split(",")]])
			writer.writeheader()
			writer.writerows(rows)
	queried_path.write_text(json.dumps(queried, indent=1) + "\n")


def _espresso_products(ra_deg: float, dec_deg: float) -> list[str]:
	"""Public ESPRESSO spectrum product IDs pointed within the search box around a position."""
	half_ra = ESO_SEARCH_HALF_BOX_DEG / np.cos(np.radians(dec_deg))
	query = (
		"SELECT dp_id FROM ivoa.ObsCore WHERE instrument_name = 'ESPRESSO' AND dataproduct_type = 'spectrum'"
		f" AND s_ra BETWEEN {ra_deg - half_ra} AND {ra_deg + half_ra}"
		f" AND s_dec BETWEEN {dec_deg - ESO_SEARCH_HALF_BOX_DEG} AND {dec_deg + ESO_SEARCH_HALF_BOX_DEG}"
	)
	response = requests.get(
		ESO_TAP, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}, timeout=180
	)
	response.raise_for_status()
	return [row["dp_id"] for row in csv.DictReader(io.StringIO(response.text))]


def _espresso_ccf(dp_id: str) -> Path:
	"""Download (once) the science-fibre CCF file associated with an ESPRESSO product."""
	response = requests.get(ESO_DATALINK, params={"ID": f"ivo://eso.org/ID?{dp_id}"}, timeout=120)
	response.raise_for_status()
	for row in re.findall(r"<TR>(.*?)</TR>", response.text, re.S):
		cells = [re.sub(r"<!\[CDATA\[|\]\]>", "", cell).strip() for cell in re.findall(r"<TD>(.*?)</TD>", row, re.S)]
		# ESPRESSO_CCF_A_* (newer DRS) or ES_SCCA_* (older); not the telluric-corrected CCF
		if len(cells) > 6 and re.match(r"(ESPRESSO_CCF_A_|ES_SCCA_)", cells[6]):
			destination = ESPRESSO_CCF_DIRECTORY / cells[6]
			download_file(cells[1], destination)
			return destination
	raise RuntimeError(f"No CCF file linked to {dp_id}")


def download_espresso(targets: list[str] | None = None, refresh: bool = False) -> None:
	"""Collect ESPRESSO DRS CCF RVs from the ESO archive for ``targets``.

	Each public spectrum's CCF file (~0.3 MB) is cached in ESPRESSO/ccf/, and its
	header RV, error, BJD, mask, and mode go into one CSV. Targets already queried
	are skipped unless ``refresh``.
	"""
	targets = targets or fitted_targets()
	path = rv_io.ESPRESSO_DATABASE
	ESPRESSO_CCF_DIRECTORY.mkdir(parents=True, exist_ok=True)
	existing = list(csv.DictReader(path.open())) if path.is_file() else []
	queried_path = path.with_name("queried_targets.json")
	queried = json.loads(queried_path.read_text()) if queried_path.is_file() else {}
	pending = [t for t in targets if refresh or t not in queried]
	print(f"ESPRESSO: {len(pending)} of {len(targets)} targets to query")
	records = rv_io.simbad_records(pending)
	rows = [row for row in existing if row["target"] not in pending]
	for target in pending:
		record = records.get(target)
		if record is None:
			print(f"  {target}: no SIMBAD position, skipped", file=sys.stderr)
			continue
		try:
			found = []
			for dp_id in _espresso_products(record["ra_deg"], record["dec_deg"]):
				header = fits.getheader(_espresso_ccf(dp_id))
				if "HIERARCH ESO QC CCF RV" not in header:
					continue
				found.append({
					"target": target, "dp_id": dp_id, "file": header.get("ARCFILE", ""),
					"bjd": header["HIERARCH ESO QC BJD"], "rv_km_s": header["HIERARCH ESO QC CCF RV"],
					"rv_error_km_s": header["HIERARCH ESO QC CCF RV ERROR"],
					"mask": header["HIERARCH ESO QC CCF MASK"], "mode": header["HIERARCH ESO INS MODE"],
					"pipeline": header.get("HIERARCH ESO PRO REC1 PIPE ID", ""),
					"program": header.get("HIERARCH ESO OBS PROG ID", ""),
				})
		except (requests.RequestException, RuntimeError, OSError) as error:
			print(f"  {target}: failed ({error}); will retry next run", file=sys.stderr)
			continue
		rows.extend(found)
		queried[target] = datetime.date.today().isoformat()
		print(f"  {target}: {len(found)} ESPRESSO RVs")
	if rows:
		with path.open("w", newline="") as output:
			writer = csv.DictWriter(output, fieldnames=ESPRESSO_COLUMNS)
			writer.writeheader()
			writer.writerows(rows)
	queried_path.write_text(json.dumps(queried, indent=1) + "\n")


SOURCES = {
	"vizier": download_vizier,
	"exoarchive": download_exoarchive,
	"sophie": download_sophie,
	"neid": download_neid,
	"espresso": download_espresso,
}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument(
		"sources", nargs="*", help=f"any of: {', '.join(SOURCES)} (default: all)"
	)
	args = parser.parse_args()
	if unknown := set(args.sources) - set(SOURCES):
		parser.error(f"unknown source(s): {', '.join(sorted(unknown))}")
	for name, download in SOURCES.items():
		if not args.sources or name in args.sources:
			download()


if __name__ == "__main__":
	main()
