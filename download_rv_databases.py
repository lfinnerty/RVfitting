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
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time

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
HARPS_DRS_CCF_DIRECTORY = rv_io.HARPS_DRS_DATABASE.parent / "ccf"
HARPS_DRS_MIN_MJD = 59_580.0  # 2022-01-01: earlier HARPS RVs come from the RVBank releases
HARPS_NAME_MATCH_HALF_BOX_DEG = 20 / 3600  # wider box, name-checked, for high proper-motion stars
HARPS_MAX_PER_NIGHT = 5  # time-series nights are thinned to this many evenly spaced spectra
HARPS_DRS_COLUMNS = ["target", "dp_id", "file", "bjd", "rv_km_s", "rv_error_km_s", "mask", "pipeline", "program"]
ELODIE_URL = "http://atlas.obs-hp.fr/elodie/fE.cgi"
ELODIE_CONE_ARCSEC = 20.0
ELODIE_COLUMNS = ["target", "night", "imanum", "mask", "bjd", "rv_km_s", "sn", "sigfit", "ampfit"]
ESPRESSO_COLUMNS = ["target", "dp_id", "file", "bjd", "rv_km_s", "rv_error_km_s", "mask", "mode", "pipeline", "program"]
ARXIV_EPRINT = "https://arxiv.org/e-print"
LITERATURE_DIRECTORY = rv_io.LITERATURE_DATABASE.parent
# RV tables transcribed from arXiv sources: (arXiv ID, TeX file, line that starts the
# table, star, instrument, BJD offset, reference, year). Rows are read from the start
# line up to the next \hline or \enddata; each row's first three numbers are BJD, RV
# and error in m/s.
LITERATURE_TABLES = [
	("2111.15028", "rv_timeseries.tex", r"\multicolumn{3}{c}{HIP086221}", "HIP 86221", "CHIRON", 2_450_000,
	 "Paredes+2021", 2021),
	("1310.7328", "ms.tex", r"\tablecaption{Relative Radial Velocities of HD 285507", "HD 285507", "TRES",
	 2_456_000, "Quinn+2014", 2014),
]
LITERATURE_COLUMNS = ["star", "bjd", "rv_m_s", "rv_error_m_s", "instrument", "reference", "year"]
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


def _eso_products(
	ra_deg: float, dec_deg: float, instrument: str, half_box_deg: float = ESO_SEARCH_HALF_BOX_DEG,
	min_mjd: float | None = None,
) -> list[dict]:
	"""Public ``instrument`` spectrum products (dp_id, target_name, s_ra, s_dec, t_min) in a box."""
	half_ra = half_box_deg / np.cos(np.radians(dec_deg))
	query = (
		"SELECT dp_id, target_name, s_ra, s_dec, t_min FROM ivoa.ObsCore"
		f" WHERE instrument_name = '{instrument}' AND dataproduct_type = 'spectrum'"
		f" AND s_ra BETWEEN {ra_deg - half_ra} AND {ra_deg + half_ra}"
		f" AND s_dec BETWEEN {dec_deg - half_box_deg} AND {dec_deg + half_box_deg}"
		+ (f" AND t_min >= {min_mjd}" if min_mjd is not None else "")
	)
	response = requests.get(
		ESO_TAP, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}, timeout=180
	)
	response.raise_for_status()
	return list(csv.DictReader(io.StringIO(response.text)))


def _datalink_files(dp_id: str) -> list[tuple[str, str]]:
	"""(URL, file name) of every file the ESO datalink service associates with a product."""
	response = requests.get(ESO_DATALINK, params={"ID": f"ivo://eso.org/ID?{dp_id}"}, timeout=120)
	response.raise_for_status()
	files = []
	for row in re.findall(r"<TR>(.*?)</TR>", response.text, re.S):
		cells = [re.sub(r"<!\[CDATA\[|\]\]>", "", cell).strip() for cell in re.findall(r"<TD>(.*?)</TD>", row, re.S)]
		if len(cells) > 6:
			files.append((cells[1], cells[6]))
	return files


def _espresso_ccf(dp_id: str) -> Path:
	"""Download (once) the science-fibre CCF file associated with an ESPRESSO product."""
	for url, name in _datalink_files(dp_id):
		# ESPRESSO_CCF_A_* (newer DRS) or ES_SCCA_* (older); not the telluric-corrected CCF
		if re.match(r"(ESPRESSO_CCF_A_|ES_SCCA_)", name):
			destination = ESPRESSO_CCF_DIRECTORY / name
			download_file(url, destination)
			return destination
	raise RuntimeError(f"No CCF file linked to {dp_id}")


def _harps_ccf(dp_id: str) -> Path:
	"""Extract (once) the fibre-A CCF file from a HARPS DRS 3.x product's ancillary tarball."""
	for url, name in _datalink_files(dp_id):
		if not name.endswith(".tar"):
			continue
		stem = name.removesuffix(".tar")
		cached = sorted(HARPS_DRS_CCF_DIRECTORY.glob(f"{stem.split('_DRS_')[0]}_ccf_*_A.fits"))
		if cached:
			return cached[0]
		archive = HARPS_DRS_CCF_DIRECTORY / name
		download_file(url, archive)
		with tarfile.open(archive) as tar:
			member = next((m for m in tar.getmembers() if re.search(r"_ccf_[^/]*_A\.fits$", m.name)), None)
			if member is None:
				archive.unlink()
				raise RuntimeError(f"No CCF file in {name}")
			destination = HARPS_DRS_CCF_DIRECTORY / Path(member.name).name
			with tar.extractfile(member) as source, destination.open("wb") as output:
				shutil.copyfileobj(source, output)
		archive.unlink()
		return destination
	raise RuntimeError(f"No DRS tarball linked to {dp_id}")


def _per_target_download(label, path, columns, targets, refresh, fetch) -> None:
	"""Run ``fetch(target, simbad_record) -> rows`` for targets not yet queried; keep one CSV.

	Targets whose fetch fails are not marked as queried, so the next run retries them.
	"""
	targets = targets or fitted_targets()
	path.parent.mkdir(parents=True, exist_ok=True)
	existing = list(csv.DictReader(path.open())) if path.is_file() else []
	queried_path = path.with_name("queried_targets.json")
	queried = json.loads(queried_path.read_text()) if queried_path.is_file() else {}
	pending = [t for t in targets if refresh or t not in queried]
	print(f"{label}: {len(pending)} of {len(targets)} targets to query")
	records = rv_io.simbad_records(pending)
	rows = [row for row in existing if row["target"] not in pending]
	for target in pending:
		record = records.get(target)
		if record is None:
			print(f"  {target}: no SIMBAD position, skipped", file=sys.stderr)
			continue
		try:
			found = fetch(target, record)
		except (requests.RequestException, RuntimeError, OSError, KeyError) as error:
			print(f"  {target}: failed ({error}); will retry next run", file=sys.stderr)
			continue
		rows.extend(found)
		queried[target] = datetime.date.today().isoformat()
		print(f"  {target}: {len(found)} {label} RVs")
	if rows:
		with path.open("w", newline="") as output:
			writer = csv.DictWriter(output, fieldnames=columns)
			writer.writeheader()
			writer.writerows(rows)
	queried_path.write_text(json.dumps(queried, indent=1) + "\n")


def download_espresso(targets: list[str] | None = None, refresh: bool = False) -> None:
	"""Collect ESPRESSO DRS CCF RVs from the ESO archive for ``targets``.

	Each public spectrum's CCF file (~0.3 MB) is cached in ESPRESSO/ccf/, and its
	header RV, error, BJD, mask, and mode go into one CSV. Targets already queried
	are skipped unless ``refresh``.
	"""
	ESPRESSO_CCF_DIRECTORY.mkdir(parents=True, exist_ok=True)

	def fetch(target, record):
		found = []
		for product in _eso_products(record["ra_deg"], record["dec_deg"], "ESPRESSO"):
			header = fits.getheader(_espresso_ccf(product["dp_id"]))
			if "HIERARCH ESO QC CCF RV" not in header:
				continue
			found.append({
				"target": target, "dp_id": product["dp_id"], "file": header.get("ARCFILE", ""),
				"bjd": header["HIERARCH ESO QC BJD"], "rv_km_s": header["HIERARCH ESO QC CCF RV"],
				"rv_error_km_s": header["HIERARCH ESO QC CCF RV ERROR"],
				"mask": header["HIERARCH ESO QC CCF MASK"], "mode": header["HIERARCH ESO INS MODE"],
				"pipeline": header.get("HIERARCH ESO PRO REC1 PIPE ID", ""),
				"program": header.get("HIERARCH ESO OBS PROG ID", ""),
			})
		return found

	_per_target_download("ESPRESSO", rv_io.ESPRESSO_DATABASE, ESPRESSO_COLUMNS, targets, refresh, fetch)


def _thin_per_night(products: list[dict], limit: int) -> list[dict]:
	"""Keep at most ``limit`` evenly spaced products per night (MJD rounded down at local noon)."""
	nights: dict[int, list[dict]] = {}
	for product in sorted(products, key=lambda p: float(p["t_min"])):
		nights.setdefault(int(float(product["t_min"]) - 0.5), []).append(product)
	kept = []
	for night in nights.values():
		indices = np.unique(np.linspace(0, len(night) - 1, min(limit, len(night))).round().astype(int))
		kept += [night[index] for index in indices]
	return kept


def download_harps_drs(targets: list[str] | None = None, refresh: bool = False) -> None:
	"""Collect HARPS DRS 3.x CCF RVs from ESO archive products taken since 2022.

	Products are matched within ESO_SEARCH_HALF_BOX_DEG of the SIMBAD position, or
	within HARPS_NAME_MATCH_HALF_BOX_DEG if their target name is one of the star's
	identifiers (for high proper-motion stars). Nights with long time series are
	thinned to HARPS_MAX_PER_NIGHT spectra. Only the fibre-A CCF of each ~6 MB
	tarball is kept, in HARPS_DRS/ccf/.
	"""
	HARPS_DRS_CCF_DIRECTORY.mkdir(parents=True, exist_ok=True)

	def fetch(target, record):
		ra, dec = record["ra_deg"], record["dec_deg"]
		keys = rv_io.identifier_keys(target)
		products = [
			product for product in _eso_products(ra, dec, "HARPS", HARPS_NAME_MATCH_HALF_BOX_DEG, HARPS_DRS_MIN_MJD)
			if rv_io.catalog_key(product["target_name"]) in keys
			or (abs(float(product["s_dec"]) - dec) < ESO_SEARCH_HALF_BOX_DEG
				and abs((float(product["s_ra"]) - ra + 180) % 360 - 180) * np.cos(np.radians(dec)) < ESO_SEARCH_HALF_BOX_DEG)
		]
		found = []
		for product in _thin_per_night(products, HARPS_MAX_PER_NIGHT):
			header = fits.getheader(_harps_ccf(product["dp_id"]))
			if "HIERARCH ESO DRS CCF RVC" not in header:
				continue
			found.append({
				"target": target, "dp_id": product["dp_id"], "file": header.get("ARCFILE", ""),
				"bjd": header["HIERARCH ESO DRS BJD"], "rv_km_s": header["HIERARCH ESO DRS CCF RVC"],
				"rv_error_km_s": header["HIERARCH ESO DRS CCF NOISE"],
				"mask": header["HIERARCH ESO DRS CCF MASK"],
				"pipeline": header.get("HIERARCH ESO DRS VERSION", ""),
				"program": header.get("HIERARCH ESO OBS PROG ID", ""),
			})
		return found

	_per_target_download("HARPS DRS", rv_io.HARPS_DRS_DATABASE, HARPS_DRS_COLUMNS, targets, refresh, fetch)


@cache
def _ohp() -> EarthLocation:
	return EarthLocation(lon=5.7133 * u.deg, lat=43.9317 * u.deg, height=650 * u.m)


def _elodie_bjd(night: str, imanum: str, ra_deg: float, dec_deg: float) -> float | None:
	"""Mid-exposure BJD_TDB of an ELODIE spectrum from its header's UT start and exposure
	time (None if the header has no time stamp)."""
	text = requests.get(ELODIE_URL, params={"n": "e500", "c": "i", "z": "fd", "o": f"elodie:{night}/{imanum}"}, timeout=120).text
	text = re.sub(r"<[^>]+>", " ", text)
	values = {key: re.search(rf"{key}\s+'?([0-9.E+-]+)", text) for key in ("DATETU", "HDEBUT", "EXPTIME")}
	if not all(values.values()):
		return None
	date, start_hours, exposure = values["DATETU"].group(1), float(values["HDEBUT"].group(1)), float(values["EXPTIME"].group(1))
	middle = Time(f"{date[:4]}-{date[4:6]}-{date[6:]}", scale="utc") + (start_hours / 24 + exposure / 172_800) * u.day
	star = SkyCoord(ra_deg, dec_deg, unit="deg")
	return float((middle.tdb + middle.light_travel_time(star, location=_ohp())).jd)


def download_elodie(targets: list[str] | None = None, refresh: bool = False) -> None:
	"""Collect ELODIE (OHP, 1994-2006) CCF RVs within ELODIE_CONE_ARCSEC of each target.

	The CCF table gives the barycentric RV (vfit), S/N, and CCF width and depth but no
	time stamp, so each spectrum's header is read for its UT start and exposure time.
	Failed CCF fits (vfit = 0), sky-fibre CCFs, and spectra without a time stamp are skipped.
	"""
	def fetch(target, record):
		ra, dec = record["ra_deg"], record["dec_deg"]
		centre = SkyCoord(ra, dec, unit="deg")
		name = "J" + re.sub(r"(\d{6}\.\d)\d*", r"\1", centre.to_string("hmsdms", sep="", precision=1).replace(" ", ""), count=1)
		name = re.sub(r"\.\d+$", "", name)
		text = requests.get(ELODIE_URL, params={"n": "e501", "a": "csv", "o": name}, timeout=180).text
		if not re.search(r"matched (no|\d+) records?", text):
			raise RuntimeError("ELODIE query returned no cone-search summary")
		found = []
		for line in text.splitlines():
			fields = line.split("\t")
			if line.startswith(("#", "$")) or len(fields) < 14 or fields[6] != "obj":
				continue
			night, imanum, mask, sn, vfit, sigfit, ampfit = fields[4], fields[5], fields[7], fields[10], fields[11], fields[12], fields[13]
			pointing = SkyCoord(fields[1][1:3] + "h" + fields[1][3:5] + "m" + fields[1][5:9] + "s " + fields[1][9:12] + "d" + fields[1][12:14] + "m" + fields[1][14:] + "s")
			if float(vfit) == 0 or float(sigfit) <= 0 or pointing.separation(centre).arcsec > ELODIE_CONE_ARCSEC:
				continue
			bjd = _elodie_bjd(night, imanum, ra, dec)
			if bjd is None:
				continue
			found.append({
				"target": target, "night": night, "imanum": imanum, "mask": mask,
				"bjd": f"{bjd:.6f}", "rv_km_s": vfit, "sn": sn,
				"sigfit": sigfit, "ampfit": ampfit,
			})
		return found

	_per_target_download("ELODIE", rv_io.ELODIE_DATABASE, ELODIE_COLUMNS, targets, refresh, fetch)


def _arxiv_source(arxiv_id: str) -> Path:
	"""Download and unpack (once) an arXiv e-print into literature/<id>/."""
	directory = LITERATURE_DIRECTORY / arxiv_id
	if directory.is_dir():
		return directory
	archive = LITERATURE_DIRECTORY / f"{arxiv_id}.tar"
	download_file(f"{ARXIV_EPRINT}/{arxiv_id}", archive)
	unpacked = directory.with_name(directory.name + ".part")
	unpacked.mkdir(parents=True, exist_ok=True)
	try:
		with tarfile.open(archive) as tar:
			tar.extractall(unpacked, filter="data")
	except tarfile.ReadError:  # single-file submissions are gzipped TeX
		with gzip.open(archive) as source, (unpacked / "main.tex").open("wb") as output:
			shutil.copyfileobj(source, output)
	archive.unlink()
	unpacked.rename(directory)
	return directory


def _table_rows(tex: str, start: str) -> list[list[float]]:
	"""The numeric rows of a TeX table from the line containing ``start``."""
	lines = tex.splitlines()
	first = next(index for index, line in enumerate(lines) if start in line)
	rows, started = [], False
	for line in lines[first + 1:]:
		code = line.split("%")[0]
		if started and re.search(r"\\(hline|enddata)", code):
			break
		if code.count("&") >= 2:
			numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", re.sub(r"\$\^\{?\\star\}?\$|\\[A-Za-z]+", " ", code))
			if len(numbers) >= 3:
				rows.append([float(number) for number in numbers[:3]])
				started = True
	return rows


def download_literature() -> None:
	"""Transcribe the RV tables in LITERATURE_TABLES from their arXiv sources."""
	LITERATURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
	rows = []
	for arxiv_id, tex_file, start, star, instrument, offset, reference, year in LITERATURE_TABLES:
		table = _table_rows((_arxiv_source(arxiv_id) / tex_file).read_text(errors="replace"), start)
		print(f"  {reference} ({arxiv_id}): {len(table)} {instrument} RVs of {star}")
		rows += [[star, f"{bjd + offset:.6f}", rv, error, instrument, reference, year] for bjd, rv, error in table]
	with rv_io.LITERATURE_DATABASE.open("w", newline="") as output:
		writer = csv.writer(output, lineterminator="\n")
		writer.writerow(LITERATURE_COLUMNS)
		writer.writerows(rows)


SOURCES = {
	"vizier": download_vizier,
	"exoarchive": download_exoarchive,
	"sophie": download_sophie,
	"neid": download_neid,
	"espresso": download_espresso,
	"harps_drs": download_harps_drs,
	"elodie": download_elodie,
	"literature": download_literature,
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
