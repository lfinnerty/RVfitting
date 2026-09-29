#!/usr/bin/env python3
"""Download public RV databases into RVdatabases/, skipping anything already cached.

Files are written to ``<name>.part`` and renamed only once complete, so an
interrupted run resumes where it stopped and never leaves a truncated file in
place of a valid one.
"""

import argparse
import gzip
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

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
}
# The NASA Exoplanet Archive bulk-download script (tracked in git) lists every table.
EXOARCHIVE_WGET_SCRIPT = rv_io.EXOARCHIVE_DATABASE / "wget_exoarchive_20260924.bat"
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


SOURCES = {"vizier": download_vizier, "exoarchive": download_exoarchive, "sophie": download_sophie}


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
