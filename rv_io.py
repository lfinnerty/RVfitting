"""Readers, host lookups, and SIMBAD caching for the local RV databases."""

import csv
import json
import re
import sys
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astroquery.simbad import Simbad


DATABASE_ROOT = Path(__file__).resolve().parent / "RVdatabases"
TEKLU_DATABASE = DATABASE_ROOT / "tablea1_Teklu.dat"
EXOARCHIVE_DATABASE = DATABASE_ROOT / "exoarchive"
FULTON_DATABASE = DATABASE_ROOT / "rv_data_fulton"
HEBRARD_DATABASE = DATABASE_ROOT / "Hebrard2016" / "rvdata.dat"
HARPS_DATABASE = DATABASE_ROOT / "HARPS" / "rvbank.dat"
SYNTHETICS_DATABASE = DATABASE_ROOT / "Synthetics"
SIMBAD_CACHE = DATABASE_ROOT / "simbad_cache.json"
MINIMUM_BJD = 2_447_161.5  # 1988-01-01 00:00 UTC
SIMBAD_BATCH_SIZE = 500

# Fixed-width name columns shared by the Teklu and HARPS catalogs.
TEKLU_NAME = slice(0, 14)
TEKLU_SIMBAD_NAME = slice(15, 45)
HARPS_NAME = slice(0, 14)

EXOARCHIVE_TIME_FRAMES = {
	# "HJD-TBD" is a typo present in some archive files.
	"BJD", "BJD-UTC", "BJD-TDB", "JD", "JD-UTC", "HJD", "HJD-UTC", "HJD-TBD", "FCJD", "MJD",
}
OBSERVATORY_ALIASES = (
	("mauna kea", "keck"),
	("manua kea", "keck"),
	("maun kea", "keck"),
	("lick", "lick observatory"),
	("la silla", "La Silla Observatory"),
	("las campanas", "Las Campanas Observatory"),
	("mcdonald", "McDonald Observatory"),
	("apache point", "Apache Point Observatory"),
	("siding spring", "Siding Spring Observatory"),
	("okayama", "Okayama Astrophysical Observatory"),
	("paranal", "Cerro Paranal"),
	("roque de los muchachos", "Roque de los Muchachos"),
	("la palma", "Roque de los Muchachos"),
	("whipple", "Whipple Observatory"),
	("calar alto", "Observatorio de Calar Alto"),
	("haute provence", "ohp"),
	("kitt peak", "Kitt Peak National Observatory"),
	("xinglong", "Beijing XingLong Observatory"),
)
# Sites missing from Astropy's registry. Barycentric corrections are insensitive to
# kilometre-level position errors, so approximate coordinates suffice.
OBSERVATORY_COORDINATES = {
	"tautenburg": EarthLocation.from_geodetic(11.711 * u.deg, 50.980 * u.deg, 341 * u.m),
	"bohyunsan": EarthLocation.from_geodetic(128.977 * u.deg, 36.165 * u.deg, 1162 * u.m),
}

RVData = tuple[np.ndarray, np.ndarray, np.ndarray]


def normalize_identifier(identifier: str) -> str:
	"""Normalize catalog identifiers for case- and whitespace-insensitive matching."""
	return "".join(identifier.casefold().split())


# ---------------------------------------------------------------------------
# Host identifiers
# ---------------------------------------------------------------------------

def tbl_star_id(text: str) -> str | None:
	"""Return the STAR_ID from an ExoArchive-style ``.tbl`` file."""
	match = re.search(r"^\\STAR_ID\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
	return match.group(1).strip() if match else None


def fulton_star_id(first_line: str) -> str | None:
	"""Return the HD identifier from the first line of a Fulton RV CSV file."""
	match = re.fullmatch(r"# star HD number,\s*(\d+)\s*\n?", first_line)
	return f"HD {match.group(1)}" if match else None


def teklu_names(line: str) -> tuple[str, str]:
	"""Return the catalog and SIMBAD names from a Teklu table row."""
	return line[TEKLU_NAME].strip(), line[TEKLU_SIMBAD_NAME].strip()


def exoarchive_hosts(directory: Path = EXOARCHIVE_DATABASE) -> Iterator[tuple[str, Path]]:
	"""Yield (host, path) for each ExoArchive table with a STAR_ID."""
	for path in sorted(directory.glob("*.tbl")):
		host = tbl_star_id(path.read_text(encoding="utf-8", errors="replace"))
		if host:
			yield host, path


def fulton_hosts(directory: Path = FULTON_DATABASE) -> Iterator[tuple[str, Path]]:
	"""Yield (host, path) for each Fulton CSV with an HD-number header."""
	for path in sorted(directory.glob("*_rv.csv")):
		with path.open(encoding="utf-8") as data_file:
			host = fulton_star_id(data_file.readline())
		if host:
			yield host, path


def database_hosts() -> set[str]:
	"""Collect host identifiers from every supported local RV database."""
	identifiers = {host for host, _ in exoarchive_hosts()}
	identifiers.update(host for host, _ in fulton_hosts())
	if TEKLU_DATABASE.is_file():
		with TEKLU_DATABASE.open(encoding="ascii") as data_file:
			for line in data_file:
				identifiers.update(teklu_names(line))
	if HEBRARD_DATABASE.is_file():
		with HEBRARD_DATABASE.open(encoding="ascii") as data_file:
			for line in data_file:
				fields = line.split()
				if len(fields) >= 6:
					identifiers.add(fields[4])
	if HARPS_DATABASE.is_file():
		with HARPS_DATABASE.open(encoding="ascii") as data_file:
			identifiers.update(line[HARPS_NAME].strip() for line in data_file)
	identifiers.discard("")
	return identifiers


# ---------------------------------------------------------------------------
# SIMBAD cache
# ---------------------------------------------------------------------------

_simbad_cache: dict[str, dict] | None = None


def _load_simbad_cache() -> dict[str, dict]:
	global _simbad_cache
	if _simbad_cache is None:
		_simbad_cache = (
			json.loads(SIMBAD_CACHE.read_text(encoding="utf-8"))
			if SIMBAD_CACHE.is_file() else {}
		)
	return _simbad_cache


def simbad_records(identifiers: Iterable[str]) -> dict[str, dict]:
	"""Return SIMBAD records keyed by identifier, batch-querying only uncached ones.

	Each record has ``main_id``, ``ra_deg``, ``dec_deg``, ``rv_km_s`` (or None),
	and ``ids``. Unresolved identifiers and failed queries are omitted and not cached.
	"""
	stored = _load_simbad_cache()
	records = {}
	missing = []
	for identifier in dict.fromkeys(identifiers):
		record = stored.get(normalize_identifier(identifier))
		if record is None:
			missing.append(identifier)
		else:
			records[identifier] = record
	if not missing:
		return records

	print(f"Querying SIMBAD for {len(missing)} uncached identifier(s)")
	simbad = Simbad()
	simbad.add_votable_fields("ids", "rvz_radvel")
	updated = False
	for start in range(0, len(missing), SIMBAD_BATCH_SIZE):
		batch = missing[start:start + SIMBAD_BATCH_SIZE]
		try:
			result = simbad.query_objects(batch)
		except Exception as error:
			print(f"Warning: SIMBAD query failed: {error}", file=sys.stderr)
			continue
		for row in result:
			main_id = str(row["main_id"]).strip()
			if not main_id or np.ma.is_masked(row["ra"]):
				continue
			velocity = row["rvz_radvel"]
			record = {
				"main_id": main_id,
				"ra_deg": float(row["ra"]),
				"dec_deg": float(row["dec"]),
				"rv_km_s": None if np.ma.is_masked(velocity) else float(velocity),
				"ids": [value.strip() for value in str(row["ids"]).split("|")],
			}
			identifier = str(row["user_specified_id"]).strip()
			stored[normalize_identifier(identifier)] = record
			records[identifier] = record
			updated = True
	if updated:
		SIMBAD_CACHE.write_text(json.dumps(stored, indent=1) + "\n", encoding="utf-8")
	return records


def simbad_record(identifier: str) -> dict | None:
	"""Return the cached or freshly queried SIMBAD record for one identifier."""
	return simbad_records([identifier]).get(identifier)


def find_teklu_id(database: Path, star: str) -> str | None:
	"""Return a Teklu catalog name matching one of the star's SIMBAD identifiers."""
	database_ids = {}
	with database.open(encoding="ascii") as data_file:
		for line in data_file:
			name, simbad_name = teklu_names(line)
			for identifier in (name, simbad_name):
				if identifier:
					database_ids[normalize_identifier(identifier)] = name

	record = simbad_record(star)
	if record is None:
		return None
	for identifier in [record["main_id"], *record["ids"]]:
		match = database_ids.get(normalize_identifier(identifier))
		if match is not None:
			return match
	return None


# ---------------------------------------------------------------------------
# RV readers
# ---------------------------------------------------------------------------

def _first_three_floats(fields: list[str]) -> tuple[float, float, float] | None:
	"""Parse the leading time, velocity, and error columns of a data row."""
	try:
		time, velocity, error = map(float, fields[:3])
	except ValueError:
		return None
	return time, velocity, error


def _rv_arrays(rows: list[tuple[float, float, float]]) -> RVData:
	"""Discard pre-1988 points and return time-sorted RV arrays."""
	data = np.asarray(rows, dtype=float).reshape(-1, 3)
	data = data[data[:, 0] >= MINIMUM_BJD]
	data = data[np.argsort(data[:, 0])]
	return data[:, 0], data[:, 1], data[:, 2]


def _read_teklu_rows(database: Path, star: str) -> list[tuple[float, float, float]]:
	target = normalize_identifier(star)
	rows = []
	with database.open(encoding="ascii") as data_file:
		for line in data_file:
			if target not in map(normalize_identifier, teklu_names(line)):
				continue
			# The ReadMe says km/s, but the catalog RV values are in m/s.
			# The byte ranges follow the tablea1_Teklu.dat description in ReadMe;
			# missing values are written as "-".
			row = _first_three_floats([line[82:95], line[132:148], line[149:156]])
			if row is not None:
				rows.append(row)
	return rows


def read_teklu_rvs(database: Path, star: str) -> RVData:
	"""Return Teklu BJD, NZP-corrected RV, and RV error, falling back to a SIMBAD alias."""
	rows = _read_teklu_rows(database, star)
	if not rows:
		alias = find_teklu_id(database, star)
		if alias is None:
			print(f"{star}: no matching Teklu RV data found")
		else:
			print(f"{star}: SIMBAD identifier found in Teklu database as {alias}")
			rows = _read_teklu_rows(database, alias)
	return _rv_arrays(rows)


def _tbl_header_value(text: str, field: str) -> str | None:
	"""Return one (optionally quoted) ExoArchive header value."""
	match = re.search(rf"^\\{field}\s*=\s*[\"']?([^\"'\n]+)", text, re.MULTILINE)
	return match.group(1).strip() if match is not None else None


def _read_tbl_rows(text: str) -> list[tuple[float, float, float]]:
	"""Return the (time, velocity, error) rows of an ExoArchive-style table."""
	rows = []
	for line in text.splitlines():
		if not line.strip() or line.lstrip().startswith(("\\", "|")):
			continue
		row = _first_three_floats(line.split())
		if row is not None:
			rows.append(row)
	return rows


def _matching_tbl_texts(database: Path, star: str) -> Iterator[str]:
	"""Yield the text of each ``.tbl`` file whose STAR_ID matches ``star``."""
	target = normalize_identifier(star)
	for path in sorted(database.glob("*.tbl")):
		text = path.read_text(encoding="utf-8")
		star_id = tbl_star_id(text)
		if star_id is not None and normalize_identifier(star_id) == target:
			yield text


@cache
def exoarchive_location(site: str | None) -> EarthLocation | None:
	"""Resolve common ExoArchive observatory labels to Earth locations."""
	if not site:
		return None
	name = site.casefold()
	for fragment, location in OBSERVATORY_COORDINATES.items():
		if fragment in name:
			return location
	site_name = next(
		(location_name for fragment, location_name in OBSERVATORY_ALIASES if fragment in name),
		site,
	)
	try:
		return EarthLocation.of_site(site_name)
	except Exception:
		return None


def read_exoarchive_rvs(database: Path, star: str) -> RVData:
	"""Return ExoArchive RV points in Teklu's systemic-velocity convention."""
	rows = []
	skipped_time_frame = 0
	skipped_site = 0
	target_coordinates = None

	for text in _matching_tbl_texts(database, star):
		velocity_definition = _tbl_header_value(text, "COLUMN_RADIAL_VELOCITY")
		if velocity_definition is None:
			continue
		already_barycentric = "relative to barycenter" in velocity_definition.casefold()

		date_units = _tbl_header_value(text, "DATE_UNITS")
		time_frame = (_tbl_header_value(text, "TIME_REFERENCE_FRAME") or "").upper()
		if (date_units or "").casefold() != "days" or time_frame not in EXOARCHIVE_TIME_FRAMES:
			skipped_time_frame += 1
			continue

		file_rows = _read_tbl_rows(text)
		if not file_rows:
			continue
		times, velocities, errors = np.asarray(file_rows).T
		if time_frame == "MJD":
			times = np.where(times < 1_000_000, times + 2_400_000.5, times)

		if not already_barycentric:
			location = exoarchive_location(_tbl_header_value(text, "OBSERVATORY_SITE"))
			if location is None:
				skipped_site += 1
				continue
			if target_coordinates is None:
				record = simbad_record(star)
				if record is None:
					raise ValueError(f"Could not resolve coordinates for ExoArchive target {star!r}")
				target_coordinates = SkyCoord(record["ra_deg"], record["dec_deg"], unit=u.deg)
			observation_times = Time(
				times, format="jd", scale="utc" if time_frame.endswith("UTC") else "tdb"
			)
			velocities = velocities + target_coordinates.radial_velocity_correction(
				obstime=observation_times, location=location
			).to_value(u.m / u.s)

		# Teklu reports velocities relative to the host systemic velocity.
		velocities = velocities - np.median(velocities)
		rows.extend(zip(times, velocities, errors))

	if skipped_time_frame or skipped_site:
		print(
			f"{star}: skipped {skipped_time_frame} ExoArchive file(s)"
			f" with unsupported date frames and {skipped_site} with unknown sites"
		)
	return _rv_arrays(rows)


def read_synthetic_rvs(database: Path, star: str) -> RVData:
	"""Return synthetic ExoArchive-style RV points for ``star``."""
	rows = []
	for text in _matching_tbl_texts(database, star):
		rows.extend(_read_tbl_rows(text))
	return _rv_arrays(rows)


def read_fulton_rvs(database: Path, star: str) -> RVData:
	"""Return Fulton RV points for ``star``."""
	target = normalize_identifier(star)
	rows = []
	for host, path in fulton_hosts(database):
		if normalize_identifier(host) != target:
			continue
		with path.open(encoding="utf-8", newline="") as data_file:
			data_file.readline()
			for fields in csv.reader(data_file):
				row = _first_three_floats(fields)
				if row is not None:
					time, velocity, error = row
					rows.append((time + 2_440_000, velocity, error))
	return _rv_arrays(rows)


def read_hebrard_rvs(database: Path, star: str) -> RVData:
	"""Return Hebrard RV points in Teklu's systemic-relative m/s convention."""
	target = normalize_identifier(star)
	rows = []
	with database.open(encoding="ascii") as data_file:
		for line in data_file:
			fields = line.split()
			if len(fields) < 6 or normalize_identifier(fields[4]) != target:
				continue
			row = _first_three_floats(fields)
			if row is not None:
				bjd_minus_2400000, velocity_km_s, error_km_s = row
				rows.append((bjd_minus_2400000 + 2_400_000, velocity_km_s, error_km_s))

	if not rows:
		return _rv_arrays(rows)
	record = simbad_record(star)
	if record is None or record["rv_km_s"] is None:
		raise ValueError(f"No SIMBAD systemic radial velocity found for Hebrard target {star!r}")
	return _rv_arrays([
		(time, (velocity - record["rv_km_s"]) * 1_000, error * 1_000)
		for time, velocity, error in rows
	])


def read_harps_rvs(database: Path, star: str) -> RVData:
	"""Return corrected HARPS SERVAL/NZP RV points in m/s for ``star``."""
	target = normalize_identifier(star)
	rows = []
	with database.open(encoding="ascii") as data_file:
		for line in data_file:
			if normalize_identifier(line[HARPS_NAME]) != target:
				continue
			row = _first_three_floats([line[15:28], line[29:41], line[42:54]])
			if row is not None and -9_999_999 not in row[1:]:
				rows.append(row)
	return _rv_arrays(rows)


# ---------------------------------------------------------------------------
# Combined loading
# ---------------------------------------------------------------------------

READERS = {
	"Teklu": (read_teklu_rvs, TEKLU_DATABASE),
	"ExoArchive": (read_exoarchive_rvs, EXOARCHIVE_DATABASE),
	"Fulton": (read_fulton_rvs, FULTON_DATABASE),
	"Hebrard": (read_hebrard_rvs, HEBRARD_DATABASE),
	"HARPS": (read_harps_rvs, HARPS_DATABASE),
	"Synthetic": (read_synthetic_rvs, SYNTHETICS_DATABASE),
}


def load_datasets(
	star: str, sources: Iterable[str], teklu_database: Path = TEKLU_DATABASE
) -> list[tuple[str, RVData]]:
	"""Load each requested source for ``star``, keeping only non-empty datasets.

	HARPS uses the same compact names as Teklu, so it is retried under the Teklu
	alias when the direct name has no HARPS data.
	"""
	datasets = []
	for source in sources:
		reader, database = READERS[source]
		if source == "Teklu":
			database = teklu_database
		data = reader(database, star)
		if source == "HARPS" and not len(data[0]) and teklu_database.is_file():
			alias = find_teklu_id(teklu_database, star)
			if alias is not None and normalize_identifier(alias) != normalize_identifier(star):
				data = reader(database, alias)
		if len(data[0]):
			datasets.append((source, data))
	return datasets


def combine_rv_data(
	datasets: list[tuple[str, RVData]],
	duplicate_tolerance: float = 300 / 86_400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Combine source datasets, dropping observations sharing a timestamp."""
	if not datasets:
		raise ValueError("No RV measurements found in any database")
	times, velocities, errors = (
		np.concatenate([data[column] for _, data in datasets]) for column in range(3)
	)
	labels = np.concatenate(
		[np.full(len(data[0]), source, dtype=object) for source, data in datasets]
	)
	order = np.argsort(times, kind="stable")
	keep = []
	for index in order:
		if not keep or times[index] - times[keep[-1]] > duplicate_tolerance:
			keep.append(index)
	return times[keep], velocities[keep], errors[keep], labels[keep]
