"""Readers, host lookups, and SIMBAD caching for the local RV databases."""

import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astroquery.simbad import Simbad


DATABASE_ROOT = Path(__file__).resolve().parent / "RVdatabases"
TEKLU_DATABASE = DATABASE_ROOT / "tablea1_Teklu.dat"
EXOARCHIVE_DATABASE = DATABASE_ROOT / "exoarchive"
CLS_DATABASE = DATABASE_ROOT / "CLS" / "table6.dat.gz"
RVBANK_DATABASE = DATABASE_ROOT / "HARPS_RVBank2" / "table4.dat.gz"
RVBANK_TARGETS = DATABASE_ROOT / "HARPS_RVBank2" / "table1.dat.gz"
# The 2020 RVBank release keeps ~3% of points (e.g. CoRoT hosts) absent from v2.
RVBANK2020_DATABASE = DATABASE_ROOT / "HARPS" / "rvbank.dat"
SOPHIE_DATABASE = DATABASE_ROOT / "SOPHIE"
NEID_DATABASE = DATABASE_ROOT / "NEID" / "neid_l2.csv"
HEBRARD_DATABASE = DATABASE_ROOT / "Hebrard2016" / "rvdata.dat"
SYNTHETICS_DATABASE = DATABASE_ROOT / "Synthetics"
SIMBAD_CACHE = DATABASE_ROOT / "simbad_cache.json"
MINIMUM_BJD = 2_447_161.5  # 1988-01-01 00:00 UTC
SIMBAD_BATCH_SIZE = 500

# Publication year of each source's analysis. When two sources contain the same
# exposure, the more recent analysis is kept (see combine_rv_data). ExoArchive
# files use the year in their REFERENCE header; SOPHIE archive pipeline RVs use
# their observation year.
ANALYSIS_YEAR = {
	"Teklu": 2025, "NEID": 2024, "RVBank": 2024, "CLS": 2021, "RVBank2020": 2020, "Hebrard": 2016,
	"Synthetic": 9999,
}

# Two same-instrument points from different datasets closer than this are one
# exposure. Hamilton releases timestamp its long exposures differently (start vs
# midpoint, HJD vs BJD), so their copies differ by up to ~30 minutes.
DUPLICATE_TOLERANCE_DAYS = 300 / 86_400
INSTRUMENT_DUPLICATE_TOLERANCE_DAYS = {"Hamilton": 1_800 / 86_400}

# Instrument upgrades that introduce RV zero-point offsets, fitted separately.
HARPS_FIBRE_UPGRADE_BJD = 2_457_174.5  # 2015-06-03
SOPHIE_PLUS_UPGRADE_BJD = 2_455_730.5  # 2011-06-14
NEID_CONTRERAS_FIRE_BJD = 2_459_745.5  # 2022-06-15; NEID resumed in late 2023
# SOPHIE archive errors are photon noise only; add the instrumental floor in
# quadrature (~5 m/s before the SOPHIE+ fibre upgrade, ~1.5 m/s after).
SOPHIE_ERROR_FLOOR_M_S = {"SOPHIE": 5.0, "SOPHIE+": 1.5}

# Fixed-width name columns of the Teklu catalog.
TEKLU_NAME = slice(0, 14)
TEKLU_SIMBAD_NAME = slice(15, 45)

CLS_INSTRUMENTS = {
	# CLS code: (fit label, instrument)
	"k": ("CLS HIRES-k", "HIRES"),
	"j": ("CLS HIRES-j", "HIRES"),
	"apf": ("CLS APF", "APF"),
	"lick": ("CLS Lick", "Hamilton"),
}
# Substrings identifying the spectrograph in free-text instrument labels,
# checked in order. Used only to recognise the same exposure in two sources.
INSTRUMENT_PATTERNS = (
	("harps-n", "HARPS-N"),
	("harps", "HARPS"),
	("hires", "HIRES"),
	("hamilton", "Hamilton"),
	("hamlton", "Hamilton"),
	("levy", "APF"),
	("apf", "APF"),
	("sophie", "SOPHIE"),
	("elodie", "ELODIE"),
	("coralie", "CORALIE"),
	("ucles", "UCLES"),
	("2d coude", "Tull"),
	("2dcs", "Tull"),
	("tull", "Tull"),
	("hides", "HIDES"),
	("mike", "MIKE"),
	("fies", "FIES"),
	("sandiford", "Sandiford"),
)

EXOARCHIVE_TIME_FRAMES = {
	# "HJD-TBD" is a typo present in some archive files.
	"BJD", "BJD-UTC", "BJD-TDB", "JD", "JD-UTC", "HJD", "HJD-UTC", "HJD-TBD", "FCJD", "MJD",
}
class RVData(NamedTuple):
	"""RV points for one star; every field is an array with one entry per point."""

	time: np.ndarray  # BJD
	rv: np.ndarray  # m/s
	error: np.ndarray  # m/s
	label: np.ndarray  # zero-point group, fitted with its own offset
	instrument: np.ndarray  # spectrograph, to recognise one exposure in two sources
	year: np.ndarray  # analysis year; the newer analysis wins duplicates
	dataset: np.ndarray  # table or file; points within one dataset are never merged


Row = tuple[float, float, float]


def normalize_identifier(identifier: str) -> str:
	"""Normalize catalog identifiers for case- and whitespace-insensitive matching."""
	return "".join(identifier.casefold().split())


def catalog_key(identifier: str) -> str:
	"""Normalize an identifier and the zero-padding/alias quirks of archive names.

	For example ``HD004614`` and ``HD 4614`` or ``gl436`` and ``GJ 436`` share a key.
	"""
	key = re.sub(r"^gl(?=\d)", "gj", normalize_identifier(identifier))
	return re.sub(r"(?<=[a-z])0+(?=\d)", "", key)


def instrument_family(name: str) -> str:
	"""Map a free-text instrument label to a spectrograph name."""
	text = name.casefold()
	for fragment, family in INSTRUMENT_PATTERNS:
		if fragment in text:
			return family
	if re.match(r"^[pl]:", text):  # Howard & Fulton (2016) Hamilton dewar codes
		return "Hamilton"
	return name.strip().upper()


# ---------------------------------------------------------------------------
# Host identifiers
# ---------------------------------------------------------------------------

def tbl_star_id(text: str) -> str | None:
	"""Return the STAR_ID from an ExoArchive-style ``.tbl`` file."""
	match = re.search(r"^\\STAR_ID\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
	return match.group(1).strip() if match else None


def teklu_names(line: str) -> tuple[str, str]:
	"""Return the catalog and SIMBAD names from a Teklu table row."""
	return line[TEKLU_NAME].strip(), line[TEKLU_SIMBAD_NAME].strip()


def cls_name(cps_id: str) -> str:
	"""Return a SIMBAD-resolvable name for an abbreviated CPS identifier."""
	return f"HD {cps_id}" if cps_id.isdigit() else cps_id


def exoarchive_hosts(directory: Path = EXOARCHIVE_DATABASE) -> Iterator[tuple[str, Path]]:
	"""Yield (host, path) for each ExoArchive table with a STAR_ID."""
	for path in sorted(directory.glob("*.tbl")):
		host = tbl_star_id(path.read_text(encoding="utf-8", errors="replace"))
		if host:
			yield host, path


def database_hosts() -> set[str]:
	"""Collect host identifiers from every supported local RV database."""
	identifiers = {host for host, _ in exoarchive_hosts()}
	if TEKLU_DATABASE.is_file():
		with TEKLU_DATABASE.open(encoding="ascii") as data_file:
			for line in data_file:
				identifiers.update(teklu_names(line))
	identifiers.update(cls_name(star) for star in _cls_index())
	if RVBANK_TARGETS.is_file():
		with gzip.open(RVBANK_TARGETS, "rt", encoding="ascii") as data_file:
			identifiers.update(line[0:14].strip() for line in data_file)
	if RVBANK2020_DATABASE.is_file():
		with RVBANK2020_DATABASE.open(encoding="ascii") as data_file:
			identifiers.update(line[0:14].strip() for line in data_file)
	identifiers.update(name for name, _ in _sophie_index().values())
	identifiers.update(_hebrard_index())
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
	try:
		simbad = Simbad()
		simbad.add_votable_fields("ids", "rvz_radvel")  # contacts the server
	except Exception as error:
		print(f"Warning: SIMBAD unavailable: {error}", file=sys.stderr)
		return records
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


@cache
def identifier_keys(star: str) -> frozenset[str]:
	"""Return catalog keys for ``star`` and all of its SIMBAD identifiers."""
	record = simbad_record(star)
	names = [star, record["main_id"], *record["ids"]] if record else [star]
	return frozenset(catalog_key(name) for name in names)


def _matches(name: str, star: str) -> bool:
	return bool(name) and catalog_key(name) in identifier_keys(star)


# ---------------------------------------------------------------------------
# RV readers
# ---------------------------------------------------------------------------

def _first_three_floats(fields: list[str]) -> Row | None:
	"""Parse the leading time, velocity, and error columns of a data row."""
	try:
		time, velocity, error = map(float, fields[:3])
	except ValueError:
		return None
	# Some archive tables write missing values as NaN; zero errors would get infinite weight.
	if not np.isfinite([time, velocity, error]).all() or error <= 0:
		return None
	return time, velocity, error


def _chunk(
	rows: list[Row], label: str, instrument: str, year: int, dataset: str
) -> RVData:
	"""Build RV data for rows sharing one label, instrument, analysis, and dataset."""
	values = np.asarray(rows, dtype=float).reshape(-1, 3)
	count = len(values)
	return RVData(
		values[:, 0],
		values[:, 1],
		values[:, 2],
		np.full(count, label, dtype=object),
		np.full(count, instrument, dtype=object),
		np.full(count, year),
		np.full(count, dataset, dtype=object),
	)


def _merge(chunks: list[RVData]) -> RVData:
	"""Concatenate chunks, discard pre-1988 points, and sort by time."""
	if not chunks:
		return _chunk([], "", "", 0, "")
	merged = RVData(*(np.concatenate(columns) for columns in zip(*chunks)))
	keep = np.flatnonzero(merged.time >= MINIMUM_BJD)
	keep = keep[np.argsort(merged.time[keep], kind="stable")]
	return RVData(*(column[keep] for column in merged))


def _split_at(
	rows: list[Row], split_bjd: float, labels: tuple[str, str], **metadata
) -> list[RVData]:
	"""Return chunks for rows before and after an instrument upgrade."""
	before = [row for row in rows if row[0] < split_bjd]
	after = [row for row in rows if row[0] >= split_bjd]
	return [
		_chunk(part, label, **metadata)
		for part, label in zip((before, after), labels)
		if part
	]


def _relative_to_median(chunks: list[RVData]) -> list[RVData]:
	"""Subtract each chunk's median RV (Teklu's systemic-relative convention)."""
	return [chunk._replace(rv=chunk.rv - np.median(chunk.rv)) for chunk in chunks]


def read_teklu_rvs(database: Path, star: str) -> RVData:
	"""Return Teklu HIRES BJD, NZP-corrected RV, and RV error."""
	rows = []
	with database.open(encoding="ascii") as data_file:
		for line in data_file:
			if not any(_matches(name, star) for name in teklu_names(line)):
				continue
			# The ReadMe says km/s, but the catalog RV values are in m/s.
			# The byte ranges follow the tablea1_Teklu.dat description in ReadMe;
			# missing values are written as "-".
			row = _first_three_floats([line[82:95], line[132:148], line[149:156]])
			if row is not None:
				rows.append(row)
	return _merge([_chunk(rows, "Teklu", "HIRES", ANALYSIS_YEAR["Teklu"], "Teklu")])


def _tbl_header_value(text: str, field: str) -> str | None:
	"""Return one (optionally quoted) ExoArchive header value."""
	match = re.search(rf"^\\{field}\s*=\s*[\"']?([^\"'\n]+)", text, re.MULTILINE)
	return match.group(1).strip() if match is not None else None


def _read_tbl_rows(text: str) -> list[Row]:
	"""Return the (time, velocity, error) rows of an ExoArchive-style table."""
	rows = []
	for line in text.splitlines():
		if not line.strip() or line.lstrip().startswith(("\\", "|")):
			continue
		row = _first_three_floats(line.split())
		if row is not None:
			rows.append(row)
	return rows


@cache
def _tbl_files(database: Path) -> tuple[tuple[Path, str, str], ...]:
	"""(path, text, STAR_ID) for every ``.tbl`` file in ``database``, read once."""
	files = []
	for path in sorted(database.glob("*.tbl")):
		text = path.read_text(encoding="utf-8")
		star_id = tbl_star_id(text)
		if star_id is not None:
			files.append((path, text, star_id))
	return tuple(files)


def _matching_tbl_files(database: Path, star: str) -> Iterator[tuple[Path, str]]:
	"""Yield (path, text) for each ``.tbl`` file whose STAR_ID matches ``star``."""
	for path, text, star_id in _tbl_files(database):
		if _matches(star_id, star):
			yield path, text


def read_exoarchive_rvs(database: Path, star: str) -> RVData:
	"""Return ExoArchive RV points in Teklu's systemic-velocity convention.

	All local ExoArchive tables hold barycentric velocities (published precise RVs
	are barycentre-corrected even when the header says only "Relative radial
	velocity"), so no barycentric correction is applied.
	"""
	chunks = []
	skipped_time_frame = 0
	for path, text in _matching_tbl_files(database, star):
		date_units = _tbl_header_value(text, "DATE_UNITS")
		time_frame = (_tbl_header_value(text, "TIME_REFERENCE_FRAME") or "").upper()
		if (date_units or "").casefold() != "days" or time_frame not in EXOARCHIVE_TIME_FRAMES:
			skipped_time_frame += 1
			continue
		rows = _read_tbl_rows(text)
		if time_frame == "MJD":
			rows = [(time + 2_400_000.5 if time < 1_000_000 else time, rv, error) for time, rv, error in rows]
		if not rows:
			continue
		reference_year = re.search(r"(?:19|20)\d{2}", _tbl_header_value(text, "REFERENCES?") or "")
		chunks.append(_chunk(
			rows,
			"ExoArchive",
			instrument_family(_tbl_header_value(text, "INSTRUMENT") or ""),
			int(reference_year.group()) if reference_year else 0,
			path.name,
		))
	if skipped_time_frame:
		print(f"{star}: skipped {skipped_time_frame} ExoArchive file(s) with unsupported date frames")
	# Teklu reports velocities relative to the host systemic velocity.
	return _merge(_relative_to_median(chunks))


def read_synthetic_rvs(database: Path, star: str) -> RVData:
	"""Return synthetic ExoArchive-style RV points for ``star``."""
	return _merge([
		_chunk(_read_tbl_rows(text), "Synthetic", "Synthetic", ANALYSIS_YEAR["Synthetic"], path.name)
		for path, text in _matching_tbl_files(database, star)
	])


@cache
def _cls_index(database: Path = CLS_DATABASE) -> dict[str, list[tuple[str, Row]]]:
	"""Return CLS (instrument code, row) lists keyed by CPS identifier."""
	index = defaultdict(list)
	if database.is_file():
		with gzip.open(database, "rt", encoding="ascii") as data_file:
			for line in data_file:
				row = _first_three_floats([line[33:47], line[48:57], line[58:67]])
				if row is not None:
					index[line[7:16].strip()].append((line[17:21].strip(), row))
	return index


def read_cls_rvs(database: Path, star: str) -> RVData:
	"""Return California Legacy Survey RVs, one offset group per instrument."""
	rows_by_code = defaultdict(list)
	for cps_id, rows in _cls_index(database).items():
		if _matches(cls_name(cps_id), star):
			for code, row in rows:
				rows_by_code[code].append(row)
	return _merge([
		_chunk(rows, *CLS_INSTRUMENTS[code], ANALYSIS_YEAR["CLS"], "CLS")
		for code, rows in rows_by_code.items()
	])


@cache
def _rvbank_index(database: Path = RVBANK_DATABASE) -> dict[str, list[Row]]:
	"""Return reliable (flag 0) HARPS RVBank v2 rows keyed by catalog key."""
	index = defaultdict(list)
	if database.is_file():
		with gzip.open(database, "rt", encoding="ascii") as data_file:
			for line in data_file:
				# Byte ranges from the ReadMe: BJD, RV_mlc_nzp, e_RV_mlc_nzp, SERVAL flag.
				row = _first_three_floats([line[58:73], line[74:87], line[88:98]])
				if row is not None and line[317:322].strip() in ("0", "0.0"):
					index[catalog_key(line[0:14])].append(row)
	return index


def read_rvbank_rvs(database: Path, star: str) -> RVData:
	"""Return NZP-corrected HARPS SERVAL RVs, split at the 2015 fibre upgrade."""
	rows = [
		row
		for key, key_rows in _rvbank_index(database).items()
		if key in identifier_keys(star)
		for row in key_rows
	]
	return _merge(_split_at(
		rows,
		HARPS_FIBRE_UPGRADE_BJD,
		("HARPS-pre", "HARPS-post"),
		instrument="HARPS",
		year=ANALYSIS_YEAR["RVBank"],
		dataset="RVBank",
	))


@cache
def _rvbank2020_index(database: Path = RVBANK2020_DATABASE) -> dict[str, list[Row]]:
	"""Return HARPS RVBank (2020) rows keyed by catalog key; -9999999 marks missing."""
	index = defaultdict(list)
	if database.is_file():
		with database.open(encoding="ascii") as data_file:
			for line in data_file:
				row = _first_three_floats([line[15:28], line[29:41], line[42:54]])
				if row is not None and -9_999_999 not in row[1:]:
					index[catalog_key(line[0:14])].append(row)
	return index


def read_rvbank2020_rvs(database: Path, star: str) -> RVData:
	"""Return the 2020 HARPS RVBank release, split at the 2015 fibre upgrade."""
	rows = [
		row
		for key, key_rows in _rvbank2020_index(database).items()
		if key in identifier_keys(star)
		for row in key_rows
	]
	return _merge(_split_at(
		rows,
		HARPS_FIBRE_UPGRADE_BJD,
		("HARPS2020-pre", "HARPS2020-post"),
		instrument="HARPS",
		year=ANALYSIS_YEAR["RVBank2020"],
		dataset="RVBank2020",
	))


@cache
def _sophie_index(directory: Path = SOPHIE_DATABASE) -> dict[str, tuple[str, list[tuple]]]:
	"""Return (archive name, CCF rows) keyed by catalog key for the SOPHIE archive.

	Rows are (seq, bjd, mask, ccf_offline, rv_km_s, err_km_s); 999 marks missing RVs.
	"""
	index = {}
	for path in sorted(directory.glob("ccf_ra*.txt")):
		for line in path.read_text(encoding="utf-8").splitlines():
			fields = line.split("\t")
			if line.startswith("#") or len(fields) != 7 or fields[0] == "seq":
				continue
			try:
				seq, bjd = int(fields[0]), float(fields[2])
				velocity, error = float(fields[5]), float(fields[6])
			except ValueError:
				continue
			if 999 in (velocity, error):
				continue
			name = fields[1].strip()
			index.setdefault(catalog_key(name), (name, []))[1].append(
				(seq, bjd, fields[3].strip(), fields[4].strip(), velocity, error)
			)
	return index


def _without_gross_outliers(chunk: RVData, threshold: float = 10.0) -> RVData:
	"""Drop points more than ``threshold`` robust sigmas (1.4826 MAD) from the median.

	Pipeline archives include failed CCF fits that are off by km/s; planetary
	signals stay far inside this cut.
	"""
	deviation = np.abs(chunk.rv - np.median(chunk.rv))
	scale = 1.4826 * np.median(deviation)
	return RVData(*(column[deviation <= threshold * scale] for column in chunk))


def read_sophie_rvs(database: Path, star: str) -> RVData:
	"""Return SOPHIE archive pipeline RVs, split at the 2011 SOPHIE+ upgrade.

	Each exposure can have CCFs for several masks. Only the star's most common mask
	is used, an offline (reprocessed) CCF is preferred over the online one, and
	failed CCF fits are removed as gross outliers.
	"""
	ccfs = [
		ccf
		for key, (_, key_ccfs) in _sophie_index(database).items()
		if key in identifier_keys(star)
		for ccf in key_ccfs
	]
	if not ccfs:
		return _merge([])
	mask = Counter(ccf[2] for ccf in ccfs).most_common(1)[0][0]
	by_exposure = {}
	for seq, bjd, ccf_mask, offline, velocity, error in sorted(ccfs, key=lambda ccf: ccf[3]):
		if ccf_mask == mask:
			by_exposure[seq] = (bjd, velocity * 1_000, error * 1_000)
	chunks = []
	for (label, before, after) in (
		("SOPHIE", -np.inf, SOPHIE_PLUS_UPGRADE_BJD),
		("SOPHIE+", SOPHIE_PLUS_UPGRADE_BJD, np.inf),
	):
		floor = SOPHIE_ERROR_FLOOR_M_S[label]
		rows = [
			(bjd, velocity, np.hypot(error, floor))
			for bjd, velocity, error in by_exposure.values()
			if before <= bjd < after
		]
		if not rows:
			continue
		chunk = _without_gross_outliers(_chunk(rows, label, "SOPHIE", 0, "SOPHIE archive"))
		# Pipeline RVs are as recent as the observation itself.
		years = np.floor(2000 + (chunk.time - 2_451_544.5) / 365.25).astype(int)
		chunks.append(chunk._replace(year=years))
	return _merge(_relative_to_median(chunks))


def read_neid_rvs(database: Path, star: str) -> RVData:
	"""Return NEID L2 CCF RVs (barycentric, km/s in the archive table), split at the
	2022 Contreras-fire shutdown, relative to each group's median."""
	if not database.is_file():
		return _merge([])
	with database.open() as data_file:
		rows = []
		for row in csv.DictReader(data_file):
			if catalog_key(row["target"]) not in identifier_keys(star) or row["obsmode"].lower() != "hr":
				continue
			parsed = _first_three_floats([row["ccfjdsum"], row["ccfrvmod"], row["dvrms"]])
			if parsed is not None:
				rows.append((parsed[0], parsed[1] * 1_000, parsed[2] * 1_000))
	return _merge(_relative_to_median(_split_at(
		rows, NEID_CONTRERAS_FIRE_BJD, ("NEID-pre", "NEID-post"),
		instrument="NEID", year=ANALYSIS_YEAR["NEID"], dataset="NEID",
	)))


@cache
def _hebrard_index(database: Path = HEBRARD_DATABASE) -> dict[str, list[Row]]:
	index = defaultdict(list)
	if database.is_file():
		with database.open(encoding="ascii") as data_file:
			for line in data_file:
				fields = line.split()
				row = _first_three_floats(fields) if len(fields) >= 6 else None
				if row is not None:
					index[fields[4]].append(row)
	return index


def read_hebrard_rvs(database: Path, star: str) -> RVData:
	"""Return Hebrard RV points in Teklu's systemic-relative m/s convention."""
	rows = [
		(bjd_minus_2400000 + 2_400_000, velocity_km_s, error_km_s)
		for name, name_rows in _hebrard_index(database).items()
		if _matches(name, star)
		for bjd_minus_2400000, velocity_km_s, error_km_s in name_rows
	]
	if not rows:
		return _merge([])
	record = simbad_record(star)
	if record is None or record["rv_km_s"] is None:
		raise ValueError(f"No SIMBAD systemic radial velocity found for Hebrard target {star!r}")
	rows = [
		(time, (velocity - record["rv_km_s"]) * 1_000, error * 1_000)
		for time, velocity, error in rows
	]
	return _merge([_chunk(rows, "Hebrard", "SOPHIE", ANALYSIS_YEAR["Hebrard"], "Hebrard")])


# ---------------------------------------------------------------------------
# Combined loading
# ---------------------------------------------------------------------------

READERS = {
	"Teklu": (read_teklu_rvs, TEKLU_DATABASE),
	"ExoArchive": (read_exoarchive_rvs, EXOARCHIVE_DATABASE),
	"CLS": (read_cls_rvs, CLS_DATABASE),
	"HARPS": (read_rvbank_rvs, RVBANK_DATABASE),
	"HARPS2020": (read_rvbank2020_rvs, RVBANK2020_DATABASE),
	"SOPHIE": (read_sophie_rvs, SOPHIE_DATABASE),
	"NEID": (read_neid_rvs, NEID_DATABASE),
	"Hebrard": (read_hebrard_rvs, HEBRARD_DATABASE),
	"Synthetic": (read_synthetic_rvs, SYNTHETICS_DATABASE),
}


def load_datasets(
	star: str, sources: Iterable[str], teklu_database: Path = TEKLU_DATABASE
) -> list[RVData]:
	"""Load each requested source for ``star``, keeping only non-empty datasets."""
	datasets = []
	for source in sources:
		reader, database = READERS[source]
		if source == "Teklu":
			database = teklu_database
		data = reader(database, star)
		if len(data.time):
			datasets.append(data)
	return datasets


def combine_rv_data(
	datasets: list[RVData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Combine datasets, keeping one copy of exposures that appear in several.

	Datasets are visited from the most recent analysis to the oldest (ties going
	to the dataset listed first). Each point is matched one-to-one, closest pairs
	first, to an already-kept point from another dataset with the same instrument
	within the instrument's duplicate tolerance; matched points are dropped as
	older copies of that exposure. Points within a single dataset are never merged.
	"""
	if not datasets:
		raise ValueError("No RV measurements found in any database")
	data = RVData(*(np.concatenate(columns) for columns in zip(*datasets)))
	source_rank = np.concatenate([np.full(len(d.time), rank) for rank, d in enumerate(datasets)])

	groups: dict[tuple, list[int]] = defaultdict(list)
	for index, key in enumerate(zip(data.instrument, data.dataset, data.year)):
		groups[key].append(index)
	kept = np.zeros(len(data.time), dtype=bool)
	kept_by_instrument: dict[str, list[int]] = defaultdict(list)
	absorbed_datasets: dict[int, set[str]] = defaultdict(set)  # kept index -> matched datasets
	for key in sorted(groups, key=lambda key: (-key[2], source_rank[groups[key][0]])):
		instrument, dataset, _ = key
		tolerance = INSTRUMENT_DUPLICATE_TOLERANCE_DAYS.get(instrument, DUPLICATE_TOLERANCE_DAYS)
		pool = np.array([
			index for index in kept_by_instrument[instrument]
			if data.dataset[index] != dataset and dataset not in absorbed_datasets[index]
		], dtype=int)
		pool = pool[np.argsort(data.time[pool])]
		pool_times = data.time[pool]
		pairs = []
		for candidate in groups[key]:
			time = data.time[candidate]
			low = np.searchsorted(pool_times, time - tolerance, side="left")
			high = np.searchsorted(pool_times, time + tolerance, side="right")
			pairs.extend((abs(pool_times[j] - time), candidate, pool[j]) for j in range(low, high))
		duplicates = set()
		for _, candidate, match in sorted(pairs):
			if candidate not in duplicates and dataset not in absorbed_datasets[match]:
				duplicates.add(candidate)
				absorbed_datasets[match].add(dataset)
		for candidate in groups[key]:
			if candidate not in duplicates:
				kept[candidate] = True
				kept_by_instrument[instrument].append(candidate)

	keep = np.flatnonzero(kept)
	keep = keep[np.argsort(data.time[keep], kind="stable")]
	return data.time[keep], data.rv[keep], data.error[keep], data.label[keep]
