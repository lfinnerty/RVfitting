#!/usr/bin/env python3
"""List the stellar systems in the local RV databases.

By default, print unique systems using SIMBAD default identifiers. With
``--file-lists``, instead write the ExoArchive host-star-to-filename table.
"""

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import rv_io


def canonical_systems() -> list[str]:
	"""Return unique SIMBAD MAIN_IDs for all local host identifiers, sorted."""
	identifiers = sorted(rv_io.database_hosts(), key=rv_io.normalize_identifier)
	records = rv_io.simbad_records(identifiers)
	canonical_ids = {}
	for identifier in identifiers:
		record = records.get(identifier)
		if record is None:
			print(f"Warning: SIMBAD returned no MAIN_ID for {identifier}", file=sys.stderr)
			continue
		canonical_ids[rv_io.normalize_identifier(record["main_id"])] = record["main_id"]
	return [canonical_ids[key] for key in sorted(canonical_ids)]


def write_mapping(output_path: Path, hosts: Iterator[tuple[str, Path]]) -> None:
	"""Write one tab-separated host-star/filename row for each RV file."""
	rows = [f"{host}\t{path.name}\n" for host, path in hosts]
	with output_path.open("w", encoding="utf-8", newline="\n") as output:
		output.write("host_star\tfilename\n")
		output.writelines(rows)
	print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="List unique RV systems using SIMBAD default identifiers."
	)
	parser.add_argument(
		"--output",
		type=Path,
		help="Optional output text file; otherwise print the list to stdout.",
	)
	parser.add_argument(
		"--file-lists",
		action="store_true",
		help="Write RVdatabases/exoarchive_hosts.txt instead.",
	)
	args = parser.parse_args()

	if args.file_lists:
		write_mapping(
			rv_io.DATABASE_ROOT / "exoarchive_hosts.txt", rv_io.exoarchive_hosts()
		)
		return

	systems = canonical_systems()
	output = "".join(f"{system}\n" for system in systems)
	if args.output is None:
		print(output, end="")
	else:
		args.output.write_text(output, encoding="utf-8")
		print(f"Wrote {len(systems)} unique systems to {args.output}")


if __name__ == "__main__":
	main()
