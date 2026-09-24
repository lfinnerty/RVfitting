#!/usr/bin/env python3
"""Create host-star-to-filename lists for the local RV databases."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections.abc import Callable


def exoarchive_host(path: Path) -> str:
    """Read the host-star identifier from an Exoplanet Archive RV file."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\\STAR_ID\s*=\s*[\"']([^\"']+)[\"']", text)
    if match is None:
        raise ValueError(f"Missing STAR_ID metadata in {path}")
    return match.group(1).strip()


def fulton_host(path: Path) -> str:
    """Read the HD host identifier from a Fulton RV CSV file."""
    first_line = path.open(encoding="utf-8").readline()
    match = re.fullmatch(r"# star HD number,\s*(\d+)\s*\n?", first_line)
    if match is None:
        raise ValueError(f"Missing HD number metadata in {path}")
    return f"HD {match.group(1)}"


def write_mapping(
    directory: Path,
    output_path: Path,
    host_reader: Callable[[Path], str],
    file_suffix: str,
) -> None:
    """Write one tab-separated host-star/filename row for each RV file."""
    rows: list[tuple[str, str]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix != file_suffix:
            continue
        host = host_reader(path)
        rows.append((host, path.name))

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("host_star\tfilename\n")
        output.writelines(f"{host}\t{filename}\n" for host, filename in rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write host-star-to-filename lists for the local RV databases."
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path(__file__).resolve().parent / "RVdatabases",
        help="Root directory containing exoarchive and rv_data_fulton.",
    )
    args = parser.parse_args()

    write_mapping(
        args.database_root / "exoarchive",
        args.database_root / "exoarchive_hosts.txt",
        exoarchive_host,
        ".tbl",
    )
    write_mapping(
        args.database_root / "rv_data_fulton",
        args.database_root / "rv_data_fulton_hosts.txt",
        fulton_host,
        ".csv",
    )


if __name__ == "__main__":
    main()
