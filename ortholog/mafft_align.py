#!/usr/bin/env python3
"""Run MAFFT on per-gene ortholog FASTA files and summarize pairwise identity."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path

from Bio import AlignIO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Align ortholog protein FASTA files with MAFFT and compute pairwise identity."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "output" / "align_input",
        help="Directory with per-gene FASTA files from extract.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "alignments",
        help="Directory for MAFFT alignment outputs.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=root / "output" / "pairwise_identity.csv",
        help="CSV file for pairwise identity summary.",
    )
    parser.add_argument(
        "--mafft-bin",
        type=str,
        default="mafft",
        help="MAFFT executable name or full path.",
    )
    parser.add_argument(
        "--mafft-mode",
        type=str,
        default="--auto",
        help="MAFFT option passed before input file, e.g. --auto or --localpair.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.fa",
        help="Input FASTA glob pattern under --input-dir.",
    )
    return parser.parse_args()


def check_mafft(mafft_bin: str) -> None:
    resolved = shutil.which(mafft_bin)
    if resolved is None:
        raise FileNotFoundError(
            f"MAFFT not found: {mafft_bin}\n"
            "Install with: conda install -c bioconda mafft"
        )
    version = subprocess.run(
        [resolved, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    version_text = (version.stdout or version.stderr).strip().splitlines()
    if version_text:
        print(f"Using MAFFT: {version_text[0]}")


def run_mafft(
    input_fasta: Path,
    output_fasta: Path,
    mafft_bin: str,
    mafft_mode: str,
) -> None:
    command = [mafft_bin, mafft_mode, str(input_fasta)]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MAFFT failed for {input_fasta}:\n{result.stderr.strip()}"
        )
    output_fasta.write_text(result.stdout, encoding="utf-8")


def parse_species(record_id: str) -> str:
    if "|" in record_id:
        return record_id.rsplit("|", 1)[-1].lower()
    return record_id.lower()


def pairwise_identity(seq_a: str, seq_b: str) -> tuple[int, int, float, float]:
    if len(seq_a) != len(seq_b):
        raise ValueError("Aligned sequences must have equal length.")

    identical = 0
    aligned_non_gap = 0
    for residue_a, residue_b in zip(seq_a, seq_b):
        if residue_a == "-" or residue_b == "-":
            continue
        aligned_non_gap += 1
        if residue_a == residue_b:
            identical += 1

    if aligned_non_gap == 0:
        return 0, 0, 0.0, 0.0

    identity = 100.0 * identical / aligned_non_gap
    coverage = 100.0 * aligned_non_gap / len(seq_a)
    return identical, aligned_non_gap, identity, coverage


def summarize_alignment(alignment_path: Path, gene: str) -> list[dict[str, object]]:
    alignment = AlignIO.read(alignment_path, "fasta")
    records = list(alignment)
    if len(records) < 2:
        return []

    species_to_seq = {parse_species(record.id): str(record.seq) for record in records}
    rows: list[dict[str, object]] = []

    for species_a, species_b in combinations(sorted(species_to_seq), 2):
        identical, aligned_non_gap, identity, coverage = pairwise_identity(
            species_to_seq[species_a],
            species_to_seq[species_b],
        )
        rows.append(
            {
                "gene": gene,
                "species_a": species_a,
                "species_b": species_b,
                "identical_sites": identical,
                "aligned_non_gap_sites": aligned_non_gap,
                "identity_percent": round(identity, 2),
                "coverage_percent": round(coverage, 2),
                "alignment_file": alignment_path.name,
            }
        )
    return rows


def write_summary(rows: list[dict[str, object]], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gene",
        "species_a",
        "species_b",
        "identical_sites",
        "aligned_non_gap_sites",
        "identity_percent",
        "coverage_percent",
        "alignment_file",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    check_mafft(args.mafft_bin)

    input_files = sorted(args.input_dir.glob(args.pattern))
    if not input_files:
        raise FileNotFoundError(
            f"No input FASTA files found in {args.input_dir} ({args.pattern})"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for input_fasta in input_files:
        gene = input_fasta.stem
        output_fasta = args.output_dir / f"{gene}.aln"
        print(f"Aligning {input_fasta.name} -> {output_fasta.name}")
        run_mafft(input_fasta, output_fasta, args.mafft_bin, args.mafft_mode)
        summary_rows.extend(summarize_alignment(output_fasta, gene))

    write_summary(summary_rows, args.summary_csv)
    print(f"Wrote alignments to {args.output_dir}")
    print(f"Wrote summary to {args.summary_csv}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
