#!/usr/bin/env python3
"""Extract representative protein sequences from Ensembl pep FASTA files."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extract longest protein per gene_symbol from Ensembl pep FASTA files."
    )
    parser.add_argument(
        "--gene-list",
        type=Path,
        default=root / "gene_lists" / "core_genes.txt",
        help="Gene list file, one symbol per line.",
    )
    parser.add_argument(
        "--pep-dir",
        type=Path,
        default=root / "pep",
        help="Directory containing Ensembl pep.all.fa(.gz) files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output",
        help="Directory for extracted FASTA outputs.",
    )
    parser.add_argument(
        "--dog-fasta",
        type=Path,
        default=None,
        help="Override dog pep FASTA path.",
    )
    parser.add_argument(
        "--human-fasta",
        type=Path,
        default=None,
        help="Override human pep FASTA path.",
    )
    parser.add_argument(
        "--mouse-fasta",
        type=Path,
        default=None,
        help="Override mouse pep FASTA path.",
    )
    return parser.parse_args()


def read_gene_list(path: Path) -> list[str]:
    genes: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower() in {"intersection", "gene", "genes", "symbol"}:
            continue
        genes.append(line)
    if not genes:
        raise ValueError(f"No genes found in {path}")
    return genes


def parse_gene_symbol(description: str) -> str | None:
    for part in description.split():
        if part.startswith("gene_symbol:"):
            return part.split(":", 1)[1].upper()
    return None


def extract_longest_by_symbol(fasta_path: Path, gene_symbols: list[str]) -> dict[str, SeqRecord]:
    targets = {gene.upper() for gene in gene_symbols}
    best: dict[str, tuple[int, SeqRecord]] = {}

    fasta_str = str(fasta_path)
    opener = gzip.open if fasta_str.endswith(".gz") else open
    with opener(fasta_str, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            symbol = parse_gene_symbol(record.description)
            if symbol not in targets:
                continue
            length = len(record.seq)
            if symbol not in best or length > best[symbol][0]:
                best[symbol] = (length, record)

    return {symbol: item[1] for symbol, item in best.items()}


def default_species_fastas(pep_dir: Path, args: argparse.Namespace) -> dict[str, Path]:
    defaults = {
        "dog": pep_dir / "Canis_lupus_familiaris.CanFam3.1.pep.all.fa.gz",
        "human": pep_dir / "Homo_sapiens.GRCh38.pep.all.fa.gz",
        "mouse": pep_dir / "Mus_musculus.GRCm39.pep.all.fa.gz",
    }
    overrides = {
        "dog": args.dog_fasta,
        "human": args.human_fasta,
        "mouse": args.mouse_fasta,
    }
    return {
        species: (override if override is not None else default)
        for species, default in defaults.items()
        for override in [overrides[species]]
    }


def write_species_fasta(
    species: str,
    genes: list[str],
    found: dict[str, SeqRecord],
    output_dir: Path,
) -> dict[str, SeqRecord]:
    out_path = output_dir / f"{species}_core_proteins.fa"
    selected: dict[str, SeqRecord] = {}

    with out_path.open("w", encoding="utf-8") as handle:
        for gene in genes:
            symbol = gene.upper()
            if symbol not in found:
                print(f"MISSING: {species}\t{gene}")
                continue
            record = found[symbol]
            record = SeqRecord(
                record.seq,
                id=f"{gene}|{species}",
                description="",
            )
            selected[gene] = record
            SeqIO.write(record, handle, "fasta")

    print(f"Wrote {out_path} ({len(selected)}/{len(genes)} genes)")
    return selected


def write_gene_ortholog_fastas(
    genes: list[str],
    species_records: dict[str, dict[str, SeqRecord]],
    output_dir: Path,
) -> None:
    align_input_dir = output_dir / "align_input"
    align_input_dir.mkdir(parents=True, exist_ok=True)

    for gene in genes:
        records = [
            species_records[species][gene]
            for species in ("dog", "human", "mouse")
            if gene in species_records.get(species, {})
        ]
        if len(records) < 3:
            missing = [
                species
                for species in ("dog", "human", "mouse")
                if gene not in species_records.get(species, {})
            ]
            print(f"SKIP ortholog FASTA for {gene}, missing in: {', '.join(missing)}")
            continue

        out_path = align_input_dir / f"{gene}.fa"
        SeqIO.write(records, out_path, "fasta")
        print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    genes = read_gene_list(args.gene_list)
    species_fastas = default_species_fastas(args.pep_dir, args)

    species_records: dict[str, dict[str, SeqRecord]] = {}
    for species, fasta_path in species_fastas.items():
        if not fasta_path.exists():
            raise FileNotFoundError(f"Missing FASTA for {species}: {fasta_path}")
        found = extract_longest_by_symbol(fasta_path, genes)
        species_records[species] = write_species_fasta(
            species, genes, found, args.output_dir
        )

    write_gene_ortholog_fastas(genes, species_records, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
