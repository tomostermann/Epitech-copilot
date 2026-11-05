"""Étape 3 : observer le découpage.

    uv run python scripts/inspect_chunks.py                      # stats pour plusieurs tailles
    uv run python scripts/inspect_chunks.py --show CHEMIN        # morceaux d'un fichier
    uv run python scripts/inspect_chunks.py --show CHEMIN --size 500 --overlap 100
"""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking import chunk_document
from rag.config import load_config
from rag.loaders import Document, Skipped, load_documents, load_file


def compare_sizes(cfg, docs: list[Document]) -> None:
    print(f"{'taille':>7} {'overlap':>8} {'morceaux':>9} {'moyenne':>8} {'médiane':>8}")
    for size in (300, 500, 1000, 2000):
        overlap = size * 15 // 100
        lengths = [len(c.text) for d in docs for c in chunk_document(d, size, overlap)]
        print(f"{size:>7} {overlap:>8} {len(lengths):>9} "
              f"{statistics.mean(lengths):>8.0f} {statistics.median(lengths):>8.0f}")

    chunks = [c for d in docs for c in chunk_document(d, cfg.chunk_size, cfg.chunk_overlap)]
    per_project = Counter(c.metadata["project"] for c in chunks)
    print(f"\nAvec la config actuelle ({cfg.chunk_size}/{cfg.chunk_overlap}) : "
          f"{len(chunks)} morceaux")
    for project, n in per_project.most_common():
        print(f"  {n:>5}  {project}")


def show(cfg, rel: str, size: int, overlap: int) -> None:
    doc = load_file(cfg.docs_root / rel, cfg)
    if isinstance(doc, Skipped):
        print(f"Ignoré : {doc.reason}")
        return
    chunks = chunk_document(doc, size, overlap)
    print(f"{rel} : {len(doc.text)} caractères -> {len(chunks)} morceaux "
          f"(taille {size}, overlap {overlap})\n")
    for c in chunks:
        page = f", page {c.metadata['page']}" if "page" in c.metadata else ""
        print(f"──── morceau {c.metadata['chunk_index']} "
              f"({len(c.text)} car.{page}) " + "─" * 30)
        print(c.text)
        print()


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--show")
    ap.add_argument("--size", type=int, default=cfg.chunk_size)
    ap.add_argument("--overlap", type=int, default=cfg.chunk_overlap)
    args = ap.parse_args()

    if args.show:
        show(cfg, args.show, args.size, args.overlap)
    else:
        docs = [d for d in load_documents(cfg) if isinstance(d, Document)]
        compare_sizes(cfg, docs)


if __name__ == "__main__":
    main()
