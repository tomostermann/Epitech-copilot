"""Étapes 4-5 : indexe le dossier EPITECH dans ChromaDB, en incrémental.

    uv run python scripts/build_index.py                         # tout, incrémental
    uv run python scripts/build_index.py BACH-2-ANNEE/STARTTREK  # un seul projet
    uv run python scripts/build_index.py --full                  # tout reconstruire
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag
from rag.store import VectorStore


def show_list(title: str, items: list[str], limit: int = 10) -> None:
    if not items:
        return
    print(f"\n{title} ({len(items)}) :")
    for p in items[:limit]:
        print(f"  {p}")
    if len(items) > limit:
        print(f"  ... et {len(items) - limit} autres")


def show_duplicates(cfg) -> None:
    by_hash = defaultdict(list)
    for path, meta in VectorStore(cfg).indexed_files().items():
        by_hash[meta["hash"]].append(path)
    groups = [sorted(p) for p in by_hash.values() if len(p) > 1]
    if groups:
        print(f"\nFichiers identiques indexés plusieurs fois ({len(groups)} groupes) :")
        for g in groups[:5]:
            print("  = " + "\n    ".join(g))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="sous-dossier à indexer (relatif à docs.root)")
    ap.add_argument("--full", action="store_true", help="tout reconstruire")
    args = ap.parse_args()

    cfg = rag.load_config()
    target = args.path or "tout le dossier"
    print(f"Indexation de {target}  (modèle {cfg.embed_model})")
    start = time.time()

    def progress(done: int, total: int) -> None:
        rate = done / (time.time() - start)
        bar = "#" * (30 * done // total)
        print(f"\r  [{bar:<30}] {done}/{total} morceaux  ({rate:.0f}/s)", end="", flush=True)

    r = rag.index(args.path, config=cfg, full=args.full, progress=progress)

    if r.rebuilt:
        print(f"\nBase reconstruite : {r.rebuilt}")
    show_list("Nouveaux", r.added)
    show_list("Modifiés", r.modified)
    show_list("Supprimés", r.removed)
    print(f"\n{len(r.added)} nouveaux, {len(r.modified)} modifiés, "
          f"{len(r.removed)} supprimés, {r.unchanged} inchangés "
          f"-> {r.chunks} morceaux vectorisés en {time.time() - start:.1f} s")
    show_duplicates(cfg)


if __name__ == "__main__":
    main()
