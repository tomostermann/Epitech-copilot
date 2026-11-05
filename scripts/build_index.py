"""Étape 4 : indexe tout le dossier EPITECH dans ChromaDB.

    uv run python scripts/build_index.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_config
from rag.indexer import build_index


def main() -> None:
    cfg = load_config()
    print(f"Indexation de {cfg.docs_root}")
    print(f"  modèle : {cfg.embed_model}   base : {cfg.db_path}\n")
    start = time.time()

    def progress(done: int, total: int) -> None:
        elapsed = time.time() - start
        rate = done / elapsed if elapsed else 0
        bar = "#" * (30 * done // total)
        print(f"\r  [{bar:<30}] {done}/{total} morceaux  ({rate:.0f}/s)", end="", flush=True)

    report = build_index(cfg, progress=progress)
    print(f"\n\n{report.files} fichiers -> {report.chunks} morceaux "
          f"en {time.time() - start:.0f} s ({len(report.skipped)} fichiers ignorés)")


if __name__ == "__main__":
    main()
