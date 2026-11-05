"""Étape 2 : montre ce que le chargeur sort de ton dossier EPITECH.

    uv run python scripts/inspect_docs.py                  # vue d'ensemble
    uv run python scripts/inspect_docs.py STARTTREK        # détail d'un projet
    uv run python scripts/inspect_docs.py --show chemin/relatif/du/fichier
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_config
from rag.loaders import Document, Skipped, load_documents, load_file


def show_file(cfg, rel: str) -> None:
    result = load_file(cfg.docs_root / rel, cfg)
    if isinstance(result, Skipped):
        print(f"Ignoré : {result.reason}")
        return
    print("Métadonnées :")
    for k, v in result.metadata.items():
        print(f"  {k:9} {v}")
    print(f"\nTexte extrait ({len(result.text)} caractères) :\n")
    print(result.text[:3000])
    if len(result.text) > 3000:
        print(f"\n[... {len(result.text) - 3000} caractères de plus]")


def main() -> None:
    cfg = load_config()
    args = sys.argv[1:]

    if args[:1] == ["--show"]:
        show_file(cfg, args[1])
        return

    only = args[0] if args else None
    docs: list[Document] = []
    skipped: list[Skipped] = []
    for r in load_documents(cfg):
        if isinstance(r, Skipped):
            skipped.append(r)
        elif only is None or r.metadata["project"] == only:
            docs.append(r)

    by_project = defaultdict(list)
    for d in docs:
        by_project[(d.metadata["group"], d.metadata["project"])].append(d)

    print(f"{'projet':38} {'fichiers':>8} {'caractères':>11}  types")
    for (group, project), ds in sorted(by_project.items()):
        chars = sum(len(d.text) for d in ds)
        exts = defaultdict(int)
        for d in ds:
            exts[d.metadata["ext"]] += 1
        types = " ".join(f"{e}:{n}" for e, n in sorted(exts.items(), key=lambda x: -x[1]))
        print(f"{group + '/' + project:38} {len(ds):>8} {chars:>11,}  {types}")

    total = sum(len(d.text) for d in docs)
    print(f"\nTotal : {len(docs)} fichiers, {total:,} caractères")

    if only:
        print(f"\nFichiers de {only} :")
        for d in docs:
            print(f"  {len(d.text):>8,}  {d.metadata['path']}")

    empty = [s for s in skipped if s.reason == "fichier vide"]
    skipped = [s for s in skipped if s.reason != "fichier vide"]
    if empty:
        print(f"\n{len(empty)} fichiers vides ignorés")
    if skipped:
        print(f"\n{len(skipped)} fichiers ignorés :")
        for s in skipped[:30]:
            print(f"  {s.path}  ->  {s.reason}")
        if len(skipped) > 30:
            print(f"  ... et {len(skipped) - 30} autres")


if __name__ == "__main__":
    main()
