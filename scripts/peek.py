"""Étape 4 : interroger la base SANS LLM, pour voir ce que la recherche trouve.

    uv run python scripts/peek.py "que doit afficher model.py en cas d'erreur ?"
    uv run python scripts/peek.py "hyperparamètres PPO" --project STARTTREK -k 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_config
from rag.embeddings import Embedder
from rag.store import VectorStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--project", help="ne chercher que dans ce projet")
    args = ap.parse_args()

    cfg = load_config()
    store = VectorStore(cfg)
    vector = Embedder(cfg).embed_query(args.question)
    where = {"project": args.project} if args.project else None
    hits = store.query(vector, k=args.k, where=where)

    print(f"Question : {args.question}")
    print(f"Vecteur  : {len(vector)} nombres, début {[round(x, 3) for x in vector[:4]]}...\n")
    for rank, h in enumerate(hits, start=1):
        m = h.metadata
        page = f" p.{m['page']}" if "page" in m else ""
        print(f"#{rank}  score {h.score:.3f}   {m['project']} — {m['path']}{page}")
        print("    " + h.text[:300].replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    main()
