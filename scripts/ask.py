"""Étape 6 : poser une question au copilote (RAG complet).

    uv run python scripts/ask.py "que demande le sujet pour la partie PPO ?"
    uv run python scripts/ask.py "..." --project STARTTREK
    uv run python scripts/ask.py "..." --context        # voir les extraits envoyés au modèle
    uv run python scripts/ask.py                         # mode conversation
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag
from rag.answer import answer_stream, source_label

DIM, RESET = "\033[2m", "\033[0m"


def ask(cfg, question: str, args) -> None:
    t0 = time.time()
    hits = rag.search(question, k=args.k, project=args.project, ext=args.ext, config=cfg)
    t1 = time.time()

    print(f"{DIM}Recherche : {len(hits)} extraits en {t1 - t0:.1f} s{RESET}")
    for i, h in enumerate(hits, start=1):
        print(f"{DIM}  [{i}] {h.score:.2f}  {source_label(h)}{RESET}")
        if args.context:
            print(f"{DIM}      " + h.text[:500].replace("\n", "\n      ") + RESET)
    print()

    if not hits:
        print("Aucun extrait trouvé (le projet existe ? l'index est construit ?)")
        return
    for token in answer_stream(cfg, question, hits):
        print(token, end="", flush=True)
    print(f"\n{DIM}(réponse en {time.time() - t1:.1f} s){RESET}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("-k", type=int)
    ap.add_argument("--project", help="ne chercher que dans ce projet (ex : STARTTREK)")
    ap.add_argument("--ext", nargs="+", help="types de fichiers (ex : --ext pdf md)")
    ap.add_argument("--context", action="store_true", help="afficher les extraits")
    args = ap.parse_args()
    cfg = rag.load_config()

    if args.question:
        ask(cfg, args.question, args)
        return
    print("Copilote EPITECH — pose ta question (Ctrl+D pour quitter)\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question:
            ask(cfg, question, args)


if __name__ == "__main__":
    main()
