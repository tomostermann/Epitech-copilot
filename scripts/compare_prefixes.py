"""Étape 4 : effet des préfixes nomic sur la séparation des sens.

    uv run python scripts/compare_prefixes.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama

from rag.config import load_config

QUESTION = "Comment entraîner un agent PPO sur Lunar Lander ?"
DOCS = {
    "proche ": "Apprentissage par renforcement pour faire atterrir une fusée",
    "éloigné": "Recette de la tarte aux pommes",
}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main() -> None:
    cfg = load_config()
    client = ollama.Client(host=cfg.ollama_host)
    for label, qp, dp in [
        ("sans préfixe", "", ""),
        ("avec préfixes", cfg.embed_query_prefix, cfg.embed_document_prefix),
    ]:
        q = client.embed(model=cfg.embed_model, input=[qp + QUESTION]).embeddings[0]
        print(label)
        scores = {}
        for name, doc in DOCS.items():
            d = client.embed(model=cfg.embed_model, input=[dp + doc]).embeddings[0]
            scores[name] = cosine(q, d)
            print(f"  {name} : {scores[name]:.3f}")
        print(f"  écart   : {scores['proche '] - scores['éloigné']:.3f}\n")


if __name__ == "__main__":
    main()
