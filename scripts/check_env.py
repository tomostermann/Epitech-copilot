"""Étape 1 : vérifie que l'environnement est prêt.

Lance-le depuis la racine du projet :
    uv run python scripts/check_env.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama

from rag.config import load_config


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [KO] {msg}")
    sys.exit(1)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main() -> None:
    cfg = load_config()
    client = ollama.Client(host=cfg.ollama_host)

    print("1. Configuration")
    if not cfg.docs_root.is_dir():
        fail(f"dossier introuvable : {cfg.docs_root}")
    ok(f"dossier à indexer : {cfg.docs_root}")

    print("2. Ollama")
    try:
        installed = {m.model for m in client.list().models}
    except Exception as e:
        fail(f"Ollama ne répond pas sur {cfg.ollama_host} ({e}). L'app Ollama est lancée ?")
    ok(f"Ollama répond sur {cfg.ollama_host}")

    for name in (cfg.chat_model, cfg.embed_model):
        full = name if ":" in name else f"{name}:latest"
        if full not in installed:
            fail(f"modèle absent : {name}  ->  ollama pull {name}")
        ok(f"modèle présent : {name}")

    print("3. Embeddings (le cœur du RAG)")
    phrases = [
        "Comment entraîner un agent PPO sur Lunar Lander ?",
        "Apprentissage par renforcement pour faire atterrir une fusée",
        "Recette de la tarte aux pommes",
    ]
    vecs = client.embed(model=cfg.embed_model, input=phrases).embeddings
    ok(f"une phrase devient un vecteur de {len(vecs[0])} nombres")
    print(f"       sens proche   : {cosine(vecs[0], vecs[1]):.3f}")
    print(f"       sens éloigné  : {cosine(vecs[0], vecs[2]):.3f}")
    print("       -> plus c'est proche de 1, plus le sens est proche")

    print("4. Modèle de chat")
    reply = client.chat(
        model=cfg.chat_model,
        messages=[{"role": "user", "content": "Réponds juste : prêt."}],
    )
    ok(f"réponse : {reply.message.content.strip()[:60]}")

    print("\nEnvironnement prêt pour l'étape 2.")


if __name__ == "__main__":
    main()
