"""Étape 6 (partie 2) : la génération, le "G" de RAG.

On ne demande pas au modèle ce qu'il SAIT : on lui donne les extraits retrouvés
et on lui demande de répondre À PARTIR D'EUX. C'est ce qui permet :
  - de répondre sur tes projets, qu'il n'a jamais vus pendant son entraînement ;
  - de citer des sources vérifiables ;
  - de dire "je ne sais pas" au lieu d'inventer, quand les extraits ne suffisent pas.
"""

from typing import Iterator

import ollama

from rag.config import Config
from rag.store import Hit

SYSTEM_PROMPT = """Tu es le copilote des projets EPITECH de Tom.
Règles :
- Réponds UNIQUEMENT à partir des extraits fournis, jamais de tes connaissances générales.
- Après chaque information, cite l'extrait d'où elle vient avec son numéro, par exemple [2].
- Si les extraits ne contiennent pas la réponse, dis-le clairement et n'invente rien.
- Les extraits peuvent être en anglais : réponds dans la langue de la question.
- Sois concis."""


def source_label(hit: Hit) -> str:
    m = hit.metadata
    page = f", p.{m['page']}" if "page" in m else ""
    return f"{m['project']} — {m['path']}{page}"


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{i}] ({source_label(h)})\n{h.text}" for i, h in enumerate(hits, start=1)]
    return "Extraits :\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQuestion : {question}"


def answer_stream(cfg: Config, question: str, hits: list[Hit]) -> Iterator[str]:
    """Renvoie la réponse morceau par morceau, au fur et à mesure qu'elle s'écrit."""
    client = ollama.Client(host=cfg.ollama_host)
    stream = client.chat(
        model=cfg.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        options={"num_ctx": cfg.chat_num_ctx, "temperature": 0.1},
        stream=True,
    )
    for part in stream:
        yield part.message.content
