"""Étape 6 (partie 1) : la recherche, le "R" de RAG (Retrieval).

C'est la fonction que l'idée 3 et le serveur MCP utiliseront : elle ne dépend
d'aucun LLM de chat, seulement du modèle d'embeddings et de la base.
"""

from rag.config import Config
from rag.embeddings import Embedder
from rag.store import Hit, VectorStore


def build_filter(project: str | None = None, ext: str | list[str] | None = None) -> dict | None:
    """Filtre sur les métadonnées : la recherche ne regarde que ces morceaux."""
    conditions = []
    if project:
        conditions.append({"project": project})
    if ext:
        exts = [ext] if isinstance(ext, str) else ext
        conditions.append({"ext": {"$in": [e if e.startswith(".") else f".{e}" for e in exts]}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(
    cfg: Config,
    query: str,
    k: int | None = None,
    project: str | None = None,
    ext: str | list[str] | None = None,
) -> list[Hit]:
    vector = Embedder(cfg).embed_query(query)
    where = build_filter(project, ext)
    return VectorStore(cfg).query(vector, k=k or cfg.search_k, where=where)


def list_projects(cfg: Config) -> list[str]:
    return sorted({m["project"] for m in VectorStore(cfg).collection.get(
        where={"chunk_index": 0}, include=["metadatas"])["metadatas"]})
