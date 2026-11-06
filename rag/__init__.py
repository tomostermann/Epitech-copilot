"""Module RAG réutilisable (idée 1 et idée 3).

Utilisation depuis n'importe quel script :

    import rag
    rag.index()                          # tout le dossier, en incrémental
    rag.index("BACH-2-ANNEE/STARTTREK")  # seulement ce projet
    hits = rag.search("hyperparamètres PPO", k=5, project="STARTTREK")

Interface publique :
    index(path)       -> indexe un dossier (incrémental)
    search(query, k)  -> retrouve les passages pertinents avec leur source
"""

from pathlib import Path

from rag.config import Config, load_config
from rag.indexer import IndexReport
from rag.indexer import index as _index
from rag.search import search as _search
from rag.store import Hit


def _cfg(config: str | Path | Config) -> Config:
    return config if isinstance(config, Config) else load_config(config)


def index(
    path: str | Path | None = None,
    *,
    config: str | Path | Config = "config.yaml",
    full: bool = False,
    progress=None,
) -> IndexReport:
    return _index(_cfg(config), path, full=full, progress=progress)


def search(
    query: str,
    k: int | None = None,
    *,
    project: str | None = None,
    ext: str | list[str] | None = None,
    config: str | Path | Config = "config.yaml",
) -> list[Hit]:
    """Les k passages les plus proches de query. Chaque Hit a .text, .metadata, .score."""
    return _search(_cfg(config), query, k=k, project=project, ext=ext)


__all__ = ["index", "search", "load_config", "Config", "IndexReport", "Hit"]
