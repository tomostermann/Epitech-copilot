"""Module RAG réutilisable (idée 1 et idée 3).

Utilisation depuis n'importe quel script :

    import rag
    rag.index()                          # tout le dossier, en incrémental
    rag.index("BACH-2-ANNEE/STARTTREK")  # seulement ce projet

Interface publique :
    index(path)       -> indexe un dossier (incrémental)
    search(query, k)  -> retrouve les passages pertinents avec leur source (étape 6)
"""

from pathlib import Path

from rag.config import Config, load_config
from rag.indexer import IndexReport
from rag.indexer import index as _index


def index(
    path: str | Path | None = None,
    *,
    config: str | Path | Config = "config.yaml",
    full: bool = False,
    progress=None,
) -> IndexReport:
    cfg = config if isinstance(config, Config) else load_config(config)
    return _index(cfg, path, full=full, progress=progress)


__all__ = ["index", "load_config", "Config", "IndexReport"]
