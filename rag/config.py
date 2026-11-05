"""Chargement de la configuration depuis un fichier YAML."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    docs_root: Path
    ollama_host: str
    chat_model: str
    embed_model: str
    db_path: Path
    collection: str


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Les chemins relatifs sont résolus par rapport au fichier de config,
    # pas au dossier d'où on lance le script.
    base = path.resolve().parent
    db_path = Path(raw["vector_store"]["path"])
    if not db_path.is_absolute():
        db_path = (base / db_path).resolve()

    return Config(
        docs_root=Path(raw["docs"]["root"]).expanduser(),
        ollama_host=raw["ollama"]["host"],
        chat_model=raw["ollama"]["chat_model"],
        embed_model=raw["ollama"]["embed_model"],
        db_path=db_path,
        collection=raw["vector_store"]["collection"],
    )
