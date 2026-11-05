"""Chargement de la configuration depuis un fichier YAML."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    docs_root: Path
    project_depth: int
    extensions: frozenset[str]
    exclude_dirs: frozenset[str]
    exclude_files: tuple[str, ...]
    max_file_kb: int
    chunk_size: int
    chunk_overlap: int
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

    docs = raw["docs"]
    return Config(
        docs_root=Path(docs["root"]).expanduser(),
        project_depth=docs.get("project_depth", 1),
        extensions=frozenset(e.lower() for e in docs["extensions"]),
        exclude_dirs=frozenset(docs.get("exclude_dirs", [])),
        exclude_files=tuple(docs.get("exclude_files", [])),
        max_file_kb=docs.get("max_file_kb", 500),
        chunk_size=raw.get("chunking", {}).get("size", 1000),
        chunk_overlap=raw.get("chunking", {}).get("overlap", 150),
        ollama_host=raw["ollama"]["host"],
        chat_model=raw["ollama"]["chat_model"],
        embed_model=raw["ollama"]["embed_model"],
        db_path=db_path,
        collection=raw["vector_store"]["collection"],
    )
