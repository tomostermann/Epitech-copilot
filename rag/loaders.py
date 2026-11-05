"""Étape 2 : lire les fichiers et en sortir du texte + des métadonnées.

Un RAG ne vaut que ce qu'on lui donne : si un fichier est mal lu ici,
aucune étape suivante ne pourra rattraper l'information perdue.
"""

import fnmatch
import json
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pymupdf

from rag.config import Config


@dataclass
class Document:
    """Un fichier lu : son texte et d'où il vient."""

    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Skipped:
    """Un fichier qu'on n'a pas pu lire, et pourquoi."""

    path: str
    reason: str


# ---------- lecteurs par type de fichier ----------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    # PyMuPDF plutôt que pypdf : sur les sujets EPITECH, pypdf colle les mots
    # entre eux ("Youarepartofthefirst..."), ce qui rend les embeddings inutiles.
    pages = []
    with pymupdf.open(path) as pdf:
        texts = [page.get_text() for page in pdf]
    for i, text in enumerate(texts, start=1):
        text = text.strip()
        if text:
            # On garde le numéro de page : utile pour citer la source plus tard
            pages.append(f"[page {i}]\n{text}")
    return "\n\n".join(pages)


def read_notebook(path: Path) -> str:
    """Garde le markdown et le code, jette les sorties (graphes, tableaux)."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", [])).strip()
        if not source:
            continue
        if cell.get("cell_type") == "code":
            parts.append(f"```python\n{source}\n```")
        else:
            parts.append(source)
    return "\n\n".join(parts)


READERS = {".pdf": read_pdf, ".ipynb": read_notebook}


# ---------- parcours du dossier ----------

def iter_files(cfg: Config, root: Path | None = None) -> Iterator[Path]:
    """Parcourt root en sautant les dossiers exclus et les extensions inconnues."""
    root = root or cfg.docs_root
    for dirpath, dirnames, filenames in os.walk(root):
        # Modifier dirnames en place empêche os.walk de descendre dedans
        dirnames[:] = sorted(
            d for d in dirnames if d not in cfg.exclude_dirs and not d.startswith(".")
        )
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() not in cfg.extensions:
                continue
            if any(fnmatch.fnmatch(name, pat) for pat in cfg.exclude_files):
                continue
            yield path


def build_metadata(path: Path, cfg: Config) -> dict:
    """Déduit année et projet du chemin : BACH-2-ANNEE/STARTTREK/... """
    rel = path.relative_to(cfg.docs_root)
    parts = rel.parts
    depth = cfg.project_depth
    project = parts[depth - 1] if len(parts) > depth else "(racine)"
    group = "/".join(parts[: depth - 1]) if depth > 1 and len(parts) > depth else ""
    stat = path.stat()
    return {
        "project": project,
        "group": group,
        "path": str(rel),
        "filename": path.name,
        "ext": path.suffix.lower(),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def load_file(path: Path, cfg: Config) -> Document | Skipped:
    rel = str(path.relative_to(cfg.docs_root))
    size = path.stat().st_size
    if size == 0:
        return Skipped(rel, "fichier vide")
    reader = READERS.get(path.suffix.lower())
    # La limite de taille ne vaut que pour le texte brut. Un notebook ou un PDF
    # est gros à cause de ses images, qu'on ne garde pas de toute façon.
    if reader is None:
        if size > cfg.max_file_kb * 1024:
            return Skipped(rel, f"trop gros (> {cfg.max_file_kb} Ko)")
        reader = read_text
    try:
        text = reader(path).strip()
    except Exception as e:
        return Skipped(rel, f"erreur de lecture : {type(e).__name__}: {e}")
    # NFKC : transforme les ligatures typographiques ("ﬁ") en lettres normales ("fi")
    text = unicodedata.normalize("NFKC", text)
    if not text:
        # Typiquement un PDF scanné (des images, pas de texte)
        return Skipped(rel, "aucun texte extrait")
    return Document(text=text, metadata=build_metadata(path, cfg))


def load_documents(cfg: Config, root: Path | None = None) -> Iterator[Document | Skipped]:
    """Point d'entrée : lit tous les fichiers utiles sous root."""
    for path in iter_files(cfg, root):
        yield load_file(path, cfg)
