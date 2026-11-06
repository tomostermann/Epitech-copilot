"""Étapes 4 et 5 : fichiers -> morceaux -> vecteurs -> base, en incrémental.

Vectoriser coûte cher (144 s pour tout le dossier avec bge-m3). L'indexation
incrémentale ne retraite que ce qui a changé depuis la dernière fois :

  1. date de modification identique         -> inchangé, on ne lit même pas
  2. date différente mais même hash         -> inchangé (fichier juste ré-enregistré)
  3. hash différent                         -> modifié : on remplace ses morceaux
  4. absent de la base                      -> nouveau : on l'ajoute
  5. dans la base mais plus sur le disque   -> supprimé : on retire ses morceaux
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rag.chunking import Chunk, chunk_document
from rag.config import Config
from rag.embeddings import Embedder, text_for_embedding
from rag.loaders import Skipped, file_hash, iter_files, load_file
from rag.store import VectorStore

# À incrémenter si on change la façon de découper ou de préparer les textes :
# toute la base sera alors reconstruite automatiquement.
PIPELINE_VERSION = 1


@dataclass
class IndexReport:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    skipped: list[Skipped] = field(default_factory=list)
    chunks: int = 0
    rebuilt: str | None = None  # raison d'une reconstruction complète


def settings_fingerprint(cfg: Config) -> str:
    """Tout ce qui, s'il change, rend les vecteurs déjà stockés incompatibles."""
    return "|".join(map(str, [
        PIPELINE_VERSION, cfg.embed_model, cfg.embed_document_prefix,
        cfg.chunk_size, cfg.chunk_overlap,
    ]))


def _in_scope(rel: str, prefix: str) -> bool:
    return prefix == "." or rel == prefix or rel.startswith(prefix + "/")


def index(
    cfg: Config,
    root: Path | str | None = None,
    full: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> IndexReport:
    """Indexe root (par défaut tout docs_root). Ne touche qu'à ce qui est sous root."""
    root = Path(root) if root else cfg.docs_root
    if not root.is_absolute():
        root = cfg.docs_root / root
    try:
        prefix = str(root.relative_to(cfg.docs_root))
    except ValueError:
        raise ValueError(f"{root} n'est pas dans {cfg.docs_root}") from None

    report = IndexReport()
    store = VectorStore(cfg)
    fingerprint = settings_fingerprint(cfg)

    if full:
        report.rebuilt = "demandé (--full)"
    elif store.count() and store.settings != fingerprint:
        report.rebuilt = "réglages modifiés (modèle, préfixe ou chunking)"
    if report.rebuilt or store.settings != fingerprint:
        store.reset()
        store.set_settings(fingerprint)
        # Base vidée : il faut tout réindexer, pas seulement le sous-dossier demandé,
        # sinon les autres projets disparaîtraient de la base sans prévenir.
        root, prefix = cfg.docs_root, "."

    indexed = {p: m for p, m in store.indexed_files().items() if _in_scope(p, prefix)}
    seen: set[str] = set()
    pending: list[list[Chunk]] = []  # morceaux à vectoriser, groupés par fichier

    for path in iter_files(cfg, root):
        rel = str(path.relative_to(cfg.docs_root))
        seen.add(rel)
        old = indexed.get(rel)
        mtime = path.stat().st_mtime

        if old and old["mtime"] == mtime:
            report.unchanged += 1
            continue
        if old and old["hash"] == file_hash(path):
            store.touch_file(rel, mtime)
            report.unchanged += 1
            continue

        result = load_file(path, cfg)
        if old:
            store.delete_file(rel)
        if isinstance(result, Skipped):
            report.skipped.append(result)
            if old:
                report.removed.append(rel)
            continue
        (report.modified if old else report.added).append(rel)
        pending.append(chunk_document(result, cfg.chunk_size, cfg.chunk_overlap))

    for rel in indexed.keys() - seen:
        store.delete_file(rel)
        report.removed.append(rel)

    _embed_and_store(cfg, store, pending, report, progress)
    return report


def _embed_and_store(cfg, store, pending, report, progress) -> None:
    """Vectorise par paquets, sans jamais couper un fichier entre deux paquets.

    Si l'indexation plante en route (Ollama arrêté…), chaque fichier est soit
    entièrement en base, soit absent : le prochain lancement le reprendra.
    Un fichier à moitié stocké serait vu comme "déjà indexé" et resterait
    incomplet pour toujours.
    """
    total = sum(len(f) for f in pending)
    if not total:
        return
    embedder = Embedder(cfg)
    target = cfg.embed_batch_size * 4
    batch: list[Chunk] = []
    for file_chunks in pending:
        batch.extend(file_chunks)
        if len(batch) >= target:
            _flush(embedder, store, batch, report, progress, total)
            batch = []
    if batch:
        _flush(embedder, store, batch, report, progress, total)


def _flush(embedder, store, batch, report, progress, total) -> None:
    vectors = embedder.embed_documents([text_for_embedding(c) for c in batch])
    store.add(batch, vectors)
    report.chunks += len(batch)
    if progress:
        progress(report.chunks, total)
