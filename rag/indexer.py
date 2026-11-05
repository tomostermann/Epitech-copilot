"""Étape 4 : la chaîne complète fichiers -> morceaux -> vecteurs -> base.

Pour l'instant on réindexe tout à chaque fois. L'étape 5 rendra
l'indexation incrémentale (seulement les fichiers nouveaux ou modifiés).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rag.chunking import Chunk, chunk_document
from rag.config import Config
from rag.embeddings import Embedder, text_for_embedding
from rag.loaders import Document, Skipped, load_documents
from rag.store import VectorStore


@dataclass
class IndexReport:
    files: int = 0
    chunks: int = 0
    skipped: list[Skipped] = field(default_factory=list)


def build_index(
    cfg: Config,
    root: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> IndexReport:
    report = IndexReport()
    chunks: list[Chunk] = []
    for r in load_documents(cfg, root):
        if isinstance(r, Skipped):
            report.skipped.append(r)
            continue
        report.files += 1
        chunks.extend(chunk_document(r, cfg.chunk_size, cfg.chunk_overlap))

    store = VectorStore(cfg)
    store.reset()
    embedder = Embedder(cfg)

    # On vectorise et on stocke par paquets pour pouvoir afficher la progression
    step = cfg.embed_batch_size * 4
    for i in range(0, len(chunks), step):
        batch = chunks[i : i + step]
        vectors = embedder.embed_documents([text_for_embedding(c) for c in batch])
        store.add(batch, vectors)
        report.chunks += len(batch)
        if progress:
            progress(report.chunks, len(chunks))
    return report
