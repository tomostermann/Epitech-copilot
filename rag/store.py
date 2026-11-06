"""Étape 4 : ranger les vecteurs dans ChromaDB, sur le disque.

Une base vectorielle stocke pour chaque morceau : un identifiant, son vecteur,
son texte et ses métadonnées. Sa seule vraie fonction : "donne-moi les k
vecteurs les plus proches de celui-ci", très vite, même sur des millions.
"""

from dataclasses import dataclass

import chromadb

from rag.chunking import Chunk
from rag.config import Config


@dataclass
class Hit:
    """Un morceau retrouvé par la recherche."""

    text: str
    metadata: dict
    score: float  # similarité cosinus : 1 = même sens, 0 = aucun rapport


def chunk_id(chunk: Chunk) -> str:
    # Identifiant stable : même fichier + même position = même id.
    # Réindexer un fichier remplace donc ses morceaux au lieu de les dupliquer.
    return f"{chunk.metadata['path']}#{chunk.metadata['chunk_index']}"


class VectorStore:
    def __init__(self, cfg: Config):
        cfg.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(cfg.db_path))
        self.name = cfg.collection
        self.collection = self._open()

    def _open(self):
        return self.client.get_or_create_collection(
            self.name,
            # On fournit nos propres vecteurs (Ollama) : pas de modèle Chroma
            embedding_function=None,
            # Distance cosinus : compare la direction des vecteurs, pas leur longueur
            configuration={"hnsw": {"space": "cosine"}},
        )

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.collection = self._open()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.collection.upsert(
            ids=[chunk_id(c) for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def query(self, embedding: list[float], k: int = 5, where: dict | None = None) -> list[Hit]:
        res = self.collection.query(query_embeddings=[embedding], n_results=k, where=where)
        return [
            # Chroma renvoie une distance (1 - cosinus) : on la retourne en score
            Hit(text=t, metadata=m, score=1 - d)
            for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def count(self) -> int:
        return self.collection.count()
