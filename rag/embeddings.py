"""Étape 4 : transformer du texte en vecteurs avec Ollama.

Un embedding est une liste de nombres (768 pour nomic-embed-text) qui
représente le SENS d'un texte. Deux textes qui parlent de la même chose
donnent des vecteurs proches, même s'ils n'ont aucun mot en commun.
"""

import ollama

from rag.chunking import Chunk
from rag.config import Config


def text_for_embedding(chunk: Chunk) -> str:
    """Le texte réellement vectorisé : le morceau précédé de son contexte.

    Un morceau du sujet START TREK ne contient pas forcément le mot
    "START TREK". En ajoutant le projet et le fichier en tête, une question
    comme "que demande le sujet de START TREK ?" retrouve bien ses morceaux.
    """
    m = chunk.metadata
    return f"Projet {m['project']} — {m['path']}\n\n{chunk.text}"


class Embedder:
    def __init__(self, cfg: Config):
        self.client = ollama.Client(host=cfg.ollama_host)
        self.model = cfg.embed_model
        self.query_prefix = cfg.embed_query_prefix
        self.document_prefix = cfg.embed_document_prefix
        self.batch_size = cfg.embed_batch_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors.extend(self.client.embed(model=self.model, input=batch).embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]
