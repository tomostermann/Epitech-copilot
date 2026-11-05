"""Étape 3 : découper chaque document en morceaux (chunks).

Pourquoi découper ? Un embedding résume un texte entier en UN vecteur.
Le vecteur d'un notebook de 76 000 caractères serait une moyenne floue de
tout ce qu'il contient. Un morceau de 1 000 caractères parle d'une seule
chose, donc son vecteur est précis, et la recherche peut le retrouver.

Méthode : découpage récursif. On essaie de couper aux frontières les plus
« naturelles » d'abord (titres, puis paragraphes, puis lignes, puis phrases),
et on ne descend à un séparateur plus fin que si un morceau reste trop gros.
"""

import re
from dataclasses import dataclass, field

from rag.loaders import Document


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


# Du séparateur le plus « fort » au plus faible, selon le type de fichier.
TEXT_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
CODE_SEPARATORS = ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
CODE_EXTS = {".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".php", ".sql",
             ".css", ".sh", ".html", ".puml"}


def _split_on(text: str, sep: str) -> list[str]:
    """Coupe sur sep en gardant le séparateur au début du morceau suivant
    (pour qu'un titre « ## PPO » reste collé à son contenu)."""
    if sep == "":
        return list(text)
    parts = text.split(sep)
    return [parts[0]] + [sep + p for p in parts[1:]]


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Regroupe des petits morceaux jusqu'à `size`, avec chevauchement."""
    chunks, current, length = [], [], 0
    for piece in pieces:
        if length + len(piece) > size and current:
            chunks.append("".join(current))
            # On garde la fin du morceau précédent pour démarrer le suivant
            while current and (length > overlap or length + len(piece) > size):
                length -= len(current[0])
                current.pop(0)
        current.append(piece)
        length += len(piece)
    if current:
        chunks.append("".join(current))
    return chunks


def split_text(text: str, size: int, overlap: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]

    # Premier séparateur présent dans le texte
    sep_index = next(i for i, s in enumerate(separators) if s == "" or s in text)
    sep, finer = separators[sep_index], separators[sep_index + 1:]

    pieces = []
    for piece in _split_on(text, sep):
        if len(piece) <= size:
            pieces.append(piece)
        else:
            # Morceau encore trop gros : on le redécoupe plus finement
            pieces.extend(split_text(piece, size, overlap, finer))
    return _merge(pieces, size, overlap)


PAGE_RE = re.compile(r"\[page (\d+)\]")


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    ext = doc.metadata.get("ext", "")
    seps = CODE_SEPARATORS if ext in CODE_EXTS else TEXT_SEPARATORS
    texts = [t.strip() for t in split_text(doc.text, size, overlap, seps)]
    texts = [t for t in texts if t]

    chunks, cursor = [], 0
    for i, text in enumerate(texts):
        meta = {**doc.metadata, "chunk_index": i, "chunk_count": len(texts)}
        # Position du morceau dans le document : sert à retrouver la page
        pos = doc.text.find(text[:80], max(0, cursor - overlap))
        if pos >= 0:
            cursor = pos
        if ext == ".pdf":
            # Dernier marqueur [page N] avant le début du morceau (ou en tête)
            pages = PAGE_RE.findall(doc.text[: cursor + 12])
            if pages:
                meta["page"] = int(pages[-1])
        chunks.append(Chunk(text=text, metadata=meta))
    return chunks
