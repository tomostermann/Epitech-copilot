"""Étape 7 : serveur MCP qui expose le RAG comme des OUTILS.

Un serveur MCP ne fait rien tout seul : il attend qu'un client (Claude Desktop,
l'agent de l'idée 3, MCP Inspector...) se connecte, lui demande la liste des
outils disponibles, puis les appelle. Le client ne sait rien de ChromaDB ni
d'Ollama : il voit seulement des noms d'outils, leur description et leurs
paramètres. C'est tout l'intérêt du protocole.

Transport stdio : le client lance ce script comme un sous-processus et lui parle
en JSON-RPC sur stdin/stdout. Conséquence importante : ce script ne doit JAMAIS
écrire sur stdout (pas de print), sinon il corrompt les messages du protocole.
Les logs vont sur stderr.

    uv run python servers/docs_server.py     # se lance, puis attend un client
"""

import sys
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

import rag
from rag.search import list_projects

# Chemin absolu : le client peut lancer le serveur depuis n'importe quel dossier
CFG = rag.load_config(ROOT / "config.yaml")

server = MCPServer(
    name="epitech-docs",
    instructions=(
        "Recherche dans les documents des projets EPITECH de Tom : sujets PDF, "
        "notes, notebooks et code. Utilise list_epitech_projects pour connaître "
        "les noms exacts des projets, puis search_epitech_docs pour retrouver des "
        "passages. Cite toujours le projet et le fichier des passages utilisés."
    ),
)

# readOnlyHint : indique au client que l'outil ne modifie rien. Un client peut
# s'en servir pour l'autoriser sans demander de confirmation à l'utilisateur.
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


class Passage(BaseModel):
    """Un passage retrouvé, avec de quoi le citer."""

    rank: int
    score: float
    project: str
    path: str
    page: int | None = None
    text: str


@server.tool(annotations=READ_ONLY)
def list_epitech_projects() -> list[str]:
    """Liste les projets EPITECH indexés (ex : STARTTREK, CODENAME, 2D-GAME).

    À utiliser pour connaître le nom exact d'un projet avant de filtrer une recherche.
    """
    return list_projects(CFG)


@server.tool(annotations=READ_ONLY)
def search_epitech_docs(
    # Les descriptions des paramètres finissent dans le schéma JSON envoyé au
    # client : c'est ce que le LLM lit pour savoir comment remplir chaque argument.
    query: Annotated[str, Field(description="La question ou le sujet recherché, en langage naturel.")],
    k: Annotated[int, Field(ge=1, le=20, description="Nombre de passages à renvoyer.")] = 5,
    project: Annotated[str | None, Field(
        description="Nom exact d'un projet pour ne chercher que dedans (voir "
                    "list_epitech_projects). Vide = tout le dossier.")] = None,
    file_types: Annotated[list[str] | None, Field(
        description='Extensions à garder : ["pdf"] pour les sujets officiels, '
                    '["md"] pour les notes, ["py", "ipynb"] pour le code.')] = None,
) -> list[Passage]:
    """Recherche sémantique dans les documents des projets EPITECH de Tom.

    Renvoie les k passages dont le SENS est le plus proche de la question, même
    sans mot en commun, et même si la question et le document ne sont pas dans
    la même langue.
    """
    if project:
        known = list_projects(CFG)
        if project not in known:
            # ToolError : erreur prévue, son message est transmis au modèle client,
            # qui peut se corriger tout seul. Une autre exception n'enverrait que
            # "Error executing tool" (pour ne pas divulguer de détails internes).
            raise ToolError(f"Projet inconnu : {project}. Projets disponibles : {', '.join(known)}")

    hits = rag.search(query, k=k, project=project, ext=file_types, config=CFG)
    return [
        Passage(
            rank=i,
            score=round(h.score, 3),
            project=h.metadata["project"],
            path=h.metadata["path"],
            page=h.metadata.get("page"),
            text=h.text,
        )
        for i, h in enumerate(hits, start=1)
    ]


if __name__ == "__main__":
    server.run()  # transport stdio par défaut
