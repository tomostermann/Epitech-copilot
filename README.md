# Epitech-copilot

Copilote local pour mes projets EPITECH : il répond aux questions à partir des sujets, docs et code (RAG), et agit sur Notion (MCP). Tout tourne en local avec Ollama.

## Installation

```bash
brew install ollama uv          # ou l'app Ollama depuis ollama.com
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
uv sync
uv run python scripts/check_env.py
```

## Structure

- `rag/` : module RAG réutilisable (indépendant de l'agent et du MCP)
- `scripts/` : scripts de test par étape
- `config.yaml` : chemins et modèles

## Serveur MCP

`servers/docs_server.py` expose le RAG à n'importe quel client MCP (transport stdio) :

- `list_epitech_projects` : les projets indexés
- `search_epitech_docs(query, k, project, file_types)` : les passages les plus proches

```bash
uv run python scripts/mcp_client.py "combien d'actions ?" --project STARTTREK
npx @modelcontextprotocol/inspector uv run python servers/docs_server.py
```
