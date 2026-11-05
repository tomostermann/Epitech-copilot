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
