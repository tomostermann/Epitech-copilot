"""Étape 7 : un client MCP minimal, pour voir le protocole de l'intérieur.

Il fait exactement ce que fera l'agent de l'idée 3 : lancer le serveur,
demander ses outils, en appeler un.

    uv run python scripts/mcp_client.py                         # liste les outils
    uv run python scripts/mcp_client.py "combien d'actions ?" --project STARTTREK
    uv run python scripts/mcp_client.py "..." --raw             # réponse JSON brute
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


async def main(args) -> None:
    # Le client lance le serveur comme un sous-processus et lui parle sur stdin/stdout
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "servers" / "docs_server.py")])

    async with Client(params) as client:
        info = client.server_info
        print(f"Connecté à « {info.name} »\n" if info else "Connecté\n")

        # 1. Découverte : le client ne connaît pas les outils à l'avance, il les demande
        tools = (await client.list_tools()).tools
        print(f"{len(tools)} outils exposés :")
        for t in tools:
            print(f"\n  {t.name}")
            print("    " + (t.description or "").strip().splitlines()[0])
            if args.schema:
                print("    schéma : " + json.dumps(t.input_schema, ensure_ascii=False))

        if not args.question:
            return

        # 2. Appel : nom de l'outil + arguments en JSON, c'est tout
        arguments = {"query": args.question, "k": args.k}
        if args.project:
            arguments["project"] = args.project
        if args.ext:
            arguments["file_types"] = args.ext
        print(f"\n→ call_tool('search_epitech_docs', {json.dumps(arguments, ensure_ascii=False)})\n")
        result = await client.call_tool("search_epitech_docs", arguments)

        if result.is_error:
            print("Erreur renvoyée par le serveur :", result.content[0].text)
            return
        if args.raw:
            print(json.dumps(result.structured_content, indent=2, ensure_ascii=False))
            return
        for p in result.structured_content["result"]:
            page = f" p.{p['page']}" if p.get("page") else ""
            print(f"#{p['rank']}  {p['score']:.2f}  {p['project']} — {p['path']}{page}")
            print("    " + p["text"][:200].replace("\n", "\n    ") + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--project")
    ap.add_argument("--ext", nargs="+")
    ap.add_argument("--schema", action="store_true", help="afficher le schéma JSON des paramètres")
    ap.add_argument("--raw", action="store_true", help="afficher la réponse JSON brute")
    asyncio.run(main(ap.parse_args()))
