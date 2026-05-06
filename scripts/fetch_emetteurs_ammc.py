"""
Script one-shot : récupère la liste complète des émetteurs AMMC (~170)
et l'enregistre dans data/emetteurs_ammc.json.
L'API utilisera ce fichier si le scrape en direct échoue (SSL, etc.).

Usage (à la racine du projet RAG) :
  python scripts/fetch_emetteurs_ammc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Racine du projet
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "data" / "emetteurs_ammc.json"


def main() -> None:
    from rag_agent.scraper import AMMCScraper
    scraper = AMMCScraper()
    print("Récupération de la liste des émetteurs AMMC (toutes les pages)...")
    liste = scraper.fetch_liste_emetteurs(max_pages=20)
    if not liste:
        print("Aucun émetteur récupéré (vérifier la connexion ou le certificat SSL).")
        sys.exit(1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(liste, f, ensure_ascii=False, indent=2)
    print(f"OK : {len(liste)} émetteurs enregistrés dans {OUT_PATH}")
    sys.exit(0)


if __name__ == "__main__":
    main()
