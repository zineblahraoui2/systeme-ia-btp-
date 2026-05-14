#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -f ".venv/Scripts/activate" ]; then
  echo "Erreur: environnement virtuel introuvable: .venv/Scripts/activate"
  echo "Cree-le ou installe les dependances avant de lancer le projet."
  exit 1
fi

source .venv/Scripts/activate

APP_PORT="${PORT:-$(python -c 'import config; print(config.API_PORT)')}"
uvicorn main:app --host 0.0.0.0 --port "$APP_PORT"
