"""
main.py
-------
Point d'entrée du Système IA BTP.
Lance l'API FastAPI avec tous les routers configurés.

Usage :
    python main.py
    # ou
    uvicorn main:app --reload --port $PORT
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api.routes import auth_router, router
from config import get_settings

settings = get_settings()


def _preload_vision_models() -> None:
    from couche_data.image_collecte import _get_blip_components, _get_clip_components

    try:
        _get_blip_components()
        print("BLIP charge et pret")
    except Exception as e:
        print(f"BLIP non disponible, ingestion image continuera sans description : {e}")

    try:
        _get_clip_components()
        print("CLIP charge et pret")
    except Exception as e:
        print(f"CLIP non disponible, encodage visuel desactive : {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(asyncio.to_thread(_preload_vision_models))
    yield

# ─────────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────────

app = FastAPI(
    title="Système IA BTP",
    description=(
        "API d'intelligence artificielle pour le secteur BTP. "
        "Base vectorielle · Analyse métier · Automatisation des opérations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS (à restreindre en production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth_router, tags=["Auth"])
app.include_router(router, prefix="/api/v1", tags=["BTP"])


# ─────────────────────────────────────────────
# Lancement direct
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.API_PORT,
        log_level=settings.log_level.lower(),
    )
