"""
api/routes.py
-------------
Endpoints FastAPI exposant toutes les fonctionnalités du Système IA BTP.

Routes disponibles :
POST /ingerer/fichier     : Ingest d'un fichier uploadé
POST /ingerer/texte       : Ingest d'un texte brut
GET  /stats               : Statistiques de la base vectorielle
POST /question            : Q&A sur la base de connaissance
POST /conformite          : Vérification réglementaire
POST /risques             : Détection de risques
POST /recommandations     : Rapport de recommandations structuré
POST /audit/{projet}      : Audit complet d'un projet
GET  /alertes             : Alertes documents critiques
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel

from couche_data.gmail_collecte import exchange_code_for_token, generate_auth_url, has_valid_gmail_token
from couche_data.dtu_normes_search import (
    check_conformity,
    is_reglementaire_query,
    list_dtu_normes,
    search_reglementaire_response,
)
from couche_data.vectorisation import stats_collection
from couche_ia.llm_engine import LLMEngine, documents_to_context_chunks
from couche_ia.retriever import rechercher
from couche_execution.workflows import (
    ingerer_document,
    ingerer_fichier as ingerer_fichier_smart,
    ingerer_texte_brut,
    ingerer_dossier,
    ingerer_gmail,
    analyser_et_recommander,
    audit_projet,
    verifier_conformite_element,
)
from couche_ia.analyse_metier import repondre, detecter_risques
from couche_execution.recommandations import generer_recommandations, generer_alertes_critiques
from config import get_settings

router = APIRouter()
auth_router = APIRouter()
logger = logging.getLogger("btp_api.routes")
logger.setLevel(logging.INFO)
Path("logs").mkdir(exist_ok=True)
if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
    file_handler = logging.FileHandler("logs/api.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(file_handler)
_jobs: dict[str, dict] = {}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}
OAUTH_STATE_TTL_SECONDS = 15 * 60


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _sign_oauth_state(code_verifier: str) -> str:
    settings = get_settings()
    payload = {
        "code_verifier": code_verifier,
        "iat": int(time.time()),
        "nonce": secrets.token_urlsafe(16),
    }
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(settings.secret_key.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64url_encode(signature)}"


def _verify_oauth_state(state: Optional[str]) -> Optional[str]:
    if not state or "." not in state:
        logger.warning("OAuth Gmail state absent ou mal forme")
        return None
    try:
        payload_b64, signature_b64 = state.rsplit(".", 1)
        settings = get_settings()
        expected_signature = hmac.new(
            settings.secret_key.encode("utf-8"),
            payload_b64.encode("ascii"),
            hashlib.sha256,
        ).digest()
        received_signature = _b64url_decode(signature_b64)
        if not hmac.compare_digest(expected_signature, received_signature):
            logger.warning("OAuth Gmail state signature invalide")
            return None
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        if int(time.time()) - int(payload.get("iat", 0)) > OAUTH_STATE_TTL_SECONDS:
            logger.warning("OAuth Gmail state expire")
            return None
        code_verifier = payload.get("code_verifier")
        if not code_verifier:
            logger.warning("OAuth Gmail state sans code_verifier")
            return None
        return str(code_verifier)
    except Exception as exc:
        logger.warning("OAuth Gmail state introuvable ou invalide: %s", exc)
        return None


def _traiter_image_background(
    tmp_path: str,
    fichier_nom: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str,
    job_id: str,
) -> None:
    try:
        from couche_data.image_collecte import ingerer_image

        resultat = ingerer_image(
            fichier_path=tmp_path,
            projet=projet,
            lot_technique=lot_technique,
            criticite=criticite,
            auteur=auteur,
        )
        resultat["fichier_original"] = fichier_nom
        _jobs[job_id] = {"statut": "termine", **resultat}
    except Exception as e:
        _jobs[job_id] = {
            "statut": "erreur",
            "message": str(e),
            "fichier_original": fichier_nom,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────
# Schémas de requête
# ─────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    projet: Optional[str] = None
    k: Optional[int] = None


class TexteIngestionRequest(BaseModel):
    contenu: str
    source: str = "saisie_manuelle"
    projet: str = "non_défini"
    lot_technique: str = "non_défini"
    auteur: str = "inconnu"
    criticite: str = "normale"
    type_document: str = "general"


class ConformiteRequest(BaseModel):
    element: str


class RisquesRequest(BaseModel):
    situation: str
    projet: Optional[str] = None


class RecommandationsRequest(BaseModel):
    situation: str
    projet: Optional[str] = None


class DossierIngestionRequest(BaseModel):
    dossier: str
    projet: str = "non_defini"
    lot_technique: str = "non_defini"


class GmailIngestionRequest(BaseModel):
    query: Optional[str] = None
    max_results: Optional[int] = None
    projet: str = "non_defini"
    lot_technique: str = "non_defini"
    criticite: str = "normale"


class DtuSearchRequest(BaseModel):
    query: str
    k: Optional[int] = 6


class DtuCheckRequest(BaseModel):
    description_travaux: str
    k: Optional[int] = 5


@auth_router.get("/auth/gmail/login", summary="Demarrer OAuth Gmail")
async def gmail_login():
    """Retourne l'URL OAuth Google si aucun token Gmail valide n'existe."""
    try:
        if has_valid_gmail_token():
            return {"need_auth": False, "message": "Gmail deja connecte."}
        code_verifier = _b64url_encode(secrets.token_bytes(64))
        signed_state = _sign_oauth_state(code_verifier)
        auth_data = generate_auth_url(state=signed_state, code_verifier=code_verifier)
        auth_data.pop("code_verifier", None)
        return {"need_auth": True, **auth_data}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@auth_router.get("/auth/gmail/status", summary="Statut OAuth Gmail")
async def gmail_status():
    """Indique si un token Gmail valide existe cote backend."""
    try:
        connected = has_valid_gmail_token()
        return {
            "connected": connected,
            "need_auth": not connected,
            "message": "Gmail connecte." if connected else "Connexion Gmail requise.",
        }
    except Exception as e:
        logger.warning("Impossible de verifier le statut Gmail: %s", e)
        return {
            "connected": False,
            "need_auth": True,
            "message": "Connexion Gmail requise.",
        }


@auth_router.get("/auth/gmail/callback", summary="Callback OAuth Gmail")
async def gmail_callback(code: str, state: Optional[str] = None):
    """Echange le code OAuth Google et sauvegarde token.json."""
    try:
        code_verifier = _verify_oauth_state(state)
        if not code_verifier:
            logger.warning("Session OAuth Gmail introuvable ou expiree pour state=%s", bool(state))
            raise HTTPException(
                status_code=400,
                detail="Session OAuth introuvable ou expiree. Relance la connexion Gmail.",
            )
        exchange_code_for_token(code=code, state=state, code_verifier=code_verifier)
        return {
            "success": True,
            "message": "Connexion Gmail reussie. Retourne dans Streamlit et relance l'ingestion.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Routes d'ingestion
# ─────────────────────────────────────────────

@router.post("/ingerer/fichier", summary="Ingérer un fichier (PDF, DOCX, TXT, image OCR)")
async def ingerer_fichier(
    background_tasks: BackgroundTasks,
    fichier: UploadFile = File(...),
    projet: str = Form("non_défini"),
    lot_technique: str = Form("non_défini"),
    auteur: str = Form("inconnu"),
    criticite: str = Form("normale"),
):
    """
    Upload et ingest complet d'un fichier dans la base vectorielle.
    Formats supportés : PDF, DOCX, TXT, MD, JPG, PNG, TIFF, BMP, WEBP.
    """
    extension = Path(fichier.filename).suffix.lower()
    formats_supportes = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".ifc",
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }

    if extension not in formats_supportes:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : '{extension}'. Acceptés : {formats_supportes}",
        )

    # Sauvegarde temporaire du fichier uploadé
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        shutil.copyfileobj(fichier.file, tmp)
        tmp_path = tmp.name

    if extension in IMAGE_EXTENSIONS:
        texte_ocr_apercu = ""
        try:
            from PIL import Image
            from couche_data.image_collecte import extraire_texte_image_ocr

            with Image.open(tmp_path) as image:
                texte_ocr_apercu = extraire_texte_image_ocr(image)[:200]
        except Exception:
            pass

        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"statut": "en_cours", "fichier_original": fichier.filename}
        background_tasks.add_task(
            _traiter_image_background,
            tmp_path,
            fichier.filename,
            projet,
            lot_technique,
            criticite,
            auteur,
            job_id,
        )

        return {
            "statut": "en_cours",
            "job_id": job_id,
            "fichier": fichier.filename,
            "pipeline_utilise": "image_analyse_en_cours",
            "message": "OCR termine. Analyse BLIP/CLIP en cours en arriere-plan.",
            "texte_ocr_apercu": texte_ocr_apercu,
            "projet": projet,
        }

    try:
        resultat = ingerer_fichier_smart(
            fichier_path=tmp_path,
            projet=projet,
            lot_technique=lot_technique,
            criticite=criticite,
            auteur=auteur,
        )
        # Renommer la source avec le nom original
        resultat["fichier_original"] = fichier.filename
        return resultat
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("/job/{job_id}", summary="Statut d'un job image en arriere-plan")
async def statut_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return {"statut": "inconnu", "job_id": job_id}
    return {"job_id": job_id, **job}


@router.post("/ingerer/texte", summary="Ingérer un texte brut (email, WhatsApp, note)")
async def ingerer_texte(body: TexteIngestionRequest):
    """Ingest d'un texte brut avec métadonnées BTP."""
    try:
        return ingerer_texte_brut(
            contenu=body.contenu,
            source=body.source,
            projet=body.projet,
            lot_technique=body.lot_technique,
            auteur=body.auteur,
            criticite=body.criticite,
            type_document=body.type_document,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingerer/dossier", summary="Ingerer un dossier de documents")
async def ingerer_dossier_route(body: DossierIngestionRequest):
    """Ingest batch d'un dossier local contenant des documents supportes."""
    try:
        return ingerer_dossier(
            dossier=body.dossier,
            projet=body.projet,
            lot_technique=body.lot_technique,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingerer/gmail", summary="Ingerer des emails depuis Gmail")
async def ingerer_gmail_route(body: GmailIngestionRequest):
    """Ingest des emails Gmail via Google credentials OAuth."""
    try:
        result = ingerer_gmail(
            query=body.query,
            max_results=body.max_results,
            projet=body.projet,
            lot_technique=body.lot_technique,
            criticite=body.criticite,
        )
        if result.get("need_auth"):
            _remember_pkce_verifier(result)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Routes d'interrogation
# ─────────────────────────────────────────────

@router.post("/question", summary="Poser une question sur la base de connaissance BTP")
async def poser_question(body: QuestionRequest):
    """Q&A général sur la base vectorielle BTP."""
    request_id = uuid.uuid4().hex[:8]
    started_at = time.perf_counter()
    settings = get_settings()
    filtre = {"projet": body.projet} if body.projet else None
    k = body.k or settings.retrieval_k

    try:
        if is_reglementaire_query(body.question):
            logger.info("[question:%s] question reglementaire detectee", request_id)
            response = search_reglementaire_response(body.question, k=k)
            return {
                "question": body.question,
                "reponse": response,
                "projet": body.projet,
            }

        logger.info(
            "[question:%s] debut | projet=%s | k=%s | modele=%s | base_url=%s",
            request_id,
            body.projet or "-",
            k,
            settings.llm_model,
            settings.openai_base_url,
        )
        logger.info("[question:%s] avant recherche ChromaDB", request_id)
        docs = rechercher(body.question, k=k, filtre=filtre)
        logger.info(
            "[question:%s] apres recherche ChromaDB | docs=%s | elapsed=%.2fs",
            request_id,
            len(docs),
            time.perf_counter() - started_at,
        )
        chunks = documents_to_context_chunks(docs)
        logger.info(
            "[question:%s] avant appel LLM | chunks=%s | modele=%s",
            request_id,
            len(chunks),
            settings.llm_model,
        )
        reponse = LLMEngine().generate(body.question, chunks)
        logger.info(
            "[question:%s] apres appel LLM | chars=%s | total=%.2fs",
            request_id,
            len(reponse),
            time.perf_counter() - started_at,
        )
        return {"question": body.question, "reponse": reponse, "projet": body.projet}
    except Exception as e:
        logger.exception(
            "[question:%s] erreur | total=%.2fs",
            request_id,
            time.perf_counter() - started_at,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conformite", summary="Vérifier la conformité d'un élément aux normes BTP")
async def verifier_conformite_route(body: ConformiteRequest):
    """Vérifie la conformité réglementaire d'un élément (DTU, normes NF, code de la construction)."""
    try:
        return verifier_conformite_element(body.element)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risques", summary="Détecter les risques d'une situation projet")
async def detecter_risques_route(body: RisquesRequest):
    """Analyse une situation pour identifier les risques BTP."""
    try:
        risques = detecter_risques(body.situation, projet=body.projet)
        return {"situation": body.situation, "projet": body.projet, "risques": risques}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommandations", summary="Générer des recommandations structurées")
async def recommandations_route(body: RecommandationsRequest):
    """Génère un rapport de recommandations opérationnelles structuré."""
    try:
        rapport = generer_recommandations(body.situation, projet=body.projet)
        return rapport.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyser", summary="Analyser une situation et générer des recommandations")
async def analyser_route(body: RecommandationsRequest):
    """Workflow combiné : analyse IA + risques + recommandations."""
    try:
        return analyser_et_recommander(body.situation, projet=body.projet)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Routes d'audit et alertes
# ─────────────────────────────────────────────

@router.get("/dtu/list", summary="Lister les DTU et normes ingeres")
async def dtu_list():
    return list_dtu_normes()


@router.post("/dtu/search", summary="Recherche reglementaire DTU/NF/EN/ISO")
async def dtu_search(body: DtuSearchRequest):
    response = search_reglementaire_response(body.query, k=body.k or 10)
    return {"query": body.query, **response}


@router.post("/dtu/check-conformity", summary="Verifier une description travaux par rapport aux DTU/normes")
async def dtu_check_conformity(body: DtuCheckRequest):
    return check_conformity(body.description_travaux, k=body.k or 5)


@router.post("/audit/{projet}", summary="Audit complet d'un projet")
async def audit_route(projet: str):
    """
    Génère un audit complet : conformité + risques + alertes + recommandations.
    Opération longue (~30-60s selon la taille du projet).
    """
    try:
        return audit_projet(projet)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alertes", summary="Récupérer les alertes sur documents critiques")
async def alertes_route(projet: Optional[str] = Query(None, description="Filtrer par projet")):
    """Retourne la liste des alertes issues des documents critiques ou à haute criticité."""
    try:
        alertes = generer_alertes_critiques(projet=projet)
        return {"projet": projet, "nombre_alertes": len(alertes), "alertes": alertes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Routes utilitaires
# ─────────────────────────────────────────────

@router.get("/stats", summary="Statistiques de la base vectorielle")
async def stats_route():
    """Retourne les statistiques de la collection ChromaDB."""
    return stats_collection()


@router.get("/health", summary="Health check")
async def health():
    return {"statut": "ok", "service": "Système IA BTP"}
