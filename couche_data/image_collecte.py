from __future__ import annotations

import base64
import mimetypes
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import chromadb
from PIL import Image, ImageFilter, ImageOps
from langchain_core.documents import Document

from config import get_settings


def _prompt_analyse_image_btp(
    image_path: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str = "inconnu",
) -> str:
    return f"""
Tu es un expert BTP, conducteur de travaux et controleur qualite/securite.
Analyse cette photo de chantier en francais, de facon factuelle et operationnelle.

Contexte:
- Projet: {projet}
- Lot technique: {lot_technique}
- Criticite declaree: {criticite}
- Auteur: {auteur}
- Fichier: {Path(image_path).name}

Retourne une analyse structuree avec ces rubriques:
1. Resume visuel de la scene.
2. Elements structurels visibles: murs, poteaux, poutres, planchers, fondations, coffrage, ferraillage, reseaux.
3. Materiaux, equipements et engins visibles.
4. EPI et securite: casques, gilets, chaussures, harnais, balisage, protections collectives.
5. Non-conformites ou points de vigilance possibles.
6. Risques chantier: chute, heurt, electrique, incendie, manutention, instabilite, coactivite.
7. Actions recommandees pour le chef de chantier.

Si un element n'est pas clairement visible, indique "non visible" au lieu d'inventer.
""".strip()


def _prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _prepare_image_for_ocr(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = ImageOps.autocontrast(image, cutoff=2)
    image = image.filter(ImageFilter.SHARPEN)

    width, height = image.size
    if max(width, height) < 1000:
        scale = 1000 / max(width, height)
        image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return image


def _get_pytesseract():
    settings = get_settings()
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("pytesseract est requis pour l'OCR image. Installe `pytesseract`.") from exc

    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    return pytesseract


def extraire_texte_image_ocr(image: Image.Image) -> str:
    pytesseract = _get_pytesseract()
    try:
        prepared = _prepare_image_for_ocr(image)
        return pytesseract.image_to_string(prepared, lang="fra+ara+eng").strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            f"Tesseract OCR est introuvable a l'emplacement configure : {get_settings().tesseract_cmd}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Erreur OCR Tesseract image : {exc}") from exc


@lru_cache(maxsize=1)
def _get_blip_components():
    settings = get_settings()
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except ImportError as exc:
        raise RuntimeError("BLIP requiert `torch` et `transformers`. Installe les dependances du projet.") from exc

    try:
        model = BlipForConditionalGeneration.from_pretrained(settings.blip_model_name)
        processor = BlipProcessor.from_pretrained(settings.blip_model_name)
        model.eval()
        return model, processor, torch
    except Exception as exc:
        raise RuntimeError(f"Impossible de charger BLIP '{settings.blip_model_name}' : {exc}") from exc


def generer_description_blip(image: Image.Image) -> Optional[str]:
    """Genere une courte description visuelle. BLIP reste optionnel."""
    try:
        model, processor, torch = _get_blip_components()
        image = _prepare_image(image)
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=60)
        description = processor.decode(output[0], skip_special_tokens=True).strip()
        return description or None
    except Exception:
        return None


def _is_gemini_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "quota",
            "rate limit",
            "rate_limit",
            "resource_exhausted",
            "429",
        )
    )


def analyser_image_openai(
    image_path: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str = "inconnu",
) -> str:
    """Analyse une photo de chantier avec le modele OpenAI-compatible configure."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "Cle OpenAI manquante. Ajoute OPENAI_API_KEY dans le fichier .env "
            "ou passe VISION_BACKEND=gemini/clip temporairement."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai est requis pour VISION_BACKEND=openai. "
            "Installe les dependances avec `pip install -r requirements.txt`."
        ) from exc

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("ascii")

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    prompt = _prompt_analyse_image_btp(image_path, projet, lot_technique, criticite, auteur)

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un expert BTP. Reponds en francais avec une analyse "
                        "visuelle concise, factuelle et exploitable."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                            },
                        },
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=min(settings.llm_max_tokens, 1200),
        )
        description = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise RuntimeError(
            f"Erreur analyse OpenAI image avec le modele '{settings.llm_model}' : {exc}"
        ) from exc

    if not description:
        raise RuntimeError("OpenAI n'a retourne aucune description exploitable pour cette image.")
    return description


def analyser_image_gemini(
    image_path: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str = "inconnu",
) -> str:
    """Analyse une photo de chantier avec Gemini Flash et retourne une description BTP."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "Cle Gemini manquante. Ajoute GEMINI_API_KEY dans le fichier .env "
            "ou passe VISION_BACKEND=clip pour utiliser le fallback BLIP/CLIP."
        )

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai est requis pour VISION_BACKEND=gemini. "
            "Installe les dependances avec `pip install -r requirements.txt`."
        ) from exc

    prompt = _prompt_analyse_image_btp(image_path, projet, lot_technique, criticite, auteur)

    try:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        with Image.open(image_path) as image:
            image.load()
            image = _prepare_image(image.copy())
            response = model.generate_content(
                [prompt, image],
                generation_config={"temperature": 0.2},
            )
        description = (getattr(response, "text", None) or "").strip()
    except Exception as exc:
        if _is_gemini_quota_error(exc):
            raise RuntimeError(
                "Quota Gemini depasse ou limite de debit atteinte. "
                "Reessaie plus tard, verifie la facturation Google AI Studio, "
                "ou passe VISION_BACKEND=clip temporairement."
            ) from exc
        raise RuntimeError(f"Erreur analyse Gemini image : {exc}") from exc

    if not description:
        raise RuntimeError("Gemini n'a retourne aucune description exploitable pour cette image.")
    return description


def _metadata_base(
    fichier_path: str,
    type_document: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str = "inconnu",
    fichier_original: Optional[str] = None,
) -> dict:
    fichier_nom = fichier_original or Path(fichier_path).name
    return {
        "source": fichier_nom,
        "source_fichier": fichier_nom,
        "fichier_original": fichier_nom,
        "type_document": type_document,
        "projet": projet,
        "lot_technique": lot_technique,
        "criticite": criticite,
        "auteur": auteur,
        "date": datetime.today().strftime("%Y-%m-%d"),
        "ingere_le": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "fichier_nom": fichier_nom,
        "nom_fichier": fichier_nom,
    }


def extraire_image_ocr(
    fichier_path: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str = "inconnu",
    fichier_original: Optional[str] = None,
) -> List[Document]:
    """Pipeline 3: Tesseract OCR on image."""
    try:
        with Image.open(fichier_path) as image:
            texte = extraire_texte_image_ocr(image)
            description_blip = generer_description_blip(image)
    except Exception as exc:
        raise RuntimeError(f"Erreur extraction OCR image : {exc}") from exc

    contenu = texte
    if description_blip:
        contenu = f"[Description visuelle : {description_blip}]\n\n{texte}"

    return [
        Document(
            page_content=contenu,
            metadata=_metadata_base(
                fichier_path,
                "image_ocr",
                projet,
                lot_technique,
                criticite,
                auteur,
                fichier_original,
            )
            | {
                "ocr_engine": "tesseract",
                "description_blip": description_blip or "",
                "texte_ocr_chars": len(texte),
            },
        )
    ]


@lru_cache(maxsize=1)
def _get_clip_components():
    settings = get_settings()
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "CLIP requiert `torch` et `transformers`. Installe les dependances du projet."
        ) from exc

    try:
        model = CLIPModel.from_pretrained(settings.clip_model_name)
        processor = CLIPProcessor.from_pretrained(settings.clip_model_name)
        model.eval()
        return model, processor, torch
    except Exception as exc:
        raise RuntimeError(f"Impossible de charger le modele CLIP '{settings.clip_model_name}' : {exc}") from exc


def encoder_image_clip(image: Image.Image) -> Optional[list]:
    """Encode une image avec CLIP. Retourne None si l'encodage echoue."""
    try:
        model, processor, torch = _get_clip_components()
        image = _prepare_image(image)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().astype(float).tolist()
    except Exception:
        return None


def _stats_collection_clip(collection) -> dict:
    return {
        "collection": "btp_images_clip",
        "nombre_vecteurs": collection.count(),
        "persist_dir": get_settings().chroma_persist_dir,
    }


def ingerer_image(
    fichier_path: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str = "inconnu",
    fichier_original: Optional[str] = None,
) -> dict:
    """
    Pipeline image unifie pour les traitements longs en arriere-plan.
    Respecte la logique du projet :
    - image avec texte riche -> Documents LangChain + nettoyage/vectorisation principale
    - image sans texte riche -> OpenAI/Gemini texte dans la collection principale, ou BLIP/CLIP en fallback
    """
    settings = get_settings()

    try:
        with Image.open(fichier_path) as image:
            image.load()
            image = image.copy()
            texte_ocr = extraire_texte_image_ocr(image)
    except Exception:
        texte_ocr = ""

    if len(texte_ocr.strip()) > settings.ocr_image_text_threshold:
        from couche_data.nettoyage import nettoyer
        from couche_data.vectorisation import stats_collection, vectoriser

        docs = extraire_image_ocr(fichier_path, projet, lot_technique, criticite, auteur, fichier_original)
        docs = nettoyer(docs)
        if not docs:
            return {
                "statut": "erreur",
                "pipeline_utilise": "image_ocr",
                "message": "Document image vide ou trop court apres nettoyage.",
                "fichier": fichier_original or Path(fichier_path).name,
            }

        vectoriser(docs)
        return {
            "statut": "succes",
            "pipeline_utilise": "image_ocr",
            "fichier": fichier_original or Path(fichier_path).name,
            "documents_ingeres": len(docs),
            "projet": projet,
            "texte_ocr_detecte": True,
            "description_blip": docs[0].metadata.get("description_blip", ""),
            "clip_encode": False,
            "collection": settings.chroma_collection_name,
            "stats_collection": stats_collection(),
        }

    return extraire_image_clip(fichier_path, projet, lot_technique, criticite, auteur, fichier_original)


def extraire_image_clip(
    fichier_path: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str = "inconnu",
    fichier_original: Optional[str] = None,
) -> dict:
    """Pipeline 4: OpenAI/Gemini vision for raw photos, with BLIP/CLIP fallback."""
    settings = get_settings()
    vision_backend = settings.vision_backend.lower()
    if vision_backend in ("openai", "gpt", "llm"):
        from couche_data.nettoyage import nettoyer
        from couche_data.vectorisation import stats_collection, vectoriser

        description = analyser_image_openai(
            image_path=fichier_path,
            projet=projet,
            lot_technique=lot_technique,
            criticite=criticite,
            auteur=auteur,
        )
        metadata = _metadata_base(
            fichier_path,
            "image_openai",
            projet,
            lot_technique,
            criticite,
            auteur,
            fichier_original,
        )
        metadata["description"] = description
        metadata["vision_backend"] = "openai"
        metadata["vision_model"] = settings.llm_model
        metadata["texte_ocr_detecte"] = False

        docs = nettoyer([Document(page_content=description, metadata=metadata)])
        if not docs:
            return {
                "statut": "erreur",
                "pipeline_utilise": "image_openai",
                "message": "Description OpenAI vide ou trop courte apres nettoyage.",
                "fichier": fichier_original or Path(fichier_path).name,
            }

        vectoriser(docs)
        return {
            "statut": "succes",
            "pipeline_utilise": "image_openai",
            "fichier": fichier_original or Path(fichier_path).name,
            "documents_ingeres": len(docs),
            "projet": projet,
            "description_openai": description,
            "clip_encode": False,
            "collection": settings.chroma_collection_name,
            "stats_collection": stats_collection(),
        }

    if vision_backend == "gemini":
        from couche_data.nettoyage import nettoyer
        from couche_data.vectorisation import stats_collection, vectoriser

        description = analyser_image_gemini(
            image_path=fichier_path,
            projet=projet,
            lot_technique=lot_technique,
            criticite=criticite,
            auteur=auteur,
        )
        metadata = _metadata_base(
            fichier_path,
            "image_gemini",
            projet,
            lot_technique,
            criticite,
            auteur,
            fichier_original,
        )
        metadata["description"] = description
        metadata["vision_backend"] = "gemini"
        metadata["gemini_model"] = "gemini-2.5-flash"
        metadata["texte_ocr_detecte"] = False

        docs = nettoyer([Document(page_content=description, metadata=metadata)])
        if not docs:
            return {
                "statut": "erreur",
                "pipeline_utilise": "image_gemini",
                "message": "Description Gemini vide ou trop courte apres nettoyage.",
                "fichier": fichier_original or Path(fichier_path).name,
            }

        vectoriser(docs)
        return {
            "statut": "succes",
            "pipeline_utilise": "image_gemini",
            "fichier": fichier_original or Path(fichier_path).name,
            "documents_ingeres": len(docs),
            "projet": projet,
            "description_gemini": description,
            "clip_encode": False,
            "collection": settings.chroma_collection_name,
            "stats_collection": stats_collection(),
        }

    try:
        with Image.open(fichier_path) as image:
            image.load()
            image = image.copy()
            embedding = encoder_image_clip(image)
            description_blip = generer_description_blip(image)
    except Exception as exc:
        raise RuntimeError(f"Erreur traitement image : {exc}") from exc

    client = chromadb.PersistentClient(path=get_settings().chroma_persist_dir)
    metadata = _metadata_base(
        fichier_path,
        "image_clip",
        projet,
        lot_technique,
        criticite,
        auteur,
        fichier_original,
    )
    description = description_blip or (
        "Photo de chantier BTP sans texte OCR detecte. "
        f"Fichier: {fichier_original or Path(fichier_path).name}. "
        f"Projet: {projet}. Lot technique: {lot_technique}. "
        "Image a analyser comme element visuel de suivi chantier, qualite, securite ou avancement."
    )
    metadata["description"] = description
    metadata["description_blip"] = description_blip or ""
    metadata["clip_model"] = get_settings().clip_model_name
    metadata["clip_disponible"] = embedding is not None

    if embedding is not None:
        collection = client.get_or_create_collection(
            name="btp_images_clip",
            metadata={"hnsw:space": "cosine"},
        )
        identifiant = f"clip_{Path(fichier_path).stem}_{uuid.uuid4().hex}"
        collection.add(
            ids=[identifiant],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[description],
        )
        collection_name = "btp_images_clip"
        stats = _stats_collection_clip(collection)
    else:
        collection_name = get_settings().chroma_collection_name
        stats = {}

    from couche_data.nettoyage import nettoyer
    from couche_data.vectorisation import stats_collection, vectoriser

    text_metadata = dict(metadata)
    text_metadata["type_document"] = "image_blip" if description_blip else "image_description"
    docs = nettoyer([Document(page_content=description, metadata=text_metadata)])
    if docs:
        vectoriser(docs)
    main_stats = stats_collection()
    if collection_name == "btp_images_clip":
        stats = {"clip_collection": stats, "text_collection": main_stats}
    else:
        stats = main_stats

    return {
        "statut": "succes",
        "pipeline_utilise": "image_clip" if embedding is not None else "image_blip",
        "fichier": fichier_original or Path(fichier_path).name,
        "documents_ingeres": 1,
        "projet": projet,
        "description_blip": description_blip,
        "clip_encode": embedding is not None,
        "collection": collection_name,
        "stats_collection": stats,
    }
