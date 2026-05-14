"""
couche_data/collecte.py
-----------------------
Collecte et chargement des documents depuis différentes sources :
- Fichiers locaux (PDF, DOCX, TXT, MD)
- Dossiers entiers
- Texte brut (emails, WhatsApp, notes terrain)

Chaque document est enrichi de métadonnées BTP :
projet, lot_technique, type_document, source, auteur, date, criticite.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from PIL import Image, ImageOps
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document

from config import get_settings

settings = get_settings()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ─────────────────────────────────────────────
# Métadonnées BTP par défaut
# ─────────────────────────────────────────────

def build_metadata(
    source: str,
    projet: str = "non_défini",
    lot_technique: str = "non_défini",
    type_document: str = "general",
    auteur: str = "inconnu",
    criticite: str = "normale",
    date: Optional[str] = None,
) -> dict:
    """Construit le dictionnaire de métadonnées enrichi pour un document BTP."""
    return {
        "source": source,
        "projet": projet,
        "lot_technique": lot_technique,
        "type_document": type_document,
        "auteur": auteur,
        "criticite": criticite,
        "date": date or datetime.today().strftime("%Y-%m-%d"),
        "ingere_le": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────
# Chargeurs par type de fichier
# ─────────────────────────────────────────────

def charger_pdf(chemin: str, metadata_extra: Optional[dict] = None) -> list[Document]:
    """Charge un fichier PDF page par page."""
    loader = PyPDFLoader(chemin)
    docs = loader.load()
    metadata_extra = metadata_extra or {}

    if not any(doc.page_content.strip() for doc in docs):
        docs_ocr = charger_pdf_ocr(chemin, metadata_extra=metadata_extra)
        if docs_ocr:
            return docs_ocr

    for doc in docs:
        doc.metadata.update(build_metadata(source=chemin, type_document="pdf", **metadata_extra))
    return docs


def _get_pytesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "OCR indisponible : installe pytesseract avec `pip install pytesseract` "
            "et installe aussi Tesseract OCR sur Windows."
        ) from exc
    tesseract_cmd = shutil.which("tesseract")
    if not tesseract_cmd:
        for candidate in (
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ):
            if candidate.exists():
                tesseract_cmd = str(candidate)
                break
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    return pytesseract


def _preparer_image_ocr(image: Image.Image) -> Image.Image:
    """Prépare légèrement l'image pour améliorer la lecture OCR."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    return image


def extraire_texte_image(image: Image.Image) -> str:
    """Extrait le texte d'une image avec Tesseract OCR."""
    pytesseract = _get_pytesseract()
    image = _preparer_image_ocr(image)
    try:
        return pytesseract.image_to_string(image, lang="fra+eng").strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR est introuvable. Installe Tesseract OCR et ajoute-le au PATH Windows."
        ) from exc


def charger_image_ocr(chemin: str, metadata_extra: Optional[dict] = None) -> list[Document]:
    """Charge une photo de chantier ou un scan image via OCR."""
    metadata_extra = metadata_extra or {}
    with Image.open(chemin) as image:
        texte = extraire_texte_image(image)

    doc = Document(
        page_content=texte,
        metadata=build_metadata(
            source=chemin,
            type_document="image_ocr",
            **metadata_extra,
        )
        | {
            "ocr_engine": "tesseract",
            "nom_fichier": Path(chemin).name,
        },
    )
    return [doc]


def charger_pdf_ocr(chemin: str, metadata_extra: Optional[dict] = None) -> list[Document]:
    """Fallback OCR pour PDF scanné lorsque l'extraction texte classique est vide."""
    metadata_extra = metadata_extra or {}
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "PDF scanné détecté, mais pypdfium2 n'est pas installé. "
            "Installe `pypdfium2` pour convertir les pages PDF en images OCR."
        ) from exc

    documents: list[Document] = []
    pdf = pdfium.PdfDocument(chemin)
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            image = page.render(scale=2).to_pil()
            texte = extraire_texte_image(image)
            documents.append(
                Document(
                    page_content=texte,
                    metadata=build_metadata(
                        source=chemin,
                        type_document="pdf_ocr",
                        **metadata_extra,
                    )
                    | {
                        "page": page_index,
                        "ocr_engine": "tesseract",
                        "nom_fichier": Path(chemin).name,
                    },
                )
            )
    finally:
        pdf.close()
    return documents


def charger_docx(chemin: str, metadata_extra: Optional[dict] = None) -> list[Document]:
    """Charge un fichier Word (.docx)."""
    loader = Docx2txtLoader(chemin)
    docs = loader.load()
    metadata_extra = metadata_extra or {}
    for doc in docs:
        doc.metadata.update(build_metadata(source=chemin, type_document="docx", **metadata_extra))
    return docs


def charger_texte(chemin: str, metadata_extra: Optional[dict] = None) -> list[Document]:
    """Charge un fichier texte brut (.txt, .md)."""
    loader = TextLoader(chemin, encoding="utf-8")
    docs = loader.load()
    metadata_extra = metadata_extra or {}
    for doc in docs:
        doc.metadata.update(build_metadata(source=chemin, type_document="texte", **metadata_extra))
    return docs


def charger_texte_brut(
    contenu: str,
    metadata_extra: Optional[dict] = None,
    source: str = "saisie_manuelle",
) -> list[Document]:
    """
    Crée un Document LangChain depuis une chaîne de texte.
    Utile pour les emails, messages WhatsApp, notes terrain.
    """
    metadata_extra = metadata_extra or {}
    doc = Document(
        page_content=contenu,
        metadata=build_metadata(source=source, **metadata_extra),
    )
    return [doc]


def charger_dossier(
    dossier: str,
    glob: str = "**/*.{pdf,docx,txt,md,jpg,jpeg,png,tif,tiff,bmp,webp}",
    metadata_extra: Optional[dict] = None,
) -> list[Document]:
    """
    Charge récursivement tous les documents supportés d'un dossier.
    Supporte : PDF, DOCX, TXT, MD et images OCR.
    """
    tous_les_docs: list[Document] = []
    metadata_extra = metadata_extra or {}

    for extension, loader_cls in [
        ("**/*.pdf", PyPDFLoader),
        ("**/*.docx", Docx2txtLoader),
        ("**/*.txt", TextLoader),
        ("**/*.md", TextLoader),
    ]:
        loader = DirectoryLoader(
            dossier,
            glob=extension,
            loader_cls=loader_cls,
            show_progress=True,
            silent_errors=True,
        )
        try:
            docs = loader.load()
            for doc in docs:
                ext = Path(doc.metadata.get("source", "")).suffix.lstrip(".")
                doc.metadata.update(
                    build_metadata(
                        source=doc.metadata.get("source", dossier),
                        type_document=ext or "inconnu",
                        **metadata_extra,
                    )
                )
            tous_les_docs.extend(docs)
        except Exception as e:
            print(f"[collecte] Erreur chargement {extension} dans {dossier}: {e}")

    for extension in IMAGE_EXTENSIONS:
        for chemin_image in Path(dossier).rglob(f"*{extension}"):
            try:
                tous_les_docs.extend(charger_image_ocr(str(chemin_image), metadata_extra))
            except Exception as e:
                print(f"[collecte] Erreur OCR image {chemin_image}: {e}")

    print(f"[collecte] {len(tous_les_docs)} documents chargés depuis '{dossier}'")
    return tous_les_docs


# ─────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────

def collecter_depuis_fichier(chemin: str, metadata_extra: Optional[dict] = None) -> list[Document]:
    """
    Détecte automatiquement le type de fichier et charge les documents.
    """
    metadata_extra = metadata_extra or {}
    ext = Path(chemin).suffix.lower()
    dispatch = {
        ".pdf": charger_pdf,
        ".docx": charger_docx,
        ".txt": charger_texte,
        ".md": charger_texte,
        **{ext: charger_image_ocr for ext in IMAGE_EXTENSIONS},
    }
    fn = dispatch.get(ext)
    if fn is None:
        raise ValueError(f"Format non supporté : '{ext}'. Formats acceptés : {list(dispatch.keys())}")
    return fn(chemin, metadata_extra)
