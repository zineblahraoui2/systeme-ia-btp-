from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from config import get_settings
from couche_data.image_collecte import extraire_texte_image_ocr


def _metadata_base(
    fichier_path: str,
    page_number: int,
    type_document: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str = "inconnu",
) -> dict:
    return {
        "source": fichier_path,
        "page_number": page_number,
        "page": page_number - 1,
        "type_document": type_document,
        "projet": projet,
        "lot_technique": lot_technique,
        "criticite": criticite,
        "auteur": auteur,
        "date": datetime.today().strftime("%Y-%m-%d"),
        "ingere_le": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "nom_fichier": Path(fichier_path).name,
    }


def extraire_pdf_texte(
    fichier_path: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str = "inconnu",
) -> List[Document]:
    """Pipeline 1: direct PyMuPDF extraction."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF est requis pour extraire les PDF texte. Installe `pymupdf`.") from exc

    try:
        documents: list[Document] = []
        with fitz.open(fichier_path) as pdf:
            for index, page in enumerate(pdf, start=1):
                texte = page.get_text("text").strip()
                documents.append(
                    Document(
                        page_content=texte,
                        metadata=_metadata_base(
                            fichier_path,
                            index,
                            "pdf_texte",
                            projet,
                            lot_technique,
                            criticite,
                            auteur,
                        ),
                    )
                )
        return documents
    except Exception as exc:
        raise RuntimeError(f"Erreur extraction PDF texte : {exc}") from exc


def extraire_pdf_ocr(
    fichier_path: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str = "inconnu",
) -> List[Document]:
    """Pipeline 2: Tesseract OCR on PDF pages."""
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError("pdf2image est requis pour l'OCR PDF. Installe `pdf2image`.") from exc

    settings = get_settings()
    try:
        images = convert_from_path(fichier_path)
    except Exception as exc:
        raise RuntimeError(
            "Impossible de convertir le PDF en images pour OCR. "
            "Verifie que Poppler est installe ou utilise un PDF texte."
        ) from exc

    documents: list[Document] = []
    for index, image in enumerate(images, start=1):
        try:
            texte = extraire_texte_image_ocr(image)
        except Exception as exc:
            raise RuntimeError(f"Erreur OCR sur la page PDF {index} : {exc}") from exc

        metadata = _metadata_base(
            fichier_path,
            index,
            "pdf_ocr",
            projet,
            lot_technique,
            criticite,
            auteur,
        )
        metadata["ocr_engine"] = "tesseract"
        metadata["tesseract_cmd"] = settings.tesseract_cmd
        documents.append(Document(page_content=texte, metadata=metadata))

    return documents
