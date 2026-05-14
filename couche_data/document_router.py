from __future__ import annotations

from pathlib import Path

from PIL import Image

from config import get_settings
from couche_data.dtu_normes_collecte import est_document_reglementaire
from couche_data.image_collecte import extraire_texte_image_ocr


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}


def _extraire_texte_pdf_pymupdf(fichier_path: str) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF est requis pour router les PDF. Installe `pymupdf`.") from exc

    try:
        texte_pages: list[str] = []
        with fitz.open(fichier_path) as pdf:
            for page in pdf:
                texte_pages.append(page.get_text("text"))
        return "\n".join(texte_pages)
    except Exception as exc:
        raise RuntimeError(f"Impossible d'analyser le PDF avec PyMuPDF : {exc}") from exc


def router_document(fichier_path: str) -> str:
    """
    Returns pipeline type:
    'pdf_texte' | 'pdf_ocr' | 'image_ocr' | 'image_clip' | 'bim_ifc'
    """
    settings = get_settings()
    path = Path(fichier_path)
    extension = path.suffix.lower()

    try:
        if extension == ".pdf":
            texte = _extraire_texte_pdf_pymupdf(fichier_path)
            if est_document_reglementaire(path.name, texte[:500]):
                return "dtu_norme"
            if len(texte.strip()) > settings.ocr_pdf_text_threshold:
                return "pdf_texte"
            return "pdf_ocr"

        if extension == ".ifc":
            return "bim_ifc"

        if extension in IMAGE_EXTENSIONS:
            with Image.open(fichier_path) as image:
                texte = extraire_texte_image_ocr(image)
            if len(texte.strip()) > settings.ocr_image_text_threshold:
                return "image_ocr"
            return "image_clip"
    except Exception:
        raise

    raise ValueError(
        f"Format non supporte par le routeur intelligent : '{extension}'. "
        "Formats acceptes : PDF, IFC, JPG, JPEG, PNG, WEBP, TIFF, BMP."
    )
