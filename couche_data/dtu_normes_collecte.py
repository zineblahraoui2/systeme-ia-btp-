from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from couche_data.image_collecte import extraire_texte_image_ocr
from couche_data.vectorisation import get_embeddings, get_vectorstore, indexer, stats_collection


logger = logging.getLogger(__name__)


REGLEMENTAIRE_NAME_RE = re.compile(r"\b(DTU|NF|EN|ISO|EUROCODE|EUROCODE\s*\d+|NF\s*EN)\b", re.I)
NUMERO_RE = re.compile(
    r"\b((?:NF\s*)?(?:DTU|EN|ISO|NF\s*EN|EUROCODE)\s*[A-Z]*\s*\d+(?:[-\.]\d+)*(?:\s*[-:]\s*\d+)?)\b",
    re.I,
)
SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})\s+(.{3,160})\s*$")
ARTICLE_RE = re.compile(r"^\s*(?:Article\s+)?(\d+(?:\.\d+){1,6})\s*[-–:]?\s*(.{0,180})\s*$", re.I)
TABLE_RE = re.compile(r"\b(tableau|table)\s+\d+", re.I)

OBLIGATOIRE_RE = re.compile(r"\b(doit|doivent|obligatoire|est tenu|sera conforme|doit être|doit etre)\b", re.I)
RECOMMANDE_RE = re.compile(r"\b(recommandé|recommande|il convient|devrait|peut être|peut etre)\b", re.I)
INTERDIT_RE = re.compile(r"\b(interdit|ne doit pas|proscri|à proscrire|a proscrire)\b", re.I)
SECURITE_RE = re.compile(r"\b(sécurité|securite|incendie|feu|évacuation|evacuation|chute|danger|critique)\b", re.I)
TOLERANCE_RE = re.compile(r"\b(tolérance|tolerance|écart|ecart|minimum|maximal|maximale|minimale|au moins|inférieur|inferieur|supérieur|superieur)\b", re.I)
DOMAINES = {
    "maconnerie": ["maçonnerie", "maconnerie", "mur", "murs", "bloc", "brique"],
    "beton": ["béton", "beton", "armature", "ferraillage", "dalle", "poteau", "poutre"],
    "thermique": ["thermique", "isolation", "pont thermique", "u ="],
    "acoustique": ["acoustique", "bruit", "affaiblissement"],
    "securite_incendie": ["incendie", "feu", "résistance au feu", "resistance au feu"],
    "bim": ["bim", "ifc", "iso 19650"],
}


def est_document_reglementaire(nom_fichier: str, extrait: str = "") -> bool:
    cible = f"{nom_fichier}\n{extrait[:500]}".replace("_", " ").replace("-", " ")
    return bool(REGLEMENTAIRE_NAME_RE.search(cible) or NUMERO_RE.search(cible))


CATALOGUE_ARTICLE_SECTION = "N/A — consulter le texte complet"


def detect_dtu_doc_type(text: str) -> str:
    """
    Distingue 3 types de documents reglementaires :
    - 'catalogue_index'  : liste/index de DTU (ex: catalogue FFB/BNTEC)
    - 'dtu_complet'      : DTU complet avec articles, sections, prescriptions
    - 'dtu_partiel'      : document reglementaire partiel ou mixte
    """
    catalogue_signals = [
        "Pour les métiers du Bâtiment",
        "Normes et fascicules de documentation",
        "BNTEC",
        "disponibles notamment auprès de",
        "NF DTU publié avec un ou plusieurs amendements",
    ]
    article_signals = [
        "Article ",
        "Domaine d'application",
        "Prescriptions techniques",
        "Travaux concernés",
        "Conditions de mise en œuvre",
        "§",
    ]

    normalized = text or ""
    catalogue_score = sum(1 for signal in catalogue_signals if signal in normalized)
    article_score = sum(1 for signal in article_signals if signal in normalized)

    if catalogue_score >= 2 and article_score == 0:
        return "catalogue_index"
    if article_score >= 2:
        return "dtu_complet"
    return "dtu_partiel"


def extract_dtu_catalogue(text: str) -> list[dict]:
    """
    Extrait la liste structuree des DTU depuis un document de type catalogue/index.
    Retourne une liste de dicts, un par DTU reference.
    """
    patterns = [
        # FFB/BNTEC catalogue: "NF DTU 25.1 Enduits interieurs en platre (P71-201)"
        r"\b((?:(?:NF|XP|FD)\s+)?DTU\s+(\d+(?:\.\d+)*\*?))\s+(.+?)(?:\s*\(([A-Z]{0,3}\s?\d{1,3}[-–]\d{1,4}(?:[-–]\d+)?)\))?\s*$",
        # Legacy format with a full NF/XP/FD DTU prefix.
        r"\b((?:NF|XP|FD)\s+DTU\s+([\d.]+\*?))\s+([^\n(]{5,}?)(?:\s*\(([^)]+)\))?\s*$",
    ]
    results: list[dict] = []
    current_domain = "Général"
    seen_normes: set[str] = set()

    domain_pattern = re.compile(
        r"^[A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ][A-ZÀÂÉÈÊËÎÏÔÙÛÜÇa-zàâéèêëîïôùûüç\s\-–/]+$"
    )

    for raw_line in (text or "").split("\n"):
        line = " ".join(raw_line.strip().split())
        if not line:
            continue

        if (
            domain_pattern.match(line)
            and len(line) > 4
            and "NF DTU" not in line
            and "XP DTU" not in line
            and "FD DTU" not in line
        ):
            current_domain = line

        match = None
        for pattern in patterns:
            match = re.search(pattern, line, re.MULTILINE | re.IGNORECASE)
            if match:
                break
        if not match:
            continue

        norme = match.group(1).strip()
        if norme in seen_normes:
            continue
        seen_normes.add(norme)

        titre = match.group(3).strip(" -–:")
        ref_nf = match.group(4).strip() if len(match.groups()) >= 4 and match.group(4) else ""
        results.append(
            {
                "norme": norme,
                "titre": titre,
                "reference_nf": ref_nf,
                "domaine": current_domain,
                "type_doc": "catalogue_index",
                "article": CATALOGUE_ARTICLE_SECTION,
                "section": CATALOGUE_ARTICLE_SECTION,
                "prescription": (
                    f"La norme {norme} ({titre}) est référencée dans ce catalogue. "
                    f"Domaine : {current_domain}. "
                    f"Pour les prescriptions détaillées, articles et tolérances, "
                    f"se référer au texte intégral de {norme} disponible auprès "
                    f"du CSTB (www.cstb.fr) ou de l'AFNOR (www.afnor.fr)."
                ),
            }
        )

    return results


def _extraire_pdf_pages_avec_ocr(fichier_path: str) -> tuple[list[tuple[int, str]], str, bool]:
    pages = _extraire_pdf_texte(fichier_path)
    total_text = "\n".join(text for _, text in pages)
    ocr_utilise = False
    if len(total_text.strip()) < 300:
        pages = _extraire_pdf_ocr(fichier_path)
        total_text = "\n".join(text for _, text in pages)
        ocr_utilise = True

    if not total_text.strip():
        raise RuntimeError("Aucun texte detecte dans le PDF reglementaire.")
    return pages, total_text, ocr_utilise


def ingest_dtu_catalogue_to_chroma(
    dtu_entries: list[dict],
    collection=None,
    source_filename: str = "",
    source_date: str = "Janvier 2026",
) -> int:
    """
    Stocke chaque DTU d'un catalogue comme un document independant dans ChromaDB.
    Utilise upsert pour permettre la reingestion sans doublons.
    """
    if not dtu_entries:
        return 0

    vectorstore = collection or get_vectorstore()
    chroma_collection = getattr(vectorstore, "_collection", vectorstore)
    embeddings = get_embeddings()

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for entry in dtu_entries:
        content = (
            f"Norme: {entry['norme']}\n"
            f"Titre: {entry['titre']}\n"
            f"Domaine: {entry['domaine']}\n"
            f"Référence NF: {entry['reference_nf']}\n"
            f"Type document: Catalogue de référence\n"
            f"Prescription: {entry['prescription']}"
        )
        safe_norme = entry["norme"].replace(" ", "_").replace("*", "bis").replace("/", "_")
        safe_date = source_date.replace(" ", "_").replace("/", "_")
        doc_id = f"dtu_cat_{safe_norme}_{safe_date}"

        documents.append(content)
        metadatas.append(
            {
                "norme": entry["norme"],
                "numero_dtu": entry["norme"],
                "titre": entry["titre"],
                "domaine": entry["domaine"],
                "reference_nf": entry["reference_nf"],
                "type_doc": "catalogue_index",
                "article": CATALOGUE_ARTICLE_SECTION,
                "section": CATALOGUE_ARTICLE_SECTION,
                "source": source_filename,
                "source_fichier": source_filename,
                "edition": source_date,
                "type_document": "DTU",
                "criticite": "haute",
                "obligatoire": False,
                "type_prescription": "reference_catalogue",
                "page": "N/A",
                "score_confiance": 0.95,
                "projet": "reglementaire",
                "lot_technique": "reglementation",
                "ingere_le": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        ids.append(doc_id)

    batch_size = 100
    for start in range(0, len(documents), batch_size):
        batch_documents = documents[start : start + batch_size]
        batch_metadatas = metadatas[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        batch_embeddings = embeddings.embed_documents(batch_documents)
        chroma_collection.upsert(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas,
            embeddings=batch_embeddings,
        )

    return len(documents)


def _documents_catalogue_fallback(
    pages: list[tuple[int, str]],
    fichier_path: str,
    ocr_utilise: bool,
    projet: str = "reglementaire",
    lot_technique: str = "reglementation",
    criticite: str = "haute",
    auteur: str = "inconnu",
) -> tuple[list[Document], dict]:
    """
    Fallback anti-blocage : si un catalogue DTU est detecte mais non parse,
    on l'indexe quand meme comme document texte exploitable.
    """
    documents: list[Document] = []
    source_name = Path(fichier_path).name
    ingested_at = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    for page, texte in pages:
        text = (texte or "").strip()
        if not text:
            continue
        for index, start in enumerate(range(0, len(text), 3500), start=1):
            chunk_text = text[start : start + 3500].strip()
            if not chunk_text:
                continue
            metadata = {
                "type_document": "dtu_catalogue",
                "type_doc": "catalogue_index",
                "numero_dtu": "Catalogue DTU",
                "titre": source_name,
                "domaine": "catalogue",
                "criticite": criticite,
                "source": "externe_reglementaire",
                "source_fichier": source_name,
                "obligatoire": False,
                "type_prescription": "reference_catalogue",
                "article": CATALOGUE_ARTICLE_SECTION,
                "section": CATALOGUE_ARTICLE_SECTION,
                "page": int(page),
                "date_norme": "Non trouve",
                "version_norme": "Non trouve",
                "organisme": "BNTEC/FFB",
                "niveau_application": "national",
                "score_confiance": 0.75,
                "projet": projet,
                "lot_technique": lot_technique,
                "auteur": auteur,
                "nom_fichier": source_name,
                "chunk_reglementaire": f"fallback-{page}-{index}",
                "is_table": False,
                "ocr_utilise": ocr_utilise,
                "ingere_le": ingested_at,
            }
            content = (
                "Type document: Catalogue DTU brut\n"
                f"Source: {source_name}\n"
                f"Page: {page}\n"
                "Note: catalogue detecte mais entrees DTU non parsees automatiquement.\n"
                f"Texte catalogue:\n{chunk_text}"
            )
            documents.append(Document(page_content=content, metadata=metadata))

    summary = {
        "nom": "Catalogue DTU",
        "titre": source_name,
        "domaine": "catalogue",
        "type_document": "dtu_catalogue",
        "type_doc": "catalogue_index",
        "chunks": len(documents),
        "fallback": "texte_catalogue",
        "ocr_utilise": ocr_utilise,
        "organisme": "BNTEC/FFB",
        "date_norme": "Non trouve",
    }
    return documents, summary


def _extraire_pdf_texte(fichier_path: str) -> list[tuple[int, str]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF est requis pour lire les PDF réglementaires.") from exc

    pages: list[tuple[int, str]] = []
    with fitz.open(fichier_path) as pdf:
        for index, page in enumerate(pdf, start=1):
            pages.append((index, page.get_text("text").strip()))
    return pages


def _extraire_pdf_ocr(fichier_path: str) -> list[tuple[int, str]]:
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError("pdf2image est requis pour l'OCR PDF réglementaire.") from exc

    pages: list[tuple[int, str]] = []
    for index, image in enumerate(convert_from_path(fichier_path), start=1):
        pages.append((index, extraire_texte_image_ocr(image)))
    return pages


def _extraire_tables(fichier_path: str) -> dict[int, list[str]]:
    tables: dict[int, list[str]] = {}
    try:
        import pdfplumber
    except ImportError:
        return tables

    try:
        with pdfplumber.open(fichier_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables() or []:
                    rows = [" | ".join((cell or "").strip() for cell in row) for row in table if row]
                    table_text = "\n".join(row for row in rows if row.strip())
                    if table_text.strip():
                        tables.setdefault(page_index, []).append(table_text)
    except Exception:
        return tables
    return tables


def _numero_norme(texte: str, nom_fichier: str) -> str:
    match = NUMERO_RE.search(f"{nom_fichier}\n{texte[:3000]}")
    return match.group(1).upper().strip() if match else "Non trouvé"


def _type_document(numero: str, texte: str) -> str:
    cible = f"{numero} {texte[:1000]}".upper()
    if "DTU" in cible:
        return "DTU"
    return "NORME"


def _titre(texte: str, numero: str) -> str:
    lines = [line.strip() for line in texte.splitlines() if line.strip()]
    for line in lines[:30]:
        if numero != "Non trouvé" and numero.lower() in line.lower() and len(line) > len(numero) + 4:
            return line[:220]
    for line in lines[:20]:
        if 8 <= len(line) <= 220 and not line.lower().startswith(("page ", "sommaire")):
            return line
    return "Non trouvé"


def _domaine(texte: str) -> str:
    lower = texte.lower()
    for domaine, mots in DOMAINES.items():
        if any(mot in lower for mot in mots):
            return domaine
    return "Non trouvé"


def _date_norme(texte: str) -> str:
    match = re.search(r"\b(19|20)\d{2}\b", texte[:4000])
    return match.group(0) if match else "Non trouvé"


def _organisme(texte: str) -> str:
    upper = texte[:4000].upper()
    for organisme in ("AFNOR", "ISO", "CEN", "CSTB"):
        if organisme in upper:
            return organisme
    return "Non trouvé"


def _type_prescription(texte: str) -> str:
    if INTERDIT_RE.search(texte):
        return "interdit"
    if SECURITE_RE.search(texte):
        return "sécurité critique"
    if TOLERANCE_RE.search(texte):
        return "tolérance"
    if OBLIGATOIRE_RE.search(texte):
        return "obligatoire"
    if RECOMMANDE_RE.search(texte):
        return "recommandé"
    return "informatif"


def _score_confiance(texte: str, article: str, numero: str) -> float:
    score = 0.55
    if article != "Non trouvé":
        score += 0.15
    if numero != "Non trouvé":
        score += 0.1
    if any(regex.search(texte) for regex in (OBLIGATOIRE_RE, INTERDIT_RE, SECURITE_RE, TOLERANCE_RE)):
        score += 0.15
    if re.search(r"\d+\s*(cm|mm|m²|m2|mpa|kn|kg|°c|h)\b", texte, re.I):
        score += 0.05
    return min(round(score, 2), 0.98)


def _chunk_structurel(pages: list[tuple[int, str]], tables: dict[int, list[str]]) -> list[dict]:
    chunks: list[dict] = []
    current: Optional[dict] = None
    current_section = "Non trouvé"

    def flush() -> None:
        nonlocal current
        if current and current["texte"].strip():
            chunks.append(current)
        current = None

    for page, texte in pages:
        for raw_line in texte.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            article_match = ARTICLE_RE.match(line)
            section_match = SECTION_RE.match(line)
            if section_match:
                current_section = section_match.group(2).strip()

            if article_match:
                flush()
                current = {
                    "article": article_match.group(1),
                    "section": current_section,
                    "page": page,
                    "texte": line,
                    "is_table": False,
                }
                continue

            if current is None:
                current = {
                    "article": "Non trouvé",
                    "section": current_section,
                    "page": page,
                    "texte": line,
                    "is_table": False,
                }
            else:
                current["texte"] += "\n" + line

        for table_text in tables.get(page, []):
            flush()
            chunks.append(
                {
                    "article": "Tableau",
                    "section": current_section,
                    "page": page,
                    "texte": table_text,
                    "is_table": True,
                }
            )

    flush()
    return [chunk for chunk in chunks if len(chunk["texte"].strip()) >= 40]


def extraire_dtu_norme_pdf(
    fichier_path: str,
    projet: str = "reglementaire",
    lot_technique: str = "reglementation",
    criticite: str = "haute",
    auteur: str = "inconnu",
    pages: Optional[list[tuple[int, str]]] = None,
    total_text: Optional[str] = None,
    ocr_utilise: Optional[bool] = None,
    doc_type: Optional[str] = None,
) -> tuple[list[Document], dict]:
    if pages is None or total_text is None or ocr_utilise is None:
        pages, total_text, ocr_utilise = _extraire_pdf_pages_avec_ocr(fichier_path)
    doc_type = doc_type or detect_dtu_doc_type(total_text)
    logger.info("DTU ingestion: type document detecte=%s fichier=%s", doc_type, Path(fichier_path).name)

    tables = _extraire_tables(fichier_path)
    numero = _numero_norme(total_text, Path(fichier_path).name)
    type_document = _type_document(numero, total_text)
    titre = _titre(total_text, numero)
    domaine = _domaine(total_text)
    date_norme = _date_norme(total_text)
    organisme = _organisme(total_text)
    structural_chunks = _chunk_structurel(pages, tables)

    documents: list[Document] = []
    for index, chunk in enumerate(structural_chunks, start=1):
        prescription_type = _type_prescription(chunk["texte"])
        obligatoire = prescription_type in {"obligatoire", "sécurité critique", "tolérance"}
        metadata = {
            "type_document": type_document,
            "type_doc": doc_type,
            "numero_dtu": numero,
            "titre": titre,
            "domaine": domaine,
            "criticite": "haute",
            "source": "externe_reglementaire",
            "source_fichier": fichier_path,
            "obligatoire": obligatoire,
            "type_prescription": prescription_type,
            "article": chunk["article"],
            "section": chunk["section"],
            "page": int(chunk["page"]),
            "date_norme": date_norme,
            "version_norme": date_norme,
            "organisme": organisme,
            "niveau_application": "national" if organisme in {"AFNOR", "CSTB"} or type_document == "DTU" else "international",
            "score_confiance": _score_confiance(chunk["texte"], chunk["article"], numero),
            "projet": projet,
            "lot_technique": lot_technique,
            "auteur": auteur,
            "nom_fichier": Path(fichier_path).name,
            "chunk_reglementaire": index,
            "is_table": bool(chunk["is_table"] or TABLE_RE.search(chunk["texte"])),
            "ocr_utilise": ocr_utilise,
            "ingere_le": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        }
        content = (
            f"Norme: {numero}\n"
            f"Titre: {titre}\n"
            f"Article: {chunk['article']}\n"
            f"Section: {chunk['section']}\n"
            f"Page: {chunk['page']}\n"
            f"Type prescription: {prescription_type}\n"
            f"Citation réglementaire:\n{chunk['texte']}"
        )
        documents.append(Document(page_content=content, metadata=metadata))

    summary = {
        "nom": numero,
        "titre": titre,
        "domaine": domaine,
        "type_document": type_document,
        "type_doc": doc_type,
        "chunks": len(documents),
        "ocr_utilise": ocr_utilise,
        "organisme": organisme,
        "date_norme": date_norme,
    }
    return documents, summary


def ingerer_dtu_norme_pdf(
    fichier_path: str,
    projet: str = "reglementaire",
    lot_technique: str = "reglementation",
    criticite: str = "haute",
    auteur: str = "inconnu",
) -> dict:
    pages, total_text, ocr_utilise = _extraire_pdf_pages_avec_ocr(fichier_path)
    doc_type = detect_dtu_doc_type(total_text)
    logger.info("DTU ingestion: detection=%s fichier=%s", doc_type, Path(fichier_path).name)

    if doc_type == "catalogue_index":
        entries = extract_dtu_catalogue(total_text)
        logger.info("DTU ingestion: catalogue entries=%s fichier=%s", len(entries), Path(fichier_path).name)
        if not entries:
            logger.warning(
                "Catalogue DTU detecte sans entree parseable. Fallback texte brut. fichier=%s apercu=%r",
                Path(fichier_path).name,
                total_text[:1000],
            )
            fallback_docs, summary = _documents_catalogue_fallback(
                pages,
                fichier_path,
                ocr_utilise,
                projet=projet,
                lot_technique=lot_technique,
                criticite=criticite,
                auteur=auteur,
            )
            if not fallback_docs:
                return {
                    "statut": "erreur",
                    "status": "error",
                    "doc_type": "catalogue_index",
                    "message": "Catalogue DTU detecte mais aucun texte exploitable n'a ete extrait.",
                    "fichier": Path(fichier_path).name,
                }
            indexer(fallback_docs)
            return {
                "statut": "succès",
                "status": "success",
                "pipeline_utilise": "dtu_norme",
                "fichier": Path(fichier_path).name,
                "doc_type": "catalogue_index",
                "documents_ingeres": len(fallback_docs),
                "dtus_ingeres": 0,
                "fallback": "dtu_catalogue",
                "message": "Catalogue DTU detecte. Aucune entree structuree parsee, PDF ingere comme texte catalogue exploitable.",
                "resume_reglementaire": summary,
                "stats_collection": stats_collection(),
            }

        count = ingest_dtu_catalogue_to_chroma(
            dtu_entries=entries,
            collection=get_vectorstore(),
            source_filename=Path(fichier_path).name,
        )
        logger.info("DTU ingestion: catalogue upsert=%s fichier=%s", count, Path(fichier_path).name)
        summary = {
            "nom": "Catalogue DTU",
            "titre": Path(fichier_path).name,
            "domaine": "catalogue",
            "type_document": "DTU",
            "type_doc": "catalogue_index",
            "chunks": count,
            "dtus_ingeres": count,
            "ocr_utilise": ocr_utilise,
            "organisme": "BNTEC/FFB" if "BNTEC" in total_text[:4000].upper() else "Non trouve",
            "date_norme": "Janvier 2026" if "2026" in total_text[:4000] else "Non trouve",
        }
        return {
            "statut": "succès",
            "status": "success",
            "pipeline_utilise": "dtu_norme",
            "fichier": Path(fichier_path).name,
            "doc_type": "catalogue_index",
            "documents_ingeres": count,
            "dtus_ingeres": count,
            "message": f"{count} DTU extraits et indexes depuis le catalogue.",
            "resume_reglementaire": summary,
            "stats_collection": stats_collection(),
        }

    documents, summary = extraire_dtu_norme_pdf(
        fichier_path,
        projet,
        lot_technique,
        criticite,
        auteur,
        pages=pages,
        total_text=total_text,
        ocr_utilise=ocr_utilise,
        doc_type=doc_type,
    )
    if not documents:
        return {"statut": "erreur", "message": "Aucun chunk réglementaire extrait.", "fichier": Path(fichier_path).name}
    indexer(documents)
    logger.info("DTU ingestion: chunks indexes=%s fichier=%s", len(documents), Path(fichier_path).name)
    return {
        "statut": "succès",
        "pipeline_utilise": "dtu_norme",
        "fichier": Path(fichier_path).name,
        "documents_ingeres": len(documents),
        "resume_reglementaire": summary,
        "stats_collection": stats_collection(),
    }


def ingest_regulatory_document(file_path: str, collection=None) -> dict:
    """
    Point d'entree explicite du pipeline reglementaire.
    Detecte le type de DTU avant extraction pour eviter de traiter un catalogue comme un DTU complet.
    """
    pages, total_text, ocr_utilise = _extraire_pdf_pages_avec_ocr(file_path)
    doc_type = detect_dtu_doc_type(total_text)
    source_filename = Path(file_path).name
    logger.info("DTU ingestion: orchestration type=%s fichier=%s", doc_type, source_filename)

    if doc_type == "catalogue_index":
        entries = extract_dtu_catalogue(total_text)
        if not entries:
            logger.warning(
                "DTU ingestion: orchestration catalogue sans entree. Fallback texte brut fichier=%s apercu=%r",
                source_filename,
                total_text[:1000],
            )
            fallback_docs, summary = _documents_catalogue_fallback(pages, file_path, ocr_utilise)
            if not fallback_docs:
                return {
                    "status": "error",
                    "statut": "erreur",
                    "doc_type": "catalogue_index",
                    "message": "Catalogue DTU detecte mais aucun texte exploitable n'a ete extrait.",
                }
            indexer(fallback_docs)
            return {
                "status": "success",
                "statut": "succès",
                "doc_type": "catalogue_index",
                "documents_ingeres": len(fallback_docs),
                "dtus_ingeres": 0,
                "fallback": "dtu_catalogue",
                "message": "Catalogue DTU detecte. PDF ingere comme texte catalogue exploitable.",
                "resume_reglementaire": summary,
            }
        count = ingest_dtu_catalogue_to_chroma(
            dtu_entries=entries,
            collection=collection or get_vectorstore(),
            source_filename=source_filename,
        )
        logger.info("DTU ingestion: orchestration catalogue count=%s fichier=%s", count, source_filename)
        return {
            "status": "success",
            "statut": "succès",
            "doc_type": "catalogue_index",
            "dtus_ingeres": count,
            "documents_ingeres": count,
            "message": f"{count} DTU extraits et indexes depuis le catalogue.",
        }

    documents, summary = extraire_dtu_norme_pdf(
        file_path,
        pages=pages,
        total_text=total_text,
        ocr_utilise=ocr_utilise,
        doc_type=doc_type,
    )
    if not documents:
        return {
            "status": "error",
            "statut": "erreur",
            "doc_type": doc_type,
            "message": "Aucun chunk reglementaire extrait.",
        }
    indexer(documents)
    logger.info("DTU ingestion: orchestration chunks=%s fichier=%s", len(documents), source_filename)
    return {
        "status": "success",
        "statut": "succès",
        "doc_type": doc_type,
        "documents_ingeres": len(documents),
        "resume_reglementaire": summary,
    }
