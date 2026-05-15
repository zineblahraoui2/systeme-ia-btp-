"""
couche_execution/workflows.py
------------------------------
Workflows d'automatisation opérationnelle de bout en bout.
Chaque workflow orchestre les 3 couches du système :
Data → IA → Exécution

Workflows disponibles :
- ingerer_document()       : Ingest complet d'un fichier (collecte → nettoyage → vectorisation)
- ingerer_texte_brut()     : Ingest d'un texte (email, WhatsApp, note terrain)
- analyser_et_recommander(): Analyse d'une situation + recommandations en une seule étape
- audit_projet()           : Audit complet d'un projet (conformité + risques + alertes)
- ingerer_dossier()        : Ingest batch d'un dossier entier
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from couche_data.bim_collecte import extraire_bim_ifc
from couche_data.collecte import collecter_depuis_fichier, charger_texte_brut, charger_dossier
from couche_data.document_router import router_document
from couche_data.dtu_normes_collecte import ingerer_dtu_norme_pdf
from couche_data.gmail_collecte import collecter_gmail, generate_auth_url, has_valid_gmail_token
from couche_data.image_collecte import extraire_image_clip, extraire_image_ocr
from couche_data.nettoyage import nettoyer
from couche_data.pdf_collecte import extraire_pdf_ocr, extraire_pdf_texte
from couche_data.vectorisation import vectoriser, stats_collection
from couche_ia.analyse_metier import repondre, verifier_conformite, detecter_risques, analyser_document
from couche_execution.recommandations import (
    generer_recommandations,
    generer_alertes_critiques,
    rapport_conformite_projet,
    RapportRecommandations,
)


# ─────────────────────────────────────────────
# Workflows d'ingestion
# ─────────────────────────────────────────────

def ingerer_document(
    chemin: str,
    projet: str = "non_défini",
    lot_technique: str = "non_défini",
    auteur: str = "inconnu",
    criticite: str = "normale",
) -> dict:
    """
    Workflow complet d'ingestion d'un fichier dans la base vectorielle.
    Étapes : collecte → nettoyage → vectorisation

    Returns:
        dict avec statut, nombre de chunks indexés et métadonnées.
    """
    metadata = {
        "projet": projet,
        "lot_technique": lot_technique,
        "auteur": auteur,
        "criticite": criticite,
    }

    print(f"\n[workflow] Ingestion de : {chemin}")
    print(f"[workflow] Métadonnées : {metadata}")

    # 1. Collecte
    docs = collecter_depuis_fichier(chemin, metadata_extra=metadata)
    if not docs:
        return {"statut": "erreur", "message": "Aucun document collecté.", "fichier": chemin}

    # 2. Nettoyage
    docs = nettoyer(docs)

    # 3. Vectorisation
    vectorstore = vectoriser(docs)
    stats = stats_collection()

    return {
        "statut": "succès",
        "fichier": chemin,
        "projet": projet,
        "documents_collectes": len(docs),
        "stats_collection": stats,
    }


def ingerer_fichier(
    fichier_path: str,
    projet: str,
    lot_technique: str,
    criticite: str,
    auteur: str = "inconnu",
) -> dict:
    """
    Workflow d'ingestion fichier avec routage intelligent.
    """
    try:
        pipeline = router_document(fichier_path)
        resume_bim = None
        print(f"\n[workflow] Ingestion fichier intelligente | Pipeline : {pipeline}")

        if pipeline == "pdf_texte":
            docs = extraire_pdf_texte(fichier_path, projet, lot_technique, criticite, auteur)
        elif pipeline == "pdf_ocr":
            docs = extraire_pdf_ocr(fichier_path, projet, lot_technique, criticite, auteur)
        elif pipeline == "image_ocr":
            docs = extraire_image_ocr(fichier_path, projet, lot_technique, criticite, auteur)
        elif pipeline == "image_clip":
            return extraire_image_clip(fichier_path, projet, lot_technique, criticite, auteur)
        elif pipeline == "bim_ifc":
            docs, resume_bim = extraire_bim_ifc(fichier_path, projet, lot_technique, "haute", auteur)
        elif pipeline == "dtu_norme":
            return ingerer_dtu_norme_pdf(fichier_path, "reglementaire", "reglementation", "haute", auteur)
        else:
            return {
                "statut": "erreur",
                "message": f"Pipeline inconnu : {pipeline}",
                "fichier": Path(fichier_path).name,
            }

        if not docs:
            return {
                "statut": "erreur",
                "pipeline_utilise": pipeline,
                "message": "Aucun document extrait du fichier.",
                "fichier": Path(fichier_path).name,
            }

        docs = nettoyer(docs)
        if not docs:
            return {
                "statut": "erreur",
                "pipeline_utilise": pipeline,
                "message": "Document vide ou trop court apres nettoyage.",
                "fichier": Path(fichier_path).name,
            }

        vectoriser(docs)
        stats = stats_collection()
        return {
            "statut": "succès",
            "pipeline_utilise": pipeline,
            "fichier": Path(fichier_path).name,
            "documents_ingeres": len(docs),
            "projet": docs[0].metadata.get("projet", projet),
            "resume_bim": resume_bim,
            "stats_collection": stats,
        }
    except Exception as e:
        return {
            "statut": "erreur",
            "message": f"Erreur ingestion fichier intelligente : {e}",
            "fichier": Path(fichier_path).name,
        }


def ingerer_texte_brut(
    contenu: str,
    source: str = "saisie_manuelle",
    projet: str = "non_défini",
    lot_technique: str = "non_défini",
    auteur: str = "inconnu",
    criticite: str = "normale",
    type_document: str = "general",
) -> dict:
    """
    Workflow d'ingestion d'un texte brut (email, WhatsApp, note terrain…).
    """
    metadata = {
        "projet": projet,
        "lot_technique": lot_technique,
        "auteur": auteur,
        "criticite": criticite,
        "type_document": type_document,
    }

    print(f"\n[workflow] Ingestion texte brut | Source : {source} | Projet : {projet}")

    docs = charger_texte_brut(contenu, metadata_extra=metadata, source=source)
    docs = nettoyer(docs)

    if not docs:
        return {"statut": "erreur", "message": "Texte trop court ou vide après nettoyage."}

    vectoriser(docs)
    return {
        "statut": "succès",
        "source": source,
        "projet": projet,
        "longueur_texte": len(contenu),
    }


def ingerer_dossier(
    dossier: str,
    projet: str = "non_défini",
    lot_technique: str = "non_défini",
) -> dict:
    """
    Workflow d'ingestion batch de tous les documents d'un dossier.
    """
    metadata = {"projet": projet, "lot_technique": lot_technique}

    print(f"\n[workflow] Ingestion dossier : {dossier}")

    docs = charger_dossier(dossier, metadata_extra=metadata)
    if not docs:
        return {"statut": "erreur", "message": f"Aucun document trouvé dans '{dossier}'"}

    docs = nettoyer(docs)
    vectoriser(docs)
    stats = stats_collection()

    return {
        "statut": "succès",
        "dossier": dossier,
        "projet": projet,
        "documents_ingeres": len(docs),
        "stats_collection": stats,
    }


def ingerer_gmail(
    query: Optional[str] = None,
    max_results: Optional[int] = None,
    projet: str = "non_defini",
    lot_technique: str = "non_defini",
    criticite: str = "normale",
    filtrer_btp: bool = True,
) -> dict:
    """
    Workflow d'ingestion des emails Gmail via Google credentials.
    Etapes : Gmail API -> nettoyage -> vectorisation.
    """
    print(f"\n[workflow] Ingestion Gmail | Projet : {projet} | Requete : {query}")

    if not has_valid_gmail_token():
        auth_data = generate_auth_url()
        return {
            "need_auth": True,
            **auth_data,
            "message": "Connexion Gmail requise avant l'ingestion.",
        }

    docs = collecter_gmail(
        query=query,
        max_results=max_results,
        projet=projet,
        lot_technique=lot_technique,
        criticite=criticite,
    )
    if not docs:
        return {"statut": "erreur", "message": "Aucun email trouve pour cette requete."}
    emails_trouves = len(docs)

    if filtrer_btp:
        docs = _filtrer_emails_btp(docs, projet=projet)
    emails_filtres = len(docs)
    emails_rejetes = emails_trouves - emails_filtres

    if not docs:
        return {
            "statut": "erreur",
            "source": "gmail",
            "projet": projet,
            "query": query,
            "emails_trouves": emails_trouves,
            "emails_filtres": 0,
            "emails_ingeres": 0,
            "emails_rejetes": emails_rejetes,
            "filtrer_btp": filtrer_btp,
            "message": "Aucun email pertinent BTP apres filtrage.",
        }

    docs = nettoyer(docs)
    if not docs:
        return {
            "statut": "erreur",
            "message": "Emails vides ou trop courts apres nettoyage.",
            "emails_trouves": emails_trouves,
            "emails_filtres": emails_filtres,
            "emails_ingeres": 0,
            "emails_rejetes": emails_rejetes,
            "filtrer_btp": filtrer_btp,
        }

    vectoriser(docs)
    stats = stats_collection()

    return {
        "statut": "succès",
        "source": "gmail",
        "projet": projet,
        "query": query,
        "emails_trouves": emails_trouves,
        "emails_filtres": emails_filtres,
        "emails_ingeres": len(docs),
        "emails_rejetes": emails_rejetes,
        "filtrer_btp": filtrer_btp,
        "stats_collection": stats,
    }


MOTS_CLES_BTP_EMAIL = [
    "chantier", "béton", "beton", "fondation", "dtu", "travaux",
    "maçonnerie", "maconnerie", "devis", "charpente", "plomberie",
    "électricité", "electricite", "gros œuvre", "gros oeuvre",
    "second œuvre", "second oeuvre", "btp", "construction",
    "rénovation", "renovation", "architecte", "conducteur de travaux",
    "lot", "planning chantier", "réception", "reception", "livraison",
    "conformité", "conformite", "sécurité chantier", "securite chantier",
    "mur", "dalle", "toiture", "façade", "facade", "ferraillage",
]


def _filtrer_emails_btp(documents: list, projet: str = "non_defini") -> list:
    projet_normalise = (projet or "").strip().lower()
    filtrer_projet = projet_normalise not in {"", "non_defini", "non_défini", "non défini"}
    filtered = []

    for doc in documents:
        meta = doc.metadata or {}
        haystack = "\n".join(
            [
                str(meta.get("email_subject", "")),
                str(meta.get("email_from", "")),
                doc.page_content or "",
            ]
        ).lower()
        if not any(keyword.lower() in haystack for keyword in MOTS_CLES_BTP_EMAIL):
            continue
        if filtrer_projet and projet_normalise not in haystack:
            continue
        filtered.append(doc)

    return filtered


# ─────────────────────────────────────────────
# Workflows d'analyse
# ─────────────────────────────────────────────

def analyser_et_recommander(
    situation: str,
    projet: Optional[str] = None,
) -> dict:
    """
    Workflow combiné : analyse d'une situation + recommandations.

    Returns:
        dict contenant l'analyse IA et le rapport de recommandations.
    """
    print(f"\n[workflow] Analyse + recommandations | Projet : {projet or 'tous'}")

    # Analyse IA
    analyse = repondre(situation, projet=projet)

    # Détection de risques
    risques = detecter_risques(situation, projet=projet)

    # Recommandations structurées
    rapport = generer_recommandations(situation, projet=projet)

    return {
        "situation": situation,
        "projet": projet,
        "analyse": analyse,
        "risques": risques,
        "recommandations": rapport.model_dump(),
    }


def audit_projet(projet: str) -> dict:
    """
    Workflow d'audit complet d'un projet :
    1. Rapport de conformité réglementaire
    2. Détection des risques
    3. Alertes sur documents critiques
    4. Recommandations globales

    Returns:
        dict complet avec tous les résultats d'audit.
    """
    print(f"\n[workflow] Audit complet du projet : {projet}")

    # 1. Conformité
    print("[workflow] Étape 1/4 : Rapport de conformité...")
    conformite = rapport_conformite_projet(projet)

    # 2. Risques
    print("[workflow] Étape 2/4 : Détection des risques...")
    risques = detecter_risques(
        f"risques potentiels sur le projet {projet}",
        projet=projet,
    )

    # 3. Alertes critiques
    print("[workflow] Étape 3/4 : Alertes critiques...")
    alertes = generer_alertes_critiques(projet=projet)

    # 4. Recommandations globales
    print("[workflow] Étape 4/4 : Recommandations...")
    rapport = generer_recommandations(
        f"Audit global du projet {projet} : points d'attention et actions prioritaires",
        projet=projet,
    )

    return {
        "projet": projet,
        "conformite": conformite,
        "risques": risques,
        "alertes_critiques": alertes,
        "recommandations": rapport.model_dump(),
    }


def verifier_conformite_element(element: str) -> dict:
    """
    Workflow de vérification réglementaire d'un élément de construction.
    """
    print(f"\n[workflow] Vérification conformité : {element[:60]}...")
    resultat = verifier_conformite(element)
    return {
        "element": element,
        "analyse_conformite": resultat,
    }
