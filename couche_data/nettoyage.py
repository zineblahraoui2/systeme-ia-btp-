"""
couche_data/nettoyage.py
------------------------
Nettoyage et structuration des documents avant vectorisation :
- Suppression des doublons (hash du contenu)
- Normalisation du texte (espaces, sauts de ligne, encodage)
- Filtrage des documents vides ou trop courts
- Classification automatique par type de document BTP
"""

from __future__ import annotations

import hashlib
import re
from langchain_core.documents import Document


# ─────────────────────────────────────────────
# Constantes de classification BTP
# ─────────────────────────────────────────────

MOTS_CLES_TYPE: dict[str, list[str]] = {
    "dtu": ["dtu", "document technique unifié", "règles de construction"],
    "norme": ["norme nf", "norme en", "iso ", "standard technique"],
    "reglementation": ["code de la construction", "arrêté", "décret", "loi ", "obligation légale", "urbanisme"],
    "cstb": ["cstb", "avis technique", "guide cstb", "recommandation professionnelle"],
    "fournisseur": ["fiche technique", "notice de pose", "certification", "fiche produit"],
    "rapport_chantier": ["rapport de chantier", "compte-rendu", "cr chantier", "pv de réception"],
    "devis": ["devis", "offre de prix", "bordereau"],
    "plan": ["plan masse", "plan de coupe", "maquette", "bim", "dwg"],
    "email": ["de :", "objet :", "envoyé le", "cordialement"],
    "whatsapp": ["whatsapp", "message terrain", "photo chantier"],
}

CRITICITE_MOTS_CLES: dict[str, list[str]] = {
    "critique": ["urgence", "urgent", "bloquant", "arrêt chantier", "sinistre", "non-conformité grave"],
    "haute": ["non-conformité", "défaut", "retard", "dépassement", "risque"],
    "normale": [],  # valeur par défaut
}


# ─────────────────────────────────────────────
# Nettoyage du texte
# ─────────────────────────────────────────────

def normaliser_texte(texte: str) -> str:
    """Nettoie et normalise un texte brut."""
    # Supprimer les caractères de contrôle sauf \n
    texte = re.sub(r"[^\S\n]+", " ", texte)
    # Réduire les lignes vides consécutives
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    # Supprimer les espaces en début/fin de ligne
    texte = "\n".join(line.strip() for line in texte.splitlines())
    # Normaliser les apostrophes et guillemets
    texte = texte.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return texte.strip()


# ─────────────────────────────────────────────
# Déduplication
# ─────────────────────────────────────────────

def hash_document(doc: Document) -> str:
    """Calcule un hash MD5 du contenu d'un document."""
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


def dedoublonner(documents: list[Document]) -> list[Document]:
    """Supprime les documents dont le contenu est identique."""
    vus: set[str] = set()
    uniques: list[Document] = []
    for doc in documents:
        h = hash_document(doc)
        if h not in vus:
            vus.add(h)
            uniques.append(doc)
    supprimés = len(documents) - len(uniques)
    if supprimés:
        print(f"[nettoyage] {supprimés} doublons supprimés.")
    return uniques


# ─────────────────────────────────────────────
# Filtrage
# ─────────────────────────────────────────────

def filtrer_trop_courts(documents: list[Document], min_chars: int = 80) -> list[Document]:
    """Supprime les documents dont le contenu est trop court pour être utile."""
    avant = len(documents)
    docs = [d for d in documents if len(d.page_content.strip()) >= min_chars]
    print(f"[nettoyage] {avant - len(docs)} documents trop courts supprimés (< {min_chars} caractères).")
    return docs


# ─────────────────────────────────────────────
# Classification automatique
# ─────────────────────────────────────────────

def classifier_type_document(texte: str) -> str:
    """Détecte automatiquement le type de document BTP à partir de son contenu."""
    texte_lower = texte.lower()
    for type_doc, mots in MOTS_CLES_TYPE.items():
        if any(mot in texte_lower for mot in mots):
            return type_doc
    return "general"


def classifier_criticite(texte: str) -> str:
    """Détecte automatiquement la criticité d'un document."""
    texte_lower = texte.lower()
    for niveau, mots in CRITICITE_MOTS_CLES.items():
        if mots and any(mot in texte_lower for mot in mots):
            return niveau
    return "normale"


def enrichir_metadata_auto(doc: Document) -> Document:
    """
    Enrichit les métadonnées d'un document par classification automatique
    si elles ne sont pas déjà renseignées.
    """
    texte = doc.page_content

    if doc.metadata.get("type_document") in (None, "general", "pdf", "docx", "texte", "inconnu"):
        doc.metadata["type_document"] = classifier_type_document(texte)

    if doc.metadata.get("criticite") in (None, "normale"):
        doc.metadata["criticite"] = classifier_criticite(texte)

    return doc


# ─────────────────────────────────────────────
# Pipeline principal de nettoyage
# ─────────────────────────────────────────────

def nettoyer(documents: list[Document], min_chars: int = 80) -> list[Document]:
    """
    Pipeline complet de nettoyage :
    1. Normalisation du texte
    2. Filtrage des documents trop courts
    3. Déduplication
    4. Classification automatique des métadonnées
    """
    print(f"[nettoyage] Début : {len(documents)} documents")

    # 1. Normalisation
    for doc in documents:
        doc.page_content = normaliser_texte(doc.page_content)

    # 2. Filtrage
    documents = filtrer_trop_courts(documents, min_chars)

    # 3. Déduplication
    documents = dedoublonner(documents)

    # 4. Classification automatique
    documents = [enrichir_metadata_auto(doc) for doc in documents]

    print(f"[nettoyage] Fin : {len(documents)} documents propres")
    return documents
