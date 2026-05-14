"""
couche_ia/retriever.py
----------------------
Moteur de recherche sémantique sur la base vectorielle ChromaDB.
- Recherche par similarité (dense retrieval)
- Filtres sur les métadonnées BTP (projet, lot, type, criticité)
- Recherche hybride (similarité + filtres)
- Retourne des Documents LangChain avec scores
"""

from __future__ import annotations

from typing import Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma

from couche_data.vectorisation import get_vectorstore
from config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────
# Recherche sémantique de base
# ─────────────────────────────────────────────

def rechercher(
    query: str,
    k: int = None,
    filtre: Optional[dict] = None,
) -> list[Document]:
    """
    Recherche sémantique dans la base vectorielle.

    Args:
        query   : La question ou requête en langage naturel.
        k       : Nombre de chunks à retourner (défaut : settings.retrieval_k).
        filtre  : Filtres ChromaDB sur les métadonnées.
                  Exemple : {"projet": "résidence_les_pins", "criticite": "haute"}

    Returns:
        Liste de Documents LangChain triés par pertinence.
    """
    k = k or settings.retrieval_k
    vectorstore: Chroma = get_vectorstore()

    kwargs = {"k": k}
    if filtre:
        # ChromaDB attend les filtres sous forme {"$and": [...]} ou directement {clé: valeur}
        if len(filtre) == 1:
            kwargs["filter"] = filtre
        else:
            kwargs["filter"] = {
                "$and": [{key: {"$eq": val}} for key, val in filtre.items()]
            }

    docs = vectorstore.similarity_search(query, **kwargs)
    return docs


def rechercher_avec_score(
    query: str,
    k: int = None,
    filtre: Optional[dict] = None,
) -> list[tuple[Document, float]]:
    """
    Recherche sémantique avec scores de similarité.

    Returns:
        Liste de tuples (Document, score) triés par score décroissant.
        Score proche de 0 = très similaire (distance L2).
    """
    k = k or settings.retrieval_k
    vectorstore: Chroma = get_vectorstore()

    kwargs = {"k": k}
    if filtre:
        if len(filtre) == 1:
            kwargs["filter"] = filtre
        else:
            kwargs["filter"] = {
                "$and": [{key: {"$eq": val}} for key, val in filtre.items()]
            }

    results = vectorstore.similarity_search_with_score(query, **kwargs)
    return results


# ─────────────────────────────────────────────
# Recherches métier spécialisées
# ─────────────────────────────────────────────

def rechercher_par_projet(query: str, projet: str, k: int = None) -> list[Document]:
    """Recherche limitée aux documents d'un projet spécifique."""
    return rechercher(query, k=k, filtre={"projet": projet})


def rechercher_normes(query: str, k: int = None) -> list[Document]:
    """Recherche dans les DTU, normes et réglementation uniquement."""
    vectorstore: Chroma = get_vectorstore()
    k = k or settings.retrieval_k
    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter={"$or": [
            {"type_document": {"$eq": "dtu"}},
            {"type_document": {"$eq": "norme"}},
            {"type_document": {"$eq": "reglementation"}},
            {"type_document": {"$eq": "cstb"}},
        ]},
    )
    return docs


def rechercher_documents_critiques(projet: Optional[str] = None, k: int = 10) -> list[Document]:
    """Retourne les documents de criticité haute ou critique."""
    vectorstore: Chroma = get_vectorstore()
    filtre_criticite = {
        "$or": [
            {"criticite": {"$eq": "critique"}},
            {"criticite": {"$eq": "haute"}},
        ]
    }
    if projet:
        filtre_final = {
            "$and": [
                {"projet": {"$eq": projet}},
                filtre_criticite,
            ]
        }
    else:
        filtre_final = filtre_criticite

    # Recherche sans query sémantique (scan par filtre)
    docs = vectorstore.similarity_search(
        "problème urgence non-conformité",
        k=k,
        filter=filtre_final,
    )
    return docs


# ─────────────────────────────────────────────
# Formatage du contexte pour le LLM
# ─────────────────────────────────────────────

def formater_contexte(docs: list[Document]) -> str:
    """
    Formate les documents récupérés en un bloc de contexte
    lisible par le LLM, avec les métadonnées clés.
    """
    parties = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        entete = (
            f"[Document {i}]"
            f" | Projet : {meta.get('projet', 'N/A')}"
            f" | Type : {meta.get('type_document', 'N/A')}"
            f" | Lot : {meta.get('lot_technique', 'N/A')}"
            f" | Criticité : {meta.get('criticite', 'N/A')}"
            f" | Source : {meta.get('source', 'N/A')}"
        )
        parties.append(f"{entete}\n{doc.page_content}")

    return "\n\n---\n\n".join(parties)
