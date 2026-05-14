"""
couche_data/vectorisation.py
-----------------------------
Transformation des documents nettoyés en vecteurs stockés dans ChromaDB.
- Découpage intelligent par contexte (RecursiveCharacterTextSplitter)
- Génération d'embeddings via OpenAI text-embedding-3-small
- Indexation dans ChromaDB avec persistance sur disque
- Fonctions d'ajout incrémental et de réinitialisation
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────
# Initialisation des composants
# ─────────────────────────────────────────────

def get_splitter() -> RecursiveCharacterTextSplitter:
    """Retourne le splitter configuré pour les documents BTP."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """Retourne le modèle d'embeddings OpenAI."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def get_vectorstore() -> Chroma:
    """
    Retourne (ou crée) la base vectorielle ChromaDB persistée sur disque.
    Si la collection existe déjà, charge les vecteurs existants.
    """
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


# ─────────────────────────────────────────────
# Découpage des documents
# ─────────────────────────────────────────────

def decouper(documents: list[Document]) -> list[Document]:
    """
    Découpe les documents en chunks contextuels.
    Les métadonnées du document parent sont héritées par chaque chunk.
    """
    splitter = get_splitter()
    chunks = splitter.split_documents(documents)
    print(f"[vectorisation] {len(documents)} documents → {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# Indexation dans ChromaDB
# ─────────────────────────────────────────────

def indexer(chunks: list[Document], batch_size: int = 100) -> Chroma:
    """
    Génère les embeddings et indexe les chunks dans ChromaDB.
    Traite par batchs pour éviter les limites d'API.
    Retourne le vectorstore mis à jour.
    """
    vectorstore = get_vectorstore()

    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        vectorstore.add_documents(batch)
        print(f"[vectorisation] Indexé {min(i + batch_size, total)}/{total} chunks")

    print(f"[vectorisation] Indexation terminée. Collection : '{settings.chroma_collection_name}'")
    return vectorstore


def reinitialiser_collection() -> Chroma:
    """
    Supprime et recrée la collection ChromaDB.
    À utiliser avec précaution (perte de tous les vecteurs existants).
    """
    import chromadb
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        client.delete_collection(settings.chroma_collection_name)
        print(f"[vectorisation] Collection '{settings.chroma_collection_name}' supprimée.")
    except Exception:
        pass
    return get_vectorstore()


# ─────────────────────────────────────────────
# Pipeline complet : documents → ChromaDB
# ─────────────────────────────────────────────

def vectoriser(documents: list[Document]) -> Chroma:
    """
    Pipeline complet :
    1. Découpage en chunks
    2. Génération des embeddings
    3. Indexation dans ChromaDB

    Retourne le vectorstore prêt à être interrogé.
    """
    if not documents:
        print("[vectorisation] Aucun document à vectoriser.")
        return get_vectorstore()

    chunks = decouper(documents)
    vectorstore = indexer(chunks)
    return vectorstore


def stats_collection() -> dict:
    """Retourne les statistiques de la collection ChromaDB."""
    import chromadb
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        col = client.get_collection(settings.chroma_collection_name)
        return {
            "collection": settings.chroma_collection_name,
            "nombre_vecteurs": col.count(),
            "persist_dir": settings.chroma_persist_dir,
        }
    except Exception as e:
        return {"erreur": str(e)}
