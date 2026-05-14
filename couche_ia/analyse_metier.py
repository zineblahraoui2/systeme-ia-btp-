"""
couche_ia/analyse_metier.py
----------------------------
Moteur d'analyse métier BTP : chaînes RAG (Retrieval-Augmented Generation)
avec prompts spécialisés pour le secteur de la construction.

Fonctions disponibles :
- repondre()           : Q&A général sur la base de connaissance BTP
- verifier_conformite(): Vérification réglementaire (DTU, normes, code de la construction)
- analyser_document()  : Analyse et résumé d'un document BTP
- detecter_risques()   : Détection de risques dans un contexte projet
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional

from couche_ia.llm_engine import LLMEngine, documents_to_context_chunks
from couche_ia.retriever import rechercher, rechercher_normes, formater_contexte
from config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────
# LLM partagé
# ─────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


# ─────────────────────────────────────────────
# Prompts BTP spécialisés
# ─────────────────────────────────────────────

PROMPT_QA_BTP = ChatPromptTemplate.from_messages([
    ("system", """Tu es un assistant expert en BTP (Bâtiment et Travaux Publics).
Tu disposes d'une base de connaissance issue de documents internes (emails, rapports chantier, 
plans, devis) et de références techniques (DTU, normes NF/EN/ISO, réglementation, CSTB).

Règles :
- Réponds uniquement à partir des documents fournis dans le contexte.
- Si l'information n'est pas dans le contexte, dis-le clairement.
- Cite les sources (type de document, projet, lot) quand c'est pertinent.
- Utilise un langage professionnel adapté au secteur BTP.
- Sois précis et opérationnel.

Contexte documentaire :
{contexte}
"""),
    ("human", "{question}"),
])

PROMPT_CONFORMITE = ChatPromptTemplate.from_messages([
    ("system", """Tu es un expert en réglementation BTP et en conformité technique.
Tu maîtrises les DTU, les normes NF/EN/ISO, le Code de la Construction, et les avis CSTB.

Ta mission : analyser si une situation ou un élément de chantier est conforme aux exigences 
réglementaires et techniques en vigueur.

Format de ta réponse :
1. VERDICT : Conforme / Non conforme / À vérifier
2. RÉFÉRENCES : DTU, normes ou articles applicables
3. POINTS DE VIGILANCE : ce qui doit être vérifié ou corrigé
4. RECOMMANDATIONS : actions concrètes à mener

Contexte réglementaire disponible :
{contexte}
"""),
    ("human", "Élément à vérifier : {question}"),
])

PROMPT_ANALYSE_DOC = ChatPromptTemplate.from_messages([
    ("system", """Tu es un expert BTP chargé d'analyser des documents techniques de construction.

Produis une analyse structurée du document fourni :
1. RÉSUMÉ EXÉCUTIF (3-5 lignes)
2. INFORMATIONS CLÉS (projet, lot, intervenants, dates)
3. POINTS D'ACTION identifiés
4. RISQUES ou NON-CONFORMITÉS détectés
5. SUITES À DONNER recommandées

Sois concis et opérationnel.
"""),
    ("human", "Document à analyser :\n\n{document}"),
])

PROMPT_RISQUES = ChatPromptTemplate.from_messages([
    ("system", """Tu es un expert en gestion des risques chantier BTP.
Tu analyses les situations pour identifier les risques potentiels : 
sécurité, qualité, délais, conformité réglementaire, responsabilité.

Pour chaque risque identifié, fournis :
- Niveau : CRITIQUE / ÉLEVÉ / MODÉRÉ / FAIBLE
- Description du risque
- Impact potentiel
- Mesure de mitigation recommandée

Contexte du projet :
{contexte}
"""),
    ("human", "Situation à analyser : {question}"),
])


# ─────────────────────────────────────────────
# Chaînes RAG
# ─────────────────────────────────────────────

def repondre(question: str, projet: Optional[str] = None, k: int = None) -> str:
    """
    Q&A général sur la base de connaissance BTP.
    Recherche sémantique + génération de réponse contextualisée.
    """
    filtre = {"projet": projet} if projet else None
    docs = rechercher(question, k=k, filtre=filtre)
    chunks = documents_to_context_chunks(docs)
    return LLMEngine().generate(question, chunks)


def verifier_conformite(element: str, k: int = None) -> str:
    """
    Vérifie la conformité d'un élément aux normes et réglementations BTP.
    Recherche ciblée sur DTU, normes, réglementation et CSTB.
    """
    docs = rechercher_normes(element, k=k)
    contexte = formater_contexte(docs)

    llm = get_llm()
    chain = PROMPT_CONFORMITE | llm | StrOutputParser()

    return chain.invoke({"question": element, "contexte": contexte})


def analyser_document(contenu_document: str) -> str:
    """
    Analyse et résume un document BTP (rapport, email, CR chantier…).
    N'utilise pas le RAG — analyse directe du contenu fourni.
    """
    llm = get_llm()
    chain = PROMPT_ANALYSE_DOC | llm | StrOutputParser()

    return chain.invoke({"document": contenu_document})


def detecter_risques(situation: str, projet: Optional[str] = None, k: int = None) -> str:
    """
    Analyse une situation projet pour identifier les risques BTP.
    """
    filtre = {"projet": projet} if projet else None
    docs = rechercher(situation, k=k, filtre=filtre)
    contexte = formater_contexte(docs)

    llm = get_llm()
    chain = PROMPT_RISQUES | llm | StrOutputParser()

    return chain.invoke({"question": situation, "contexte": contexte})


# ─────────────────────────────────────────────
# Import conditionnel pour éviter l'import circulaire
# ─────────────────────────────────────────────
