"""
couche_execution/recommandations.py
-------------------------------------
Génération de recommandations opérationnelles structurées.
- Recommandations suite à une question/analyse
- Alertes automatiques sur documents critiques
- Rapport de conformité projet
- Format structuré (Pydantic) pour exploitation par l'API
"""

from __future__ import annotations

from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

from couche_ia.retriever import rechercher, rechercher_documents_critiques, formater_contexte
from couche_ia.analyse_metier import get_llm
from config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────
# Modèles de données (sorties structurées)
# ─────────────────────────────────────────────

class NiveauPriorite(str, Enum):
    critique = "critique"
    haute = "haute"
    normale = "normale"
    faible = "faible"


class ActionRecommandee(BaseModel):
    titre: str = Field(description="Titre court de l'action")
    description: str = Field(description="Description détaillée de l'action à mener")
    responsable: str = Field(description="Rôle ou personne responsable de l'action")
    delai: str = Field(description="Délai recommandé (ex: 24h, 1 semaine, avant réception)")
    priorite: NiveauPriorite = Field(description="Niveau de priorité de l'action")


class RapportRecommandations(BaseModel):
    titre: str = Field(description="Titre du rapport")
    synthese: str = Field(description="Synthèse exécutive en 3-5 lignes")
    actions: list[ActionRecommandee] = Field(description="Liste des actions recommandées")
    references_normatives: list[str] = Field(description="DTU, normes ou articles applicables")
    risque_global: NiveauPriorite = Field(description="Niveau de risque global évalué")


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

def _prompt_recommandations(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", f"""Tu es un expert BTP chargé de produire des recommandations opérationnelles.
À partir du contexte documentaire et de la situation décrite, génère un rapport structuré.
Sois précis, actionnable et professionnel.

Contexte documentaire :
{{contexte}}

{parser.get_format_instructions()}
"""),
        ("human", "Situation ou question : {situation}"),
    ])


# ─────────────────────────────────────────────
# Fonctions principales
# ─────────────────────────────────────────────

def generer_recommandations(
    situation: str,
    projet: Optional[str] = None,
    k: int = None,
) -> RapportRecommandations:
    """
    Génère un rapport de recommandations structuré pour une situation donnée.

    Returns:
        RapportRecommandations : objet Pydantic avec actions prioritisées.
    """
    filtre = {"projet": projet} if projet else None
    docs = rechercher(situation, k=k, filtre=filtre)
    contexte = formater_contexte(docs)

    parser = PydanticOutputParser(pydantic_object=RapportRecommandations)
    prompt = _prompt_recommandations(parser)
    llm = get_llm()
    chain = prompt | llm | parser

    return chain.invoke({"situation": situation, "contexte": contexte})


def generer_alertes_critiques(projet: Optional[str] = None) -> list[dict]:
    """
    Scanne la base vectorielle pour les documents critiques et
    retourne une liste d'alertes résumées.
    """
    docs_critiques = rechercher_documents_critiques(projet=projet, k=10)
    if not docs_critiques:
        return []

    llm = get_llm()
    alertes = []

    for doc in docs_critiques:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Tu es un expert BTP. Résume ce document critique en une alerte concise (2-3 lignes max). Indique : nature du problème, impact potentiel, action immédiate requise."),
            ("human", "{contenu}"),
        ])
        chain = prompt | llm | StrOutputParser()
        resume = chain.invoke({"contenu": doc.page_content[:1500]})

        alertes.append({
            "alerte": resume,
            "projet": doc.metadata.get("projet", "N/A"),
            "lot": doc.metadata.get("lot_technique", "N/A"),
            "criticite": doc.metadata.get("criticite", "N/A"),
            "source": doc.metadata.get("source", "N/A"),
            "date": doc.metadata.get("date", "N/A"),
        })

    return alertes


def rapport_conformite_projet(projet: str) -> str:
    """
    Génère un rapport de conformité réglementaire pour un projet entier.
    Recherche les documents normatifs et les compare aux données projet.
    """
    from couche_ia.analyse_metier import verifier_conformite
    from couche_ia.retriever import rechercher_par_projet

    docs_projet = rechercher_par_projet(
        "conformité réglementation normes DTU",
        projet=projet,
        k=8,
    )
    if not docs_projet:
        return f"Aucun document trouvé pour le projet '{projet}'."

    contexte_projet = formater_contexte(docs_projet)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un expert en conformité BTP. 
Génère un rapport de conformité réglementaire complet pour ce projet.
Structure ton rapport ainsi :
1. IDENTITÉ DU PROJET
2. POINTS CONFORMES
3. POINTS NON CONFORMES ou À RISQUE
4. ACTIONS CORRECTIVES PRIORITAIRES
5. CONCLUSION

Contexte projet :
{contexte}
"""),
        ("human", "Génère le rapport de conformité pour le projet : {projet}"),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"projet": projet, "contexte": contexte_projet})
