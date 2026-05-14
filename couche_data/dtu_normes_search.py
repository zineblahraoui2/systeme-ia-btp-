from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from couche_data.vectorisation import get_vectorstore


REGLEMENTAIRE_QUERY_RE = re.compile(
    r"\b(conform|réglement|reglement|norme|dtu|sécurité|securite|obligation|obligatoire|interdit|feu|incendie|tolérance|tolerance)\b",
    re.I,
)

CATALOGUE_ARTICLE_SECTION = "N/A — consulter le texte complet"


def is_reglementaire_query(query: str) -> bool:
    return bool(REGLEMENTAIRE_QUERY_RE.search(query or ""))


def _confidence_from_distance(distance: float) -> float:
    try:
        return round(max(0.0, min(1.0, 1.0 / (1.0 + float(distance)))), 3)
    except Exception:
        return 0.0


def _format_result(doc: Any, distance: float) -> dict:
    meta = dict(doc.metadata or {})
    norme = meta.get("numero_dtu") or meta.get("norme") or "Non trouvé"
    return {
        "norme": norme,
        "type_document": meta.get("type_document", "Non trouvé"),
        "type_doc": meta.get("type_doc", "dtu_complet"),
        "article": meta.get("article", "Non trouvé"),
        "section": meta.get("section", "Non trouvé"),
        "prescription": doc.page_content,
        "obligatoire": bool(meta.get("obligatoire", False)),
        "type_prescription": meta.get("type_prescription", "Non trouvé"),
        "page": meta.get("page", "Non trouvé"),
        "titre": meta.get("titre", "Non trouvé"),
        "domaine": meta.get("domaine", "Non trouvé"),
        "score": _confidence_from_distance(distance),
        "score_confiance": meta.get("score_confiance", 0.0),
        "source": meta.get("source_fichier") or meta.get("source", "Non trouvé"),
    }


def build_regulatory_fallback_response(query: str, catalogue_results: list[dict]) -> dict:
    """
    Construit une réponse utile quand aucune prescription précise n'est trouvée.
    Si des entrées de catalogue correspondent, elles sont listées comme références.
    """
    if not catalogue_results:
        return {
            "type": "reponse_reglementaire",
            "statut": "aucun_resultat",
            "message": (
                f"Aucun DTU ni prescription trouvés dans la base pour : « {query} ».\n"
                "Consultez le catalogue complet :\n"
                "- CSTB : https://www.cstb.fr\n"
                "- AFNOR : https://www.afnor.fr\n"
                "- BNTEC : https://www.bntec.fr"
            ),
            "citations": [],
        }

    domaines_vus: set[str] = set()
    citations = []
    for result in catalogue_results:
        citations.append(
            {
                "norme": result.get("norme", ""),
                "titre": result.get("titre", ""),
                "domaine": result.get("domaine", ""),
                "type_document": "DTU",
                "type_doc": "catalogue_index",
                "article": result.get("article") or CATALOGUE_ARTICLE_SECTION,
                "section": result.get("section") or CATALOGUE_ARTICLE_SECTION,
                "prescription": result.get("prescription", ""),
                "page": result.get("page", "N/A"),
                "score": result.get("score", 0.0),
                "score_confiance": result.get("score_confiance", 0.0),
            }
        )
        if result.get("domaine"):
            domaines_vus.add(result["domaine"])

    domaines_str = ", ".join(sorted(domaines_vus)) if domaines_vus else "Non trouvé"
    return {
        "type": "reponse_reglementaire",
        "statut": "reference_catalogue",
        "message": (
            f"Les DTU suivants sont référencés dans la base pour ce sujet "
            f"(domaine(s) : {domaines_str}). "
            "Ces références proviennent d'un catalogue d'index ; pour les prescriptions "
            "techniques précises (articles, tolérances, obligations), consultez le texte "
            "intégral de chaque norme auprès du CSTB ou de l'AFNOR."
        ),
        "citations": citations,
        "sources_officielles": [
            "CSTB — https://www.cstb.fr",
            "AFNOR — https://www.afnor.fr",
            "BNTEC — https://www.bntec.fr",
        ],
    }


def search_reglementaire(query: str, k: int = 6) -> list[dict]:
    vectorstore = get_vectorstore()
    try:
        results = vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter={
                "$or": [
                    {"type_document": {"$eq": "DTU"}},
                    {"type_document": {"$eq": "NORME"}},
                    {"type_document": {"$eq": "reglementation"}},
                    {"type_document": {"$eq": "norme"}},
                    {"type_document": {"$eq": "dtu"}},
                ]
            },
        )
    except Exception:
        return []
    return [_format_result(doc, distance) for doc, distance in results]


def split_regulatory_results(results: list[dict]) -> tuple[list[dict], list[dict]]:
    dtu_complets = [result for result in results if result.get("type_doc") != "catalogue_index"]
    dtu_catalogue = [result for result in results if result.get("type_doc") == "catalogue_index"]
    return dtu_complets, dtu_catalogue


def search_reglementaire_response(query: str, k: int = 10) -> dict:
    results = search_reglementaire(query, k=k)
    dtu_complets, dtu_catalogue = split_regulatory_results(results)
    if dtu_complets:
        return {
            "type": "reponse_reglementaire",
            "statut": "prescriptions_trouvees",
            "citations": dtu_complets,
        }
    if dtu_catalogue:
        return build_regulatory_fallback_response(query, dtu_catalogue)
    return build_regulatory_fallback_response(query, [])


def check_conformity(description_travaux: str, k: int = 5) -> dict:
    results = search_reglementaire(description_travaux, k=k)
    dtu_complets, dtu_catalogue = split_regulatory_results(results)
    if not dtu_complets:
        fallback = build_regulatory_fallback_response(description_travaux, dtu_catalogue)
        first = fallback.get("citations", [{}])[0] if fallback.get("citations") else {}
        return {
            "statut": "vigilance",
            "raison": fallback.get("message", "Aucune prescription trouvée dans la base réglementaire."),
            "norme": first.get("norme", "Non trouvé"),
            "article": first.get("article", CATALOGUE_ARTICLE_SECTION if first else "Non trouvé"),
            "section": first.get("section", CATALOGUE_ARTICLE_SECTION if first else "Non trouvé"),
            "citations": fallback.get("citations", []),
            "type": fallback.get("type", "reponse_reglementaire"),
            "reference_catalogue": fallback.get("statut") == "reference_catalogue",
        }

    best = dtu_complets[0]
    status = "vigilance"
    if best.get("type_prescription") in {"interdit", "sécurité critique"}:
        status = "non conforme"
    elif best.get("obligatoire"):
        status = "vigilance"

    return {
        "statut": status,
        "raison": best.get("prescription", "Non trouvé"),
        "norme": best.get("norme", "Non trouvé"),
        "article": best.get("article", "Non trouvé"),
        "section": best.get("section", "Non trouvé"),
        "page": best.get("page", "Non trouvé"),
        "citations": dtu_complets,
    }


def list_dtu_normes() -> list[dict]:
    vectorstore = get_vectorstore()
    try:
        raw = vectorstore.get(
            where={
                "$or": [
                    {"type_document": {"$eq": "DTU"}},
                    {"type_document": {"$eq": "NORME"}},
                ]
            },
            include=["metadatas"],
        )
    except Exception:
        return []

    grouped: dict[str, dict] = defaultdict(lambda: {"nom": "Non trouvé", "domaine": "Non trouvé", "chunks": 0})
    for meta in raw.get("metadatas", []) or []:
        nom = meta.get("numero_dtu") or meta.get("norme") or meta.get("titre") or "Non trouvé"
        grouped[nom]["nom"] = nom
        grouped[nom]["domaine"] = meta.get("domaine", grouped[nom]["domaine"])
        grouped[nom]["titre"] = meta.get("titre", "Non trouvé")
        grouped[nom]["type_document"] = meta.get("type_document", "Non trouvé")
        grouped[nom]["type_doc"] = meta.get("type_doc", "dtu_complet")
        grouped[nom]["chunks"] += 1
    return sorted(grouped.values(), key=lambda item: item["nom"])
