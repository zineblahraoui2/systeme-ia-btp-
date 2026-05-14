from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document

NOT_FOUND = "Non trouvé"
FILE_SCHEMA_RE = re.compile(r"FILE_SCHEMA\(\('(.+?)'\)\)", re.IGNORECASE | re.DOTALL)
HEADER_FIELD_RE = re.compile(r"(FILE_NAME|FILE_DESCRIPTION|FILE_SCHEMA)\s*\((.*?)\)\s*;", re.IGNORECASE | re.DOTALL)
ENTITY_RE = re.compile(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*?)\)\s*;", re.IGNORECASE | re.DOTALL)
QUOTED_RE = re.compile(r"'((?:[^']|'')*)'")

COUNT_TYPES = {
    "IFCWALL": "IFCWALL",
    "IFCWINDOW": "IFCWINDOW",
    "IFCDOOR": "IFCDOOR",
    "IFCSLAB": "IFCSLAB",
    "IFCCOLUMN": "IFCCOLUMN",
    "IFCBEAM": "IFCBEAM",
    "IFCSPACE": "IFCSPACE",
}


def _read_ifc_text(fichier_path: str) -> str:
    try:
        text = Path(fichier_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(f"Impossible de lire le fichier IFC brut : {exc}") from exc
    if not text.strip():
        raise RuntimeError("Fichier IFC vide.")
    return text


def _clean_ifc_string(value: Any) -> str:
    if value in (None, "", "$", "*"):
        return NOT_FOUND
    return str(value).replace("''", "'").strip() or NOT_FOUND


def _safe_attr(obj: Any, name: str) -> str:
    return _clean_ifc_string(getattr(obj, name, None))


def _quoted_values(raw: str) -> list[str]:
    return [_clean_ifc_string(match.group(1)) for match in QUOTED_RE.finditer(raw or "")]


def _parse_header(text: str) -> dict[str, Any]:
    header = {
        "FILE_SCHEMA": NOT_FOUND,
        "FILE_NAME": NOT_FOUND,
        "FILE_DESCRIPTION": NOT_FOUND,
    }

    schema_match = FILE_SCHEMA_RE.search(text)
    if schema_match:
        header["FILE_SCHEMA"] = _clean_ifc_string(schema_match.group(1))

    for match in HEADER_FIELD_RE.finditer(text):
        field = match.group(1).upper()
        raw_value = match.group(2)
        values = _quoted_values(raw_value)
        if field == "FILE_SCHEMA" and header["FILE_SCHEMA"] == NOT_FOUND:
            header["FILE_SCHEMA"] = values[0] if values else NOT_FOUND
        elif field == "FILE_NAME":
            header["FILE_NAME"] = {
                "name": values[0] if len(values) > 0 else NOT_FOUND,
                "timestamp": values[1] if len(values) > 1 else NOT_FOUND,
                "author": values[2:] if len(values) > 2 else [],
            }
        elif field == "FILE_DESCRIPTION":
            header["FILE_DESCRIPTION"] = values or NOT_FOUND

    return header


def _raw_entities(text: str) -> list[dict[str, str]]:
    return [
        {
            "step_id": f"#{match.group(1)}",
            "type": match.group(2).upper(),
            "raw": match.group(3),
        }
        for match in ENTITY_RE.finditer(text)
    ]


def _raw_entity_counts(entities: list[dict[str, str]]) -> dict[str, int]:
    counts = {key: 0 for key in COUNT_TYPES}
    for entity in entities:
        entity_type = entity["type"]
        if entity_type in counts:
            counts[entity_type] += 1
    return counts


def _raw_guid_records(entities: list[dict[str, str]], entity_types: set[str]) -> dict[str, list[dict[str, str]]]:
    records = {entity_type: [] for entity_type in sorted(entity_types)}
    for entity in entities:
        entity_type = entity["type"]
        if entity_type not in records:
            continue
        values = _quoted_values(entity["raw"])
        records[entity_type].append(
            {
                "step_id": entity["step_id"],
                "guid": values[0] if values else NOT_FOUND,
                "name": values[2] if len(values) > 2 else NOT_FOUND,
            }
        )
    return records


def _raw_materials(entities: list[dict[str, str]]) -> list[str]:
    materials: list[str] = []
    for entity in entities:
        if entity["type"] == "IFCMATERIAL":
            values = _quoted_values(entity["raw"])
            materials.append(values[0] if values else NOT_FOUND)
    return sorted({item for item in materials if item != NOT_FOUND})


def _raw_units(entities: list[dict[str, str]]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for entity in entities:
        if entity["type"].endswith("UNIT"):
            values = _quoted_values(entity["raw"])
            units.append(
                {
                    "step_id": entity["step_id"],
                    "type": entity["type"],
                    "values": values if values else [NOT_FOUND],
                }
            )
    return units


def _raw_spatial_relations(entities: list[dict[str, str]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for entity in entities:
        if entity["type"] not in {"IFCRELAGGREGATES", "IFCRELCONTAINEDINSPATIALSTRUCTURE"}:
            continue
        refs = re.findall(r"#\d+", entity["raw"])
        relations.append(
            {
                "step_id": entity["step_id"],
                "type": entity["type"],
                "references": refs,
            }
        )
    return relations


def _owner_name(owner_history: Any) -> str:
    if owner_history is None:
        return NOT_FOUND
    user = getattr(owner_history, "OwningUser", None)
    person = getattr(user, "ThePerson", None)
    organisation = getattr(user, "TheOrganization", None)
    parts = [
        getattr(person, "GivenName", None),
        getattr(person, "FamilyName", None),
        getattr(person, "Identification", None),
    ]
    name = " ".join(str(part) for part in parts if part).strip()
    if name:
        return name
    org_name = getattr(organisation, "Name", None)
    if org_name:
        return str(org_name)
    app = getattr(owner_history, "OwningApplication", None)
    return _clean_ifc_string(
        getattr(app, "ApplicationFullName", None) or getattr(app, "ApplicationIdentifier", None)
    )


def _creation_date(owner_history: Any) -> str:
    timestamp = getattr(owner_history, "CreationDate", None) if owner_history is not None else None
    if not timestamp:
        return NOT_FOUND
    try:
        return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return _clean_ifc_string(timestamp)


def _model_by_type(model: Any, type_name: str) -> list[Any]:
    try:
        return list(model.by_type(type_name))
    except Exception:
        return []


def _entity_record(entity: Any) -> dict[str, str]:
    return {
        "guid": _safe_attr(entity, "GlobalId"),
        "name": _safe_attr(entity, "Name"),
        "object_type": _safe_attr(entity, "ObjectType"),
    }


def _model_hierarchy(model: Any) -> dict[str, list[dict[str, str]]]:
    return {
        "IFCPROJECT": [_entity_record(item) for item in _model_by_type(model, "IfcProject")],
        "IFCSITE": [_entity_record(item) for item in _model_by_type(model, "IfcSite")],
        "IFCBUILDING": [_entity_record(item) for item in _model_by_type(model, "IfcBuilding")],
        "IFCBUILDINGSTOREY": [_entity_record(item) for item in _model_by_type(model, "IfcBuildingStorey")],
    }


def _model_project_info(model: Any) -> tuple[str, str, str]:
    projects = _model_by_type(model, "IfcProject")
    if not projects:
        return NOT_FOUND, NOT_FOUND, NOT_FOUND
    project = projects[0]
    owner_history = getattr(project, "OwnerHistory", None)
    return _safe_attr(project, "Name"), _owner_name(owner_history), _creation_date(owner_history)


def _model_materials(model: Any) -> list[str]:
    return sorted(
        {
            _safe_attr(material, "Name")
            for material in _model_by_type(model, "IfcMaterial")
            if _safe_attr(material, "Name") != NOT_FOUND
        }
    )


def _model_units(model: Any) -> list[str]:
    projects = _model_by_type(model, "IfcProject")
    if not projects:
        return []
    units = getattr(getattr(projects[0], "UnitsInContext", None), "Units", None) or []
    result = []
    for unit in units:
        unit_type = _safe_attr(unit, "UnitType")
        name = _safe_attr(unit, "Name")
        prefix = _safe_attr(unit, "Prefix")
        result.append(" | ".join(part for part in [unit_type, prefix, name] if part != NOT_FOUND))
    return result


def _model_spatial_relations(model: Any) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for rel_type in ("IfcRelAggregates", "IfcRelContainedInSpatialStructure"):
        for rel in _model_by_type(model, rel_type):
            parent = getattr(rel, "RelatingObject", None) or getattr(rel, "RelatingStructure", None)
            children = getattr(rel, "RelatedObjects", None) or getattr(rel, "RelatedElements", None) or []
            relations.append(
                {
                    "type": rel_type.upper(),
                    "parent": _entity_record(parent) if parent is not None else NOT_FOUND,
                    "children": [_entity_record(child) for child in children],
                }
            )
    return relations


def _model_guid_records(model: Any) -> dict[str, list[dict[str, str]]]:
    requested = {
        "IFCWALL": "IfcWall",
        "IFCWINDOW": "IfcWindow",
        "IFCDOOR": "IfcDoor",
        "IFCSLAB": "IfcSlab",
        "IFCCOLUMN": "IfcColumn",
        "IFCBEAM": "IfcBeam",
        "IFCSPACE": "IfcSpace",
    }
    return {
        key: [_entity_record(item) for item in _model_by_type(model, ifc_type)]
        for key, ifc_type in requested.items()
    }


def _load_ifcopenshell_model(fichier_path: str) -> Any:
    try:
        import ifcopenshell
    except ImportError:
        return None
    try:
        return ifcopenshell.open(fichier_path)
    except Exception:
        return None


def _detect_lot_technique(counts: dict[str, int]) -> str:
    if counts.get("IFCWALL", 0) or counts.get("IFCSLAB", 0) or counts.get("IFCCOLUMN", 0) or counts.get("IFCBEAM", 0):
        return "gros_oeuvre_structure"
    if counts.get("IFCDOOR", 0) or counts.get("IFCWINDOW", 0):
        return "second_oeuvre_menuiseries"
    if counts.get("IFCSPACE", 0):
        return "architecture"
    return "BIM"


def extraire_bim_ifc(
    fichier_path: str,
    projet: Optional[str] = None,
    lot_technique: Optional[str] = None,
    criticite: str = "haute",
    auteur: Optional[str] = None,
) -> tuple[list[Document], dict]:
    """Extraction IFC stricte: valeurs exactes, JSON, aucun enrichissement invente."""
    raw_text = _read_ifc_text(fichier_path)
    if "FILE_SCHEMA" not in raw_text.upper() and "IFCPROJECT" not in raw_text.upper():
        raise RuntimeError("Fichier IFC corrompu ou non reconnu.")

    header = _parse_header(raw_text)
    raw_entities = _raw_entities(raw_text)
    raw_counts = _raw_entity_counts(raw_entities)
    model = _load_ifcopenshell_model(fichier_path)

    if model is not None:
        project_name, ifc_author, created_at = _model_project_info(model)
        hierarchy = _model_hierarchy(model)
        materials = _model_materials(model)
        units = _model_units(model)
        guid_records = _model_guid_records(model)
        spatial_relations = _model_spatial_relations(model)
        parser = "ifcopenshell"
    else:
        project_name = NOT_FOUND
        ifc_author = NOT_FOUND
        created_at = NOT_FOUND
        hierarchy = {
            key: _raw_guid_records(raw_entities, {key}).get(key, [])
            for key in ("IFCPROJECT", "IFCSITE", "IFCBUILDING", "IFCBUILDINGSTOREY")
        }
        materials = _raw_materials(raw_entities)
        units = _raw_units(raw_entities)
        guid_records = _raw_guid_records(raw_entities, set(COUNT_TYPES))
        spatial_relations = _raw_spatial_relations(raw_entities)
        parser = "raw_ifc_text"

    if project_name == NOT_FOUND and hierarchy.get("IFCPROJECT"):
        project_name = hierarchy["IFCPROJECT"][0].get("name", NOT_FOUND)

    detected_lot = lot_technique if lot_technique and lot_technique != "non_defini" else _detect_lot_technique(raw_counts)
    metadata = {
        "source": fichier_path,
        "type_document": "BIM",
        "projet": project_name if project_name != NOT_FOUND else (projet or "non_defini"),
        "lot_technique": detected_lot,
        "criticite": "haute",
        "auteur": auteur if auteur and auteur != "inconnu" else ifc_author,
        "date": created_at,
        "ingere_le": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "fichier_nom": Path(fichier_path).name,
        "nom_fichier": Path(fichier_path).name,
        "extraction_mode": "strict_ifc",
        "ifc_schema": header["FILE_SCHEMA"],
    }

    extraction = {
        "mode": "strict_ifc",
        "parser": parser,
        "header": header,
        "metadata": {
            "IFCPROJECT": hierarchy.get("IFCPROJECT") or NOT_FOUND,
            "IFCSITE": hierarchy.get("IFCSITE") or NOT_FOUND,
            "IFCBUILDING": hierarchy.get("IFCBUILDING") or NOT_FOUND,
            "IFCBUILDINGSTOREY": hierarchy.get("IFCBUILDINGSTOREY") or NOT_FOUND,
            "author": ifc_author,
            "creation_date": created_at,
        },
        "counts": raw_counts,
        "materials": materials or NOT_FOUND,
        "units": units or NOT_FOUND,
        "guids": guid_records,
        "spatial_relations": spatial_relations or NOT_FOUND,
    }

    text = json.dumps(extraction, ensure_ascii=False, indent=2, sort_keys=True)
    summary = {
        "projet_bim": metadata["projet"],
        "auteur": metadata["auteur"],
        "date_creation": created_at,
        "schema": header["FILE_SCHEMA"],
        "lot_technique_detecte": detected_lot,
        "elements_extraits": raw_counts,
        "mode": "strict_ifc",
        "parser": parser,
    }
    return [Document(page_content=text, metadata=metadata)], summary
