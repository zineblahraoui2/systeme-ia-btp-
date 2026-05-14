"""
couche_data/gmail_collecte.py
-----------------------------
Collecte des emails Gmail via Google OAuth credentials.

Le premier lancement utilise credentials.json pour creer token.json.
Les emails sont convertis en Documents LangChain avec metadonnees BTP.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from langchain_core.documents import Document

from config import get_settings


GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
logger = logging.getLogger(__name__)


def _base64url_no_padding(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _credentials_paths() -> tuple[Path, Path]:
    settings = get_settings()
    return Path(settings.google_credentials_file), Path(settings.google_token_file)


def _load_credentials_from_env(token_json: str) -> Optional[Credentials]:
    try:
        token_info = json.loads(token_json)
        return Credentials.from_authorized_user_info(token_info, GMAIL_SCOPES)
    except Exception as exc:
        logger.warning("GMAIL_TOKEN est present mais invalide: %s", exc)
        return None


def _save_gmail_token(creds: Credentials, token_path: Path) -> None:
    settings = get_settings()
    token_json = creds.to_json()
    if settings.is_railway:
        os.environ["GMAIL_TOKEN"] = token_json
        logger.warning(
            "Token Gmail rafraichi sur Railway. Mets a jour la variable GMAIL_TOKEN "
            "dans Railway Variables avec le nouveau JSON du token."
        )
        return

    token_path.write_text(token_json, encoding="utf-8")


def _build_flow(state: Optional[str] = None) -> Flow:
    settings = get_settings()
    credentials_path, _ = _credentials_paths()
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Fichier Google credentials introuvable : {credentials_path}. "
            "Telecharge le fichier OAuth client depuis Google Cloud et place-le ici."
        )
    if settings.google_redirect_uri.startswith(("http://localhost", "http://127.0.0.1")):
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    return Flow.from_client_secrets_file(
        str(credentials_path),
        scopes=GMAIL_SCOPES,
        redirect_uri=settings.google_redirect_uri,
        state=state,
    )


def get_valid_gmail_credentials() -> Optional[Credentials]:
    settings = get_settings()
    _, token_path = _credentials_paths()
    creds: Optional[Credentials] = None

    if settings.gmail_token:
        creds = _load_credentials_from_env(settings.gmail_token)
    elif token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), GMAIL_SCOPES)
        except Exception:
            return None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_gmail_token(creds, token_path)
        except Exception:
            return None

    if creds and creds.valid:
        return creds
    return None


def has_valid_gmail_token() -> bool:
    return get_valid_gmail_credentials() is not None


def generate_auth_url() -> dict[str, str]:
    flow = _build_flow()
    code_verifier = _base64url_no_padding(secrets.token_bytes(64))
    code_challenge = _base64url_no_padding(
        hashlib.sha256(code_verifier.encode("ascii")).digest()
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        code_challenge=code_challenge,
        code_challenge_method="S256",
    )
    return {
        "auth_url": auth_url,
        "state": state,
        "code_verifier": code_verifier,
    }


def exchange_code_for_token(
    code: str,
    state: Optional[str] = None,
    code_verifier: Optional[str] = None,
) -> Credentials:
    _, token_path = _credentials_paths()
    flow = _build_flow(state=state)
    flow.fetch_token(code=code, code_verifier=code_verifier)
    creds = flow.credentials
    _save_gmail_token(creds, token_path)
    return creds


def _get_gmail_credentials() -> Credentials:
    creds = get_valid_gmail_credentials()
    if not creds:
        raise RuntimeError("Authentification Gmail requise.")
    return creds


def _gmail_service():
    creds = _get_gmail_credentials()
    return build("gmail", "v1", credentials=creds)


def _decode_base64url(data: str) -> str:
    raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    return raw.decode("utf-8", errors="replace")


def _headers_to_dict(headers: list[dict]) -> dict[str, str]:
    return {h.get("name", "").lower(): h.get("value", "") for h in headers}


def _extract_text_from_payload(payload: dict) -> str:
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if body_data and mime_type in {"text/plain", "text/html"}:
        text = _decode_base64url(body_data)
        if mime_type == "text/html":
            # Nettoyage HTML leger sans dependance supplementaire.
            import re

            text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
            text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
            text = re.sub(r"<[^>]+>", " ", text)
        return text

    parts = payload.get("parts", []) or []
    extracted: list[str] = []
    for part in parts:
        text = _extract_text_from_payload(part)
        if text:
            extracted.append(text)
    return "\n\n".join(extracted)


def _format_email_content(headers: dict[str, str], body: str) -> str:
    return "\n".join(
        [
            f"De : {headers.get('from', 'inconnu')}",
            f"A : {headers.get('to', 'inconnu')}",
            f"Date : {headers.get('date', '')}",
            f"Objet : {headers.get('subject', '(sans objet)')}",
            "",
            body.strip(),
        ]
    ).strip()


def collecter_gmail(
    query: Optional[str] = None,
    max_results: Optional[int] = None,
    projet: str = "non_defini",
    lot_technique: str = "non_defini",
    criticite: str = "normale",
) -> list[Document]:
    """
    Recupere des emails Gmail et les convertit en Documents.

    Args:
        query: requete Gmail, ex: "newer_than:30d", "from:client@example.com".
        max_results: nombre maximum d'emails a recuperer.
        projet: metadata projet.
        lot_technique: metadata lot.
        criticite: metadata criticite.
    """
    settings = get_settings()
    query = query or settings.gmail_query
    max_results = max_results or settings.gmail_max_results

    service = _gmail_service()
    result = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results)
        .execute()
    )
    messages = result.get("messages", [])
    documents: list[Document] = []

    for item in messages:
        message = (
            service.users()
            .messages()
            .get(userId="me", id=item["id"], format="full")
            .execute()
        )
        payload = message.get("payload", {})
        headers = _headers_to_dict(payload.get("headers", []))
        body = _extract_text_from_payload(payload)
        content = _format_email_content(headers, body)

        date_value = headers.get("date", "")
        try:
            parsed_date = parsedate_to_datetime(date_value).date().isoformat()
        except Exception:
            parsed_date = ""

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": f"gmail:{message.get('id')}",
                    "gmail_id": message.get("id"),
                    "thread_id": message.get("threadId"),
                    "projet": projet,
                    "lot_technique": lot_technique,
                    "type_document": "email",
                    "auteur": headers.get("from", "inconnu"),
                    "criticite": criticite,
                    "date": parsed_date,
                    "email_from": headers.get("from", ""),
                    "email_to": headers.get("to", ""),
                    "email_subject": headers.get("subject", ""),
                    "gmail_query": query,
                },
            )
        )

    print(f"[gmail] {len(documents)} emails collectes avec la requete '{query}'")
    return documents
