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
import imaplib
import json
import logging
import os
import secrets
import ssl
from datetime import datetime, timedelta
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import Message
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
_runtime_gmail_token: Optional[str] = None


def _base64url_no_padding(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _credentials_paths() -> tuple[Path, Path]:
    settings = get_settings()
    return Path(settings.google_credentials_file), Path(settings.google_token_file)


def _load_token_from_env(token_json: str) -> Optional[Credentials]:
    try:
        token_info = json.loads(token_json)
        return Credentials.from_authorized_user_info(token_info, GMAIL_SCOPES)
    except Exception as exc:
        logger.warning("GMAIL_TOKEN est present mais invalide: %s", exc)
        return None


def _current_gmail_token() -> Optional[str]:
    settings = get_settings()
    return os.getenv("GMAIL_TOKEN") or _runtime_gmail_token or settings.gmail_token


def _load_client_config_from_env(credentials_json: str) -> Optional[dict]:
    try:
        client_config = json.loads(credentials_json)
        if not isinstance(client_config, dict):
            raise ValueError("JSON racine invalide")
        if "installed" not in client_config and "web" not in client_config:
            raise ValueError("JSON OAuth Google attendu avec cle 'installed' ou 'web'")
        return client_config
    except Exception as exc:
        logger.warning("GMAIL_CREDENTIALS est present mais invalide: %s", exc)
        return None


def _save_gmail_token(creds: Credentials, token_path: Path) -> None:
    global _runtime_gmail_token
    settings = get_settings()
    token_json = creds.to_json()
    _runtime_gmail_token = token_json
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
    if settings.google_redirect_uri.startswith(("http://localhost", "http://127.0.0.1")):
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

    if settings.gmail_credentials:
        client_config = _load_client_config_from_env(settings.gmail_credentials)
        if not client_config:
            raise RuntimeError("Variable GMAIL_CREDENTIALS invalide.")
        return Flow.from_client_config(
            client_config,
            scopes=GMAIL_SCOPES,
            redirect_uri=settings.google_redirect_uri,
            state=state,
        )

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Fichier Google credentials introuvable : {credentials_path}. "
            "Telecharge le fichier OAuth client depuis Google Cloud et place-le ici, "
            "ou configure GMAIL_CREDENTIALS sur Railway."
        )

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
    gmail_token = _current_gmail_token()

    if gmail_token:
        creds = _load_token_from_env(gmail_token)
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


def gmail_configuration_status() -> dict:
    """Retourne l'etat de configuration Gmail sans exposer les secrets."""
    settings = get_settings()
    credentials_path, token_path = _credentials_paths()
    token_configured = bool(_current_gmail_token()) or token_path.exists()
    credentials_configured = bool(settings.gmail_credentials) or credentials_path.exists()
    provider_configured = credentials_configured or token_configured
    connected = has_valid_gmail_token()
    return {
        "provider": "gmail",
        "providers_configures": ["gmail"] if provider_configured else [],
        "provider_configured": provider_configured,
        "credentials_configured": credentials_configured,
        "token_configured": token_configured,
        "connected": connected,
        "need_auth": not connected,
    }


def generate_auth_url(state: Optional[str] = None, code_verifier: Optional[str] = None) -> dict[str, str]:
    flow = _build_flow(state=state)
    code_verifier = code_verifier or _base64url_no_padding(secrets.token_bytes(64))
    code_challenge = _base64url_no_padding(
        hashlib.sha256(code_verifier.encode("ascii")).digest()
    )
    auth_url, returned_state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
        code_challenge=code_challenge,
        code_challenge_method="S256",
    )
    return {
        "auth_url": auth_url,
        "state": returned_state,
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


def _decode_mime_header(value: str) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _strip_html(text: str) -> str:
    import re

    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _extract_text_from_payload(payload: dict) -> str:
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if body_data and mime_type in {"text/plain", "text/html"}:
        text = _decode_base64url(body_data)
        if mime_type == "text/html":
            text = _strip_html(text)
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


def _extract_text_from_email_message(message: Message) -> str:
    parts: list[str] = []
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", "")).lower()
            if "attachment" in disposition or content_type not in {"text/plain", "text/html"}:
                continue
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            parts.append(_strip_html(text) if content_type == "text/html" else text)
    else:
        payload = message.get_payload(decode=True)
        if payload:
            charset = message.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            parts.append(_strip_html(text) if message.get_content_type() == "text/html" else text)
    return "\n\n".join(part.strip() for part in parts if part.strip())


def collecter_imap(
    host: str,
    port: int,
    email_address: str,
    password: str,
    folder: str = "INBOX",
    days: int = 30,
    max_results: int = 20,
    use_ssl: bool = True,
    projet: str = "non_defini",
    lot_technique: str = "non_defini",
    criticite: str = "normale",
) -> list[Document]:
    """
    Recupere des emails via IMAP et les convertit en Documents.
    Utile pour tester sans OAuth avec un mot de passe d'application.
    """
    since = (datetime.utcnow() - timedelta(days=max(1, days))).strftime("%d-%b-%Y")
    if use_ssl:
        mail = imaplib.IMAP4_SSL(host, port, ssl_context=ssl.create_default_context())
    else:
        mail = imaplib.IMAP4(host, port)

    try:
        mail.login(email_address, password)
        status, _ = mail.select(folder)
        if status != "OK":
            raise RuntimeError(f"Dossier IMAP introuvable: {folder}")

        status, data = mail.search(None, "SINCE", since)
        if status != "OK":
            raise RuntimeError("Recherche IMAP impossible.")

        message_ids = (data[0] or b"").split()
        message_ids = list(reversed(message_ids))[: max(1, max_results)]
        documents: list[Document] = []

        for message_id in message_ids:
            status, fetched = mail.fetch(message_id, "(RFC822)")
            if status != "OK" or not fetched:
                continue
            raw_email = next((item[1] for item in fetched if isinstance(item, tuple)), None)
            if not raw_email:
                continue

            parsed = message_from_bytes(raw_email)
            headers = {
                "from": _decode_mime_header(parsed.get("From", "")),
                "to": _decode_mime_header(parsed.get("To", "")),
                "date": parsed.get("Date", ""),
                "subject": _decode_mime_header(parsed.get("Subject", "")),
            }
            body = _extract_text_from_email_message(parsed)
            content = _format_email_content(headers, body)

            date_value = headers.get("date", "")
            try:
                parsed_date = parsedate_to_datetime(date_value).date().isoformat()
            except Exception:
                parsed_date = ""

            imap_uid = f"{email_address}:{folder}:{message_id.decode('ascii', errors='ignore')}"
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": f"imap:{hashlib.md5(imap_uid.encode('utf-8')).hexdigest()}",
                        "email_source_id": imap_uid,
                        "provider": "imap",
                        "projet": projet,
                        "lot_technique": lot_technique,
                        "type_document": "email",
                        "auteur": headers.get("from", "inconnu"),
                        "criticite": criticite,
                        "date": parsed_date,
                        "email_from": headers.get("from", ""),
                        "email_to": headers.get("to", ""),
                        "email_subject": headers.get("subject", ""),
                        "imap_folder": folder,
                    },
                )
            )
    finally:
        try:
            mail.logout()
        except Exception:
            pass

    print(f"[imap] {len(documents)} emails collectes depuis {email_address}/{folder}")
    return documents
