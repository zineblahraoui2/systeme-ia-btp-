from __future__ import annotations

import socket
import time

import requests
import streamlit as st

import config


API_URL = config.BACKEND_URL
DEFAULT_API_URL = API_URL
API_TIMEOUT_SECONDS = 120


st.set_page_config(
    page_title="Système IA BTP",
    page_icon="🏗️",
    layout="wide",
)


def api_url(path: str) -> str:
    base_url = st.session_state.get("api_base_url", DEFAULT_API_URL).rstrip("/")
    return f"{base_url}{path}"


def auth_url(path: str) -> str:
    base_url = st.session_state.get("api_base_url", DEFAULT_API_URL).rstrip("/")
    if base_url.endswith("/api/v1"):
        base_url = base_url[: -len("/api/v1")]
    return f"{base_url}{path}"


def sync_gmail_status(force: bool = False) -> bool:
    if not force and "gmail_connected" in st.session_state:
        return bool(st.session_state.gmail_connected)
    try:
        response = requests.get(auth_url("/auth/gmail/status"), timeout=API_TIMEOUT_SECONDS)
        if response.ok:
            data = response.json()
            st.session_state.gmail_connected = bool(data.get("connected"))
        else:
            st.session_state.gmail_connected = False
    except requests.RequestException:
        st.session_state.gmail_connected = False
    return bool(st.session_state.gmail_connected)


def show_response(response: requests.Response) -> None:
    try:
        data = response.json()
    except ValueError:
        st.text(response.text)
        return

    if response.ok:
        st.success("Requête réussie")
        st.json(data)
    else:
        st.error(f"Erreur {response.status_code}")
        st.json(data)


def show_pipeline_badge(pipeline: str | None) -> None:
    labels = {
        "pdf_texte": ("Extraction directe PDF", "green"),
        "pdf_ocr": ("OCR Tesseract PDF scanné", "blue"),
        "image_ocr": ("OCR Tesseract image", "orange"),
        "image_openai": ("OpenAI vision chantier", "violet"),
        "bim_ifc": ("BIM IFC", "violet"),
        "image_gemini": ("Gemini vision chantier", "violet"),
        "image_clip": ("CLIP vision photo", "violet"),
        "image_blip": ("Description BLIP", "violet"),
        "image_analyse_en_cours": ("Analyse image en cours", "gray"),
    }
    if not pipeline:
        return
    label, color = labels.get(pipeline, (pipeline, "gray"))
    if color == "green":
        st.success(label)
    elif color == "blue":
        st.info(label)
    elif color == "orange":
        st.warning(label)
    elif color == "violet":
        st.markdown(f"**Pipeline utilise :** :violet[{label}]")
    elif color == "gray":
        st.markdown(f"**Pipeline utilise :** {label}")
    else:
        st.write(f"Pipeline utilise : {label}")


def poll_job(job_id: str, fichier_nom: str) -> None:
    with st.spinner("Analyse visuelle IA en cours..."):
        for _ in range(36):
            time.sleep(5)
            try:
                response = requests.get(api_url(f"/job/{job_id}"), timeout=10)
                job = response.json()
            except Exception:
                continue

            statut = job.get("statut")
            if statut == "termine":
                st.success(f"{fichier_nom} : analyse terminee")
                description = (
                    job.get("description_openai")
                    or job.get("description_gemini")
                    or job.get("description_blip")
                )
                if description:
                    st.markdown(f"**Description visuelle :** {description}")

                cols = st.columns(3)
                cols[0].metric("Pipeline", job.get("pipeline_utilise", "-"))
                cols[1].metric("Vision", job.get("pipeline_utilise", "-"))
                cols[2].metric(
                    "Vecteurs",
                    job.get("stats_collection", {}).get("nombre_vecteurs", "-"),
                )
                st.json(job)
                return

            if statut == "erreur":
                st.error(f"Erreur analyse image : {job.get('message', 'Erreur inconnue')}")
                return

    st.warning("L'analyse continue en arriere-plan. Consulte les stats ou relance plus tard.")


st.title("Système IA BTP")
st.caption("Interface de test pour ingestion, recherche, conformité, risques et recommandations.")

with st.sidebar:
    st.header("Connexion API")
    current_api_url = st.session_state.get("api_base_url", "")
    if current_api_url and current_api_url != DEFAULT_API_URL and "localhost" in current_api_url:
        st.session_state.api_base_url = DEFAULT_API_URL
    st.session_state.api_base_url = st.text_input(
        "URL API",
        value=st.session_state.get("api_base_url", DEFAULT_API_URL),
    )
    sync_gmail_status(force=True)

    if st.button("Tester /health", use_container_width=True):
        try:
            response = requests.get(api_url("/health"), timeout=API_TIMEOUT_SECONDS)
            show_response(response)
        except requests.RequestException as exc:
            st.error(f"API inaccessible : {exc}")

    st.divider()
    st.write("Démarre l'API avec :")
    st.code(f"py -m uvicorn main:app --reload --port {config.API_PORT}")


tab_chat, tab_ingest, tab_reglementaire, tab_analysis, tab_admin = st.tabs(
    ["Question", "Ingestion", "Base Réglementaire", "Analyse", "Admin"]
)


with tab_chat:
    st.subheader("Question sur la base documentaire")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        question = st.text_area(
            "Question",
            placeholder="Ex: Quelles sont les non-conformités relevées sur le lot électricité ?",
            height=140,
        )
    with col_b:
        projet = st.text_input("Projet (optionnel)", key="question_projet")
        k = st.number_input("Nombre de chunks", min_value=1, max_value=20, value=6)

    if st.button("Poser la question", type="primary"):
        if not question.strip():
            st.warning("Saisis une question.")
        else:
            payload = {"question": question, "projet": projet or None, "k": int(k)}
            try:
                response = requests.post(api_url("/question"), json=payload, timeout=API_TIMEOUT_SECONDS)
                show_response(response)
            except requests.RequestException as exc:
                st.error(f"Erreur API : {exc}")


with tab_ingest:
    st.subheader("Ajouter des documents")

    upload_col, text_col = st.columns(2)

    with upload_col:
        st.markdown("**Fichier**")
        fichier = st.file_uploader(
            "PDF, DOCX, TXT, MD, IFC ou image",
            type=["pdf", "docx", "txt", "md", "ifc", "jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
        )
        projet_fichier = st.text_input("Projet", value="non_defini", key="file_project")
        lot_fichier = st.text_input("Lot technique", value="non_defini", key="file_lot")
        auteur_fichier = st.text_input("Auteur", value="inconnu", key="file_author")
        criticite_fichier = st.selectbox(
            "Criticité",
            ["normale", "haute", "critique", "faible"],
            key="file_criticite",
        )

        if st.button("Ingérer le fichier"):
            if fichier is None:
                st.warning("Choisis un fichier.")
            else:
                files = {"fichier": (fichier.name, fichier.getvalue(), fichier.type)}
                data = {
                    "projet": projet_fichier,
                    "lot_technique": lot_fichier,
                    "auteur": auteur_fichier,
                    "criticite": criticite_fichier,
                }
                try:
                    response = requests.post(
                        api_url("/ingerer/fichier"),
                        files=files,
                        data=data,
                        timeout=API_TIMEOUT_SECONDS,
                    )
                    if not response.ok:
                        show_response(response)
                    else:
                        result = response.json()
                        show_pipeline_badge(result.get("pipeline_utilise"))

                        if result.get("statut") == "en_cours":
                            st.success(f"{result.get('fichier', fichier.name)} recu")
                            apercu = result.get("texte_ocr_apercu", "").strip()
                            if apercu:
                                st.info(f"Texte OCR detecte : {apercu[:150]}")
                            else:
                                st.info("Pas de texte detecte. Analyse visuelle IA lancee.")

                            job_id = result.get("job_id")
                            if job_id:
                                poll_job(job_id, result.get("fichier", fichier.name))
                        else:
                            if result.get("pipeline_utilise") == "dtu_norme":
                                st.success("Document réglementaire ingéré")
                                st.json(result.get("resume_reglementaire", {}))
                            if result.get("pipeline_utilise") == "bim_ifc":
                                resume = result.get("resume_bim") or {}
                                st.success("Maquette BIM IFC ingeree")
                                st.write(f"**Projet BIM :** {resume.get('projet_bim', '-')}")
                                st.write(f"**Schema IFC :** {resume.get('schema', '-')}")
                                st.write(f"**Auteur :** {resume.get('auteur', '-')}")
                                st.write(f"**Date :** {resume.get('date_creation', '-')}")
                                st.write(f"**Mode :** {resume.get('mode', '-')}")
                                st.json(resume.get("elements_extraits", {}))
                            show_response(response)
                except requests.RequestException as exc:
                    st.error(f"Erreur API : {exc}")

    with text_col:
        st.markdown("**Texte brut**")
        contenu = st.text_area("Contenu", height=190, placeholder="Colle ici un email, CR chantier, note terrain...")
        source = st.text_input("Source", value="saisie_manuelle")
        projet_texte = st.text_input("Projet", value="non_defini", key="text_project")
        lot_texte = st.text_input("Lot technique", value="non_defini", key="text_lot")
        type_document = st.selectbox(
            "Type document",
            ["general", "email", "whatsapp", "rapport_chantier", "devis", "dtu", "norme", "reglementation"],
        )

        if st.button("Ingérer le texte"):
            if not contenu.strip():
                st.warning("Saisis un contenu.")
            else:
                payload = {
                    "contenu": contenu,
                    "source": source,
                    "projet": projet_texte,
                    "lot_technique": lot_texte,
                    "type_document": type_document,
                }
                try:
                    response = requests.post(api_url("/ingerer/texte"), json=payload, timeout=API_TIMEOUT_SECONDS)
                    show_response(response)
                except requests.RequestException as exc:
                    st.error(f"Erreur API : {exc}")

    st.divider()
    st.markdown("**Dossier local**")
    dossier = st.text_input("Chemin du dossier à ingérer")
    dossier_projet = st.text_input("Projet", value="non_defini", key="folder_project")
    dossier_lot = st.text_input("Lot technique", value="non_defini", key="folder_lot")
    if st.button("Ingérer le dossier"):
        if not dossier.strip():
            st.warning("Indique un chemin de dossier.")
        else:
            payload = {
                "dossier": dossier,
                "projet": dossier_projet,
                "lot_technique": dossier_lot,
            }
            try:
                response = requests.post(api_url("/ingerer/dossier"), json=payload, timeout=API_TIMEOUT_SECONDS)
                show_response(response)
            except requests.RequestException as exc:
                st.error(f"Erreur API : {exc}")

    st.divider()
    st.markdown("**Gmail**")
    gmail_connected = sync_gmail_status()
    if gmail_connected:
        st.success("Gmail connectÃ©")
    else:
        st.info("Gmail non connectÃ©")
    gmail_query = st.text_input(
        "Requête Gmail",
        value="newer_than:30d (BTP OR chantier OR travaux OR béton OR fondation OR lot OR retard OR livraison OR fournisseur OR maçonnerie OR gros_oeuvre)",
        help="Exemples : newer_than:30d, from:client@example.com, subject:chantier",
    )
    gmail_max_results = st.number_input(
        "Nombre d'emails",
        min_value=1,
        max_value=100,
        value=20,
    )
    gmail_projet = st.text_input("Projet", value="non_defini", key="gmail_project")
    gmail_lot = st.text_input("Lot technique", value="non_defini", key="gmail_lot")
    gmail_criticite = st.selectbox(
        "Criticité",
        ["normale", "haute", "critique", "faible"],
        key="gmail_criticite",
    )

    if st.button("Ingérer les emails Gmail"):
        payload = {
            "query": gmail_query or None,
            "max_results": int(gmail_max_results),
            "projet": gmail_projet,
            "lot_technique": gmail_lot,
            "criticite": gmail_criticite,
        }
        try:
            with st.spinner("Vérification de la connexion Gmail..."):
                auth_response = requests.get(auth_url("/auth/gmail/login"), timeout=API_TIMEOUT_SECONDS)
            auth_response.raise_for_status()
            auth_data = auth_response.json()
            st.session_state.gmail_connected = not bool(auth_data.get("need_auth"))

            if auth_data.get("need_auth"):
                st.warning("Connexion Gmail requise.")
                st.link_button("Connecter Gmail", auth_data["auth_url"])
            else:
                with st.spinner("Ingestion des emails Gmail..."):
                    response = requests.post(api_url("/ingerer/gmail"), json=payload, timeout=API_TIMEOUT_SECONDS)
                try:
                    data = response.json()
                except ValueError:
                    show_response(response)
                else:
                    if response.ok and data.get("need_auth"):
                        st.session_state.gmail_connected = False
                        st.warning("Connexion Gmail requise.")
                        st.link_button("Connecter Gmail", data["auth_url"])
                    else:
                        if response.ok:
                            st.session_state.gmail_connected = True
                        show_response(response)
        except requests.RequestException as exc:
            st.error(f"Erreur API : {exc}")


with tab_reglementaire:
    st.subheader("Base réglementaire DTU / NF / EN / ISO")
    reg_upload, reg_search = st.columns(2)

    with reg_upload:
        st.markdown("**Ingestion PDF réglementaire**")
        norme_pdf = st.file_uploader("PDF DTU, NF, EN, ISO ou Eurocode", type=["pdf"], key="dtu_pdf_upload")
        auteur_norme = st.text_input("Auteur / source", value="reglementaire", key="dtu_author")
        if st.button("Ingérer la norme", key="ingest_dtu"):
            if norme_pdf is None:
                st.warning("Choisis un PDF réglementaire.")
            else:
                files = {"fichier": (norme_pdf.name, norme_pdf.getvalue(), norme_pdf.type)}
                data = {
                    "projet": "reglementaire",
                    "lot_technique": "reglementation",
                    "auteur": auteur_norme,
                    "criticite": "haute",
                }
                try:
                    response = requests.post(
                        api_url("/ingerer/fichier"),
                        files=files,
                        data=data,
                        timeout=API_TIMEOUT_SECONDS,
                    )
                    show_response(response)
                except requests.RequestException as exc:
                    st.error(f"Erreur API : {exc}")

        if st.button("Lister les normes ingérées", key="list_dtu"):
            try:
                response = requests.get(api_url("/dtu/list"), timeout=API_TIMEOUT_SECONDS)
                show_response(response)
            except requests.RequestException as exc:
                st.error(f"Erreur API : {exc}")

    with reg_search:
        st.markdown("**Recherche réglementaire**")
        exemple = st.selectbox(
            "Exemples",
            [
                "Quelle épaisseur minimale pour un mur porteur ?",
                "Cette dalle est-elle conforme ?",
                "résistance feu escalier",
                "distance ferraillage dalle",
            ],
        )
        requete_reg = st.text_area("Requête", value=exemple, height=100)
        k_reg = st.number_input("Nombre de citations", min_value=1, max_value=20, value=6, key="k_reg")

        if st.button("Rechercher dans les normes", key="search_dtu"):
            try:
                response = requests.post(
                    api_url("/dtu/search"),
                    json={"query": requete_reg, "k": int(k_reg)},
                    timeout=API_TIMEOUT_SECONDS,
                )
                show_response(response)
            except requests.RequestException as exc:
                st.error(f"Erreur API : {exc}")

        st.markdown("**Check conformité**")
        description_travaux = st.text_area("Description travaux", value="mur porteur béton de 15 cm", height=90)
        if st.button("Vérifier conformité", key="check_dtu"):
            try:
                response = requests.post(
                    api_url("/dtu/check-conformity"),
                    json={"description_travaux": description_travaux, "k": int(k_reg)},
                    timeout=API_TIMEOUT_SECONDS,
                )
                show_response(response)
            except requests.RequestException as exc:
                st.error(f"Erreur API : {exc}")


with tab_analysis:
    st.subheader("Analyses métier")
    situation = st.text_area(
        "Situation ou élément à analyser",
        height=140,
        placeholder="Ex: Le rapport indique un défaut de mise à la terre dans le local technique.",
    )
    projet_analyse = st.text_input("Projet (optionnel)", key="analysis_project")

    action = st.radio(
        "Type d'analyse",
        ["Conformité", "Risques", "Recommandations", "Analyse complète"],
        horizontal=True,
    )

    if st.button("Lancer l'analyse", type="primary"):
        if not situation.strip():
            st.warning("Saisis une situation.")
        else:
            endpoints = {
                "Conformité": ("/conformite", {"element": situation}),
                "Risques": ("/risques", {"situation": situation, "projet": projet_analyse or None}),
                "Recommandations": ("/recommandations", {"situation": situation, "projet": projet_analyse or None}),
                "Analyse complète": ("/analyser", {"situation": situation, "projet": projet_analyse or None}),
            }
            path, payload = endpoints[action]
            try:
                response = requests.post(api_url(path), json=payload, timeout=API_TIMEOUT_SECONDS)
                show_response(response)
            except requests.RequestException as exc:
                st.error(f"Erreur API : {exc}")


with tab_admin:
    st.subheader("Base vectorielle et alertes")
    col_stats, col_alerts = st.columns(2)

    with col_stats:
        if st.button("Voir les statistiques", use_container_width=True):
            try:
                response = requests.get(api_url("/stats"), timeout=API_TIMEOUT_SECONDS)
                show_response(response)
            except requests.RequestException as exc:
                st.error(f"Erreur API : {exc}")

    with col_alerts:
        alert_project = st.text_input("Projet pour alertes (optionnel)")
        if st.button("Voir les alertes", use_container_width=True):
            params = {"projet": alert_project} if alert_project else None
            try:
                response = requests.get(api_url("/alertes"), params=params, timeout=API_TIMEOUT_SECONDS)
                show_response(response)
            except requests.RequestException as exc:
                st.error(f"Erreur API : {exc}")
