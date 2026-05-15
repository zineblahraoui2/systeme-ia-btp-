from __future__ import annotations

import socket
import time

import requests
import streamlit as st

import config

try:
    import plotly.express as px
except ImportError:
    px = None


API_URL = config.BACKEND_URL
DEFAULT_API_URL = API_URL
API_TIMEOUT_SECONDS = 120


st.set_page_config(
    page_title="Système IA BTP",
    page_icon="🏗️",
    layout="wide",
)


st.markdown(
    """
<style>
.main-header { color: #1e3a5f; font-size: 2rem; font-weight: 800; margin-bottom: 0.15rem; }
.main-subtitle { color: #6c757d; font-size: 0.98rem; margin-bottom: 0.75rem; }
.header-rule { height: 3px; width: 100%; background: linear-gradient(90deg, #1e3a5f, #2ecc71, #e67e22); border-radius: 4px; margin: 0.5rem 0 1.4rem 0; }
.section-title { color: #1e3a5f; font-weight: 750; margin-top: 0.6rem; }
.section-title:after { content: ""; display: block; width: 52px; height: 3px; background: #2ecc71; border-radius: 2px; margin-top: 6px; }
.answer-box { background: #f8f9fa; border-left: 4px solid #2ecc71; border-radius: 8px; padding: 16px; margin-top: 12px; }
.api-online { color: #2ecc71; font-weight: 700; }
.api-offline { color: #e74c3c; font-weight: 700; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { color: #495057; border-bottom: 2px solid transparent; transition: all 0.2s; }
.stTabs [aria-selected="true"] { color: #1e3a5f !important; border-bottom: 3px solid #1e3a5f; font-weight: 700; }
.stTabs [data-baseweb="tab"]:hover { color: #1e3a5f; background: rgba(30,58,95,0.05); }
.stButton > button { border-radius: 8px; font-weight: 600; transition: all 0.2s; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
.stTextInput > div > div > input, .stTextArea > div > div > textarea { border-radius: 6px; border: 1px solid #dee2e6; }
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: #1e3a5f; box-shadow: 0 0 0 2px rgba(30,58,95,0.1); }
[data-testid="metric-container"] { background: white; border-left: 4px solid #1e3a5f; border-radius: 8px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
[data-testid="stSidebar"] { background: #f8f9fa; }
[data-testid="stSidebar"] .stTextInput input { border-radius: 8px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; color: white; }
</style>
""",
    unsafe_allow_html=True,
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


def render_markdown_response(data: dict) -> None:
    answer = data.get("reponse", data)
    if isinstance(answer, (dict, list)):
        import json

        content = f"```json\n{json.dumps(answer, ensure_ascii=False, indent=2)}\n```"
    else:
        content = str(answer)
    st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
    st.markdown(content)
    st.markdown("</div>", unsafe_allow_html=True)


def _html_badge(value: str, color: str) -> str:
    return f"<span class='badge' style='background:{color}'>{value}</span>"


def _badge_color(kind: str, value: str) -> str:
    normalized = (value or "").strip().lower()
    if kind == "criticite":
        colors = {
            "critique": "#e74c3c",
            "élevée": "#e67e22",
            "elevee": "#e67e22",
            "haute": "#f39c12",
            "normale": "#2ecc71",
        }
        return colors.get(normalized, "#7f8c8d")
    colors = {
        "DTU": "#2c3e50",
        "BIM": "#8e44ad",
        "EMAIL": "#16a085",
        "PDF": "#c0392b",
        "IMAGE": "#27ae60",
        "KNOWLEDGE": "#3498db",
    }
    return colors.get((value or "").upper(), "#3498db")


def _truncate(value: str, limit: int = 40) -> str:
    value = str(value or "")
    return value if len(value) <= limit else value[: limit - 1] + "…"


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


st.markdown("<div class='main-header'>🏗️ Système IA BTP</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='main-subtitle'>Interface IA pour ingestion, recherche, conformité, risques et recommandations chantier.</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='header-rule'></div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("🔌 Connexion API")
    current_api_url = st.session_state.get("api_base_url", "")
    if current_api_url and current_api_url != DEFAULT_API_URL and "localhost" in current_api_url:
        st.session_state.api_base_url = DEFAULT_API_URL
    st.session_state.api_base_url = st.text_input(
        "URL API",
        value=st.session_state.get("api_base_url", DEFAULT_API_URL),
    )
    sync_gmail_status(force=True)
    try:
        api_status = requests.get(api_url("/health"), timeout=8).ok
    except requests.RequestException:
        api_status = False
    status_class = "api-online" if api_status else "api-offline"
    status_text = "● Online" if api_status else "● Offline"
    st.markdown(f"<span class='{status_class}'>{status_text}</span>", unsafe_allow_html=True)

    if st.button("Tester /health", use_container_width=True):
        try:
            response = requests.get(api_url("/health"), timeout=API_TIMEOUT_SECONDS)
            show_response(response)
        except requests.RequestException as exc:
            st.error(f"API inaccessible : {exc}")

    st.divider()
    st.write("Démarre l'API avec :")
    st.code(f"py -m uvicorn main:app --reload --port {config.API_PORT}")


tab_chat, tab_ingest, tab_reglementaire, tab_analysis, tab_knowledge, tab_admin = st.tabs(
    ["Question", "Ingestion", "Base Réglementaire", "Analyse", "🧠 Mémoire Projet", "Admin"]
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
                if response.ok:
                    st.success("Requête réussie")
                    render_markdown_response(response.json())
                else:
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


with tab_knowledge:
    st.markdown("<div class='section-title'>Mémoire Projet IA</div>", unsafe_allow_html=True)
    try:
        response = requests.get(api_url("/knowledge/documents"), timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
        knowledge = response.json()
    except requests.RequestException as exc:
        st.error(f"Erreur API : {exc}")
        knowledge = {"total_chunks": 0, "total_documents": 0, "vector_store": "chroma", "derniere_ingestion": "", "documents": []}

    metric_cols = st.columns(4)
    metric_cols[0].metric("📦 Chunks total", knowledge.get("total_chunks", 0))
    metric_cols[1].metric("📄 Documents total", knowledge.get("total_documents", 0))
    metric_cols[2].metric("🗄️ Vector Store", knowledge.get("vector_store", "chroma"))
    metric_cols[3].metric("🕐 Dernière ingestion", knowledge.get("derniere_ingestion") or "-")

    documents = knowledge.get("documents", []) or []
    filter_cols = st.columns(4)
    criticites = sorted({doc.get("criticite", "") for doc in documents if doc.get("criticite")})
    file_types = sorted({doc.get("file_type", "") for doc in documents if doc.get("file_type")})
    projects = sorted({doc.get("project", "") for doc in documents if doc.get("project")})
    with filter_cols[0]:
        selected_criticites = st.multiselect("Criticité", criticites)
    with filter_cols[1]:
        selected_types = st.multiselect("File type", file_types)
    with filter_cols[2]:
        selected_projects = st.multiselect("Projet", projects)
    with filter_cols[3]:
        source_query = st.text_input("Recherche source")

    filtered_docs = []
    for doc in documents:
        if selected_criticites and doc.get("criticite") not in selected_criticites:
            continue
        if selected_types and doc.get("file_type") not in selected_types:
            continue
        if selected_projects and doc.get("project") not in selected_projects:
            continue
        if source_query and source_query.lower() not in str(doc.get("source", "")).lower():
            continue
        filtered_docs.append(doc)

    rows = []
    for doc in filtered_docs:
        criticite = doc.get("criticite", "normale")
        file_type = doc.get("file_type", "knowledge")
        rows.append(
            {
                "source": _truncate(doc.get("source", ""), 40),
                "project": doc.get("project", ""),
                "lot": doc.get("lot", ""),
                "auteur": doc.get("auteur", ""),
                "criticite": _html_badge(criticite, _badge_color("criticite", criticite)),
                "file_type": _html_badge(file_type, _badge_color("file_type", file_type)),
                "ingested_at": doc.get("ingested_at", ""),
                "chunk_count": doc.get("chunk_count", 0),
            }
        )
    if rows:
        import pandas as pd

        st.markdown(pd.DataFrame(rows).to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("Aucun document ne correspond aux filtres.")

    if filtered_docs and px is not None:
        import pandas as pd

        chart_df = pd.DataFrame(filtered_docs)
        graph_cols = st.columns(2)
        with graph_cols[0]:
            file_counts = chart_df.groupby("file_type", as_index=False)["chunk_count"].sum()
            st.plotly_chart(px.pie(file_counts, names="file_type", values="chunk_count", title="Répartition par file_type"), use_container_width=True)
        with graph_cols[1]:
            project_chunks = chart_df.groupby("project", as_index=False)["chunk_count"].sum().sort_values("chunk_count")
            st.plotly_chart(px.bar(project_chunks, x="chunk_count", y="project", orientation="h", title="Chunks par projet"), use_container_width=True)
    elif px is None:
        st.warning("Plotly n'est pas installé. Ajoute plotly puis redéploie.")

    st.divider()
    st.warning("Action irréversible : cette action vide toute la collection ChromaDB.")
    confirm_clear = st.checkbox("Je confirme vouloir vider la base vectorielle", key="confirm_clear_knowledge")
    if st.button("🗑️ Vider la base", disabled=not confirm_clear):
        try:
            delete_response = requests.delete(api_url("/knowledge/documents"), timeout=API_TIMEOUT_SECONDS)
            show_response(delete_response)
            if delete_response.ok:
                st.rerun()
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
