"""
couche_ia/llm_engine.py
-----------------------
Moteur LLM direct compatible OpenAI / GitHub Models, avec fallback extractif.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from config import get_settings


def _has_llm_key() -> bool:
    return bool(get_settings().openai_api_key.strip())


def _openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=120.0,
    )


class LLMEngine:
    """Wraps an OpenAI-compatible Chat Completions client with a local fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self._use_openai = _has_llm_key()
        self._openai_model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = min(settings.llm_max_tokens, 1200)

    def generate(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        """Generate an answer grounded in the provided context chunks."""
        context = self._build_context(context_chunks)

        system_prompt = (
            "Tu es BTP AI, un assistant expert en construction et genie civil "
            "(Batiment et Travaux Publics). Tu maitrises les DTU, les normes NF EN, "
            "les Eurocodes, la reglementation francaise (RE 2020, Code du Travail, "
            "securite incendie, amiante) et les bonnes pratiques metier BTP.\n\n"
            "REGLES ABSOLUES :\n"
            "1. Reponds UNIQUEMENT a partir des documents de contexte fournis.\n"
            "2. Si l'information n'est pas dans le contexte, dis clairement : "
            "'Cette information n'est pas disponible dans les documents charges.'\n"
            "3. Cite toujours la reference du document source quand elle est disponible.\n"
            "4. Sois precis, professionnel et structure.\n"
            "5. Pour les valeurs numeriques, cite-les exactement.\n"
            "6. Reponds en francais."
        )

        user_prompt = (
            f"Documents de contexte :\n{context}\n\n"
            f"Question : {question}\n\n"
            "Fournis une reponse claire et detaillee uniquement a partir du contexte."
        )

        if self._use_openai:
            return self._openai_generate(system_prompt, user_prompt)
        return self._fallback_generate(context_chunks)

    def _openai_generate(self, system: str, user: str) -> str:
        client = _openai_client()
        response = client.chat.completions.create(
            model=self._openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def _fallback_generate(self, chunks: list[dict[str, Any]]) -> str:
        """Fallback extractif quand aucune cle API LLM n'est configuree."""
        if not chunks:
            return "Aucune information pertinente trouvee dans les documents charges."

        answer_parts = [
            "Aucune cle API LLM configuree - passages extraits directement :\n"
        ]
        for i, chunk in enumerate(chunks[:3], 1):
            source = chunk["metadata"].get("source", "Inconnu")
            answer_parts.append(
                f"**Passage {i}** (source : {source}) :\n{chunk['text']}\n"
            )

        return "\n".join(answer_parts)

    @staticmethod
    def _build_context(chunks: list[dict[str, Any]]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown")
            project = metadata.get("projet", metadata.get("project", ""))
            type_document = metadata.get("type_document", "N/A")
            lot = metadata.get("lot_technique", "N/A")
            parts.append(
                f"[Document {i} | Source: {source} | Projet: {project} | "
                f"Type: {type_document} | Lot: {lot}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)


def documents_to_context_chunks(documents: list[Any]) -> list[dict[str, Any]]:
    """Convertit des Documents LangChain en chunks attendus par LLMEngine."""
    return [
        {
            "text": doc.page_content,
            "metadata": dict(doc.metadata or {}),
        }
        for doc in documents
    ]
