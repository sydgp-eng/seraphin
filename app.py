# app.py – SERAPHIN (SERA DV Assistant UI)

import os
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma  # pip install langchain-chroma


DB_DIR = "chroma_db"
MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0.1

SYSTEM_PROMPT = """
You are SERAPHIN, the SERA Trust Domestic Violence Information Assistant for Georgia.
Provide ONLY general information (not legal advice).
Remain calm, trauma-informed, supportive, and factual.
""".strip()


@st.cache_resource
def load_retriever_and_llm():
    """Load vector retriever and Groq LLM (cached by Streamlit)."""

    # Load .env for local development (has no effect on Streamlit Cloud)
    load_dotenv()

    # 1) Try Streamlit Secrets (Cloud)
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        # st.secrets may not exist in some local contexts; ignore
        api_key = None

    # 2) Fallback to OS environment (.env for local dev)
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it to your .env file for local runs OR to Streamlit 'Secrets' "
            "for the deployed app."
        )

    # Ensure the Groq SDK used inside ChatGroq can see the key
    os.environ["GROQ_API_KEY"] = api_key

    # Embeddings (local, no external API)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store + retriever
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Groq LLM – reads GROQ_API_KEY from environment
    llm = ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE)

    return retriever, llm


def format_context(docs: List[Document]) -> str:
    """Join retrieved chunks into a single context string."""
    return "\n\n".join(d.page_content.strip() for d in docs)


def answer_question(question: str, retriever, llm) -> str:
    """Run retrieval + LLM to answer a user question."""
    docs = retriever.invoke(question)
    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    # If nothing was retrieved, fall back to general guidance
    if not docs:
        user_msg = HumanMessage(
            content=(
                "No matching context found in the knowledge base.\n\n"
                f"Provide general, high-level information about domestic violence "
                f"processes and safe next steps in Georgia for this question:\n\n"
                f"{question}"
            )
        )
        return llm.invoke([system_msg, user_msg]).content

    # Use retrieved context
    context = format_context(docs)
    user_prompt = f"""
Use the context below to answer the question. If the context is not enough,
say that clearly first, then offer only general guidance and safe next steps.

Context:
{context}

Question:
{question}
""".strip()

    user_msg = HumanMessage(content=user_prompt)
    return llm.invoke([system_msg, user_msg]).content


def main():
    # ---- Page setup ----
    st.set_page_config(
        page_title="SERAPHIN – SERA DV Assistant",
        page_icon="💜",
        layout="centered",
    )

    st.title("💜 SERAPHIN – SERA Domestic Violence Information Assistant")
    st.write(
        "SERAPHIN provides **general information only** about domestic violence "
        "laws and processes in Georgia. This is **not legal advice**."
    )

    # ---- Load retriever + model (cached) ----
    try:
        retriever, llm = load_retriever_and_llm()
    except Exception as e:
        st.error(
            "Configuration error: SERAPHIN could not start. "
            "Please check the GROQ_API_KEY setting."
        )
        st.caption(str(e))
        return

    # ---- Chat history ----
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (question, answer)

    # ---- Input box ----
    question = st.text_input(
        "Ask SERAPHIN a question:",
        placeholder=(
            "Example: How do I apply for a Temporary Protective Order (TPO) "
            "in Fulton County?"
        ),
    )

    if st.button("Submit") and question.strip():
        with st.spinner("SERAPHIN is thinking..."):
            try:
                answer = answer_question(question.strip(), retriever, llm)
            except Exception:
                st.error(
                    "SERAPHIN ran into a technical problem while answering. "
                    "Please try again in a moment."
                )
            else:
                st.session_state.history.append((question.strip(), answer))

    # ---- Conversation history ----
    st.markdown("---")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Seraphin:** {a}")
        st.markdown("---")

    # ---- Footer ----
    st.caption(
        "Disclaimer: SERAPHIN gives general information only. It does not replace a "
        "lawyer, police, or certified domestic violence advocate. If you are in "
        "immediate danger, call 911."
    )


if __name__ == "__main__":
    main()
