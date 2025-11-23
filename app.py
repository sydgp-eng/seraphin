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

# ----------------- Constants -----------------

DB_DIR = "chroma_db"
MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0.1

SYSTEM_PROMPT = """
You are SERAPHIN, the SERA Trust Domestic Violence Information Assistant for Georgia.
Provide ONLY general information (not legal advice).
Remain calm, trauma-informed, supportive, and factual.
""".strip()


# ----------------- Helpers -----------------

def get_groq_api_key() -> str | None:
    """
    Try to get GROQ_API_KEY from:
    1) Local .env (for local dev)
    2) Streamlit Cloud secrets (for deployed app)
    """
    load_dotenv()  # loads .env if present

    # 1) Environment variable (local dev)
    api_key = os.getenv("GROQ_API_KEY")

    # 2) Streamlit secrets (Streamlit Cloud)
    if not api_key:
        try:
            if "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            # st.secrets might not be available in some contexts
            api_key = None

    return api_key


@st.cache_resource
def load_retriever_and_llm():
    """Create the Chroma retriever and Groq LLM, cached by Streamlit."""
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Set it in a local .env file OR in Streamlit Cloud Secrets."
        )

    # Make sure ChatGroq can see the key via env var as well
    os.environ["GROQ_API_KEY"] = api_key

    # Embeddings + Chroma DB
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Groq LLM
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=api_key,  # explicit, in case env var not picked up
    )

    return retriever, llm


def format_context(docs: List[Document]) -> str:
    """Join retrieved chunks into a single context string."""
    return "\n\n".join(d.page_content.strip() for d in docs)


def answer_question(question: str, retriever, llm) -> str:
    """RAG pipeline: retrieve, then ask Groq with a grounded prompt."""
    docs = retriever.invoke(question)
    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    # No documents found – fall back to generic info
    if not docs:
        user_msg = HumanMessage(
            content=f"No context found. Provide general Georgia DV info for:\n\n{question}"
        )
        return llm.invoke([system_msg, user_msg]).content

    context = format_context(docs)
    user_prompt = f"""
Use this context to answer the question. If the context is not enough, say so and
offer ONLY general guidance and safe next steps. Do NOT give legal advice.

Context:
{context}

Question:
{question}
""".strip()

    user_msg = HumanMessage(content=user_prompt)
    return llm.invoke([system_msg, user_msg]).content


# ----------------- Streamlit UI -----------------

def main():
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

    # Load retriever + LLM with nice error handling
    try:
        retriever, llm = load_retriever_and_llm()
    except Exception as e:
        st.error(
            "SERAPHIN could not start because of a configuration problem.\n\n"
            f"Details: {e}"
        )
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input(
        "Ask SERAPHIN a question:",
        placeholder="Example: How do I apply for a Temporary Protective Order (TPO) in Fulton County?",
    )

    if st.button("Submit") and question.strip():
        with st.spinner("SERAPHIN is thinking..."):
            try:
                answer = answer_question(question.strip(), retriever, llm)
            except Exception:
                # Hide raw stack trace from users
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
        "Disclaimer: SERAPHIN gives general information only. It does not replace a lawyer, "
        "police, or certified domestic violence advocate. If you are in immediate danger, call 911."
    )


if __name__ == "__main__":
    main()
