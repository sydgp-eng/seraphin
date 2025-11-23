# app.py – SERAPHIN (SERA DV Assistant UI)

import os
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma


DB_DIR = "chroma_db"
MODEL_NAME = "llama-3.1-8b-instant"  # Groq model
TEMPERATURE = 0.1

SYSTEM_PROMPT = """
You are SERAPHIN, the SERA Trust Domestic Violence Information Assistant for Georgia.

Your role:
- Provide ONLY general information (not legal advice).
- Be calm, trauma-informed, supportive, and factual.
- Encourage users to contact licensed attorneys, law enforcement, or certified DV advocates for case-specific advice.
- If the user appears to be in immediate danger, advise them to contact 911 or local emergency services.

You must NOT:
- Give legal advice.
- Tell users what decision to make.
- Draft legal documents or filings.
""".strip()


@st.cache_resource
def load_retriever_and_llm():
    """Load (or fallback) retriever + Groq LLM, cached across sessions."""
    load_dotenv()

    # Make sure GROQ_API_KEY is available both locally and on Streamlit Cloud
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GROQ_API_KEY missing. Set it in .env (local) or Streamlit Secrets (cloud).")

    retriever: Optional[object] = None

    # Try to load sentence-transformer embeddings.
    # If this fails (e.g., ImportError on cloud), we gracefully fall back to LLM-only mode.
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )

        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
    except ImportError:
        # Embeddings not available in this environment – run without RAG.
        # We do NOT crash; SERAPHIN will still provide general DV information.
        retriever = None

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )

    return retriever, llm


def format_context(docs: List[Document]) -> str:
    """Join retrieved document snippets into a single context string."""
    return "\n\n".join(d.page_content.strip() for d in docs if d.page_content.strip())


def answer_question(question: str, retriever, llm) -> str:
    """Answer using RAG if retriever is available, otherwise LLM-only."""
    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    docs: List[Document] = []

    if retriever is not None:
        try:
            docs = retriever.invoke(question)
        except Exception:
            # If retrieval breaks for any reason, fall back to LLM-only.
            docs = []

    # No context available → general Georgia DV guidance
    if not docs:
        user_msg = HumanMessage(
            content=(
                "Provide general domestic violence information for Georgia to help answer "
                "the following question. Do NOT give legal advice, only high-level information.\n\n"
                f"Question: {question}"
            )
        )
        return llm.invoke([system_msg, user_msg]).content

    # We have context from Chroma → use RAG
    context = format_context(docs)
    prompt = f"""
You are SERAPHIN, the SERA Trust Domestic Violence Information Assistant for Georgia.

Use the context below to answer the user's question.
- If the context is weak, say that it may not fully match their situation and give only high-level guidance.
- Do NOT give legal advice.
- Do NOT make promises about outcomes.
- Suggest contacting a Georgia attorney or certified DV advocate for specific next steps.

Context:
{context}

User Question:
{question}
""".strip()

    user_msg = HumanMessage(content=prompt)
    response = llm.invoke([system_msg, user_msg])
    return response.content


def add_custom_styles():
    """Light styling for a calmer, more pleasant UI."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #121219;
        }

        .user-bubble {
            background-color: #2a2438;
            padding: 0.9rem 1.2rem;
            border-radius: 1rem;
            color: #f5f0ff;
            font-size: 1rem;
            line-height: 1.5;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        .bot-bubble {
            background-color: #1e1b29;
            padding: 0.9rem 1.2rem;
            border-radius: 1rem;
            border: 1px solid #4b3f8f;
            color: #f5f0ff;
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 0.8rem;
        }

        .role-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: #ba9cff;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="SERAPHIN – SERA DV Assistant",
        page_icon="💜",
        layout="centered",
    )

    add_custom_styles()

    # Sidebar info
    with st.sidebar:
        st.markdown("### 💜 SERAPHIN – Info")
        st.write(
            "SERAPHIN is an informational assistant created by **SERA Trust** "
            "to help explain domestic violence processes and resources in Georgia."
        )
        st.markdown(
            "**Important:** SERAPHIN does **not** replace:\n"
            "- A licensed attorney\n"
            "- Police or emergency services\n"
            "- A certified domestic violence advocate"
        )

    st.title("💜 SERAPHIN – SERA Domestic Violence Information Assistant")
    st.write(
        "SERAPHIN provides **general information only** about domestic violence laws and "
        "processes in **Georgia (USA)**.\n\n"
        "It cannot give legal advice or tell you what to do in your specific case."
    )

    st.info(
        "If you are in **immediate danger**, please call **911** or your local emergency number. "
        "For confidential support, you may also contact a certified domestic violence hotline or shelter."
    )

    retriever, llm = load_retriever_and_llm()

    # Show a small note if retriever is disabled in this environment
    if retriever is None:
        st.warning(
            "Search over SERA’s legal documents is temporarily unavailable in this environment. "
            "SERAPHIN will answer using general Georgia domestic violence information only."
        )

    if "history" not in st.session_state:
        st.session_state.history = []

    # Render chat history
    for message in st.session_state.history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.container():
                st.markdown('<div class="role-label">You</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown('<div class="role-label">SERAPHIN</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-bubble">{content}</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input(
        "Ask SERAPHIN a question (example: “How do I apply for a TPO in Fulton County?”)"
    )

    if user_input and user_input.strip():
        question = user_input.strip()
        st.session_state.history.append({"role": "user", "content": question})

        try:
            with st.spinner("SERAPHIN is thinking..."):
                answer = answer_question(question, retriever, llm)
            st.session_state.history.append({"role": "assistant", "content": answer})
            st.rerun()
        except Exception as e:
            st.error("SERAPHIN ran into a technical problem. Please try again in a moment.")
            st.exception(e)

    st.markdown("---")
    st.caption(
        "Disclaimer: SERAPHIN provides general information only and does **not** create an "
        "attorney–client relationship. For legal advice, please contact a licensed Georgia attorney. "
        "If you are in immediate danger, call 911."
    )


if __name__ == "__main__":
    main()
