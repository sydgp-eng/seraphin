# app.py – SERAPHIN (SERA DV Assistant UI)

import os
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma


DB_DIR = "chroma_db"
MODEL_NAME = "llama-3.1-8b-instant"   # Groq production model
# or, if you want a bigger model:
# MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1

SYSTEM_PROMPT = """
You are SERAPHIN, the SERA Trust Domestic Violence Information Assistant for Georgia.

Your role:
- Provide ONLY general information (not legal advice).
- Be calm, trauma-informed, supportive, and factual.
- Encourage users to contact licensed attorneys, law enforcement, or certified DV advocates for case-specific advice.
- If user appears to be in immediate danger, advise them to contact 911 or local emergency services.

You must NOT:
- Give legal advice.
- Tell users what decision to make.
- Draft legal documents or filings.
""".strip()


@st.cache_resource
def load_retriever_and_llm():
    """Load embeddings, Chroma retriever, and Groq LLM (cached across sessions)."""
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GROQ_API_KEY missing from environment variables")

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

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )

    return retriever, llm


def format_context(docs: List[Document]) -> str:
    """Join retrieved document snippets into a single context string."""
    return "\n\n".join(d.page_content.strip() for d in docs if d.page_content.strip())


def answer_question(question: str, retriever, llm) -> str:
    """RAG-style answer: retrieve context and call the LLM."""
    docs = retriever.invoke(question)
    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    # No context – fall back to generic Georgia DV information
    if not docs:
        user_msg = HumanMessage(
            content=f"No context found. Provide only general Georgia domestic violence information for:\n\n{question}"
        )
        return llm.invoke([system_msg, user_msg]).content

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
        /* Overall page background */
        .stApp {
            background: #f7f3fb;
        }

        /* Chat bubbles */
        .user-bubble {
            background-color: #e7e3ff;
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            margin-bottom: 0.25rem;
        }

        .bot-bubble {
            background-color: #ffffff;
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            border: 1px solid #e4d9ff;
            margin-bottom: 0.75rem;
        }

        .role-label {
            font-size: 0.8rem;
            font-weight: 600;
            opacity: 0.7;
            margin-bottom: 0.15rem;
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

    if "history" not in st.session_state:
        # Each item: {"role": "user"/"assistant", "content": str}
        st.session_state.history = []

    # Display chat history
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
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": question})

        try:
            with st.spinner("SERAPHIN is thinking..."):
                answer = answer_question(question, retriever, llm)
            st.session_state.history.append({"role": "assistant", "content": answer})
            st.rerun()
        except Exception as e:
            st.error("SERAPHIN ran into a technical problem. Please try again in a moment.")
            # You can comment this out in production if you prefer not to expose stack traces
            st.exception(e)

    st.markdown("---")
    st.caption(
        "Disclaimer: SERAPHIN provides general information only and does **not** create an "
        "attorney–client relationship. For legal advice, please contact a licensed Georgia attorney. "
        "If you are in immediate danger, call 911."
    )


if __name__ == "__main__":
    main()
