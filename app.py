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
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing in .env")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE)
    return retriever, llm


def format_context(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content.strip() for d in docs)


def answer_question(question: str, retriever, llm) -> str:
    docs = retriever.invoke(question)
    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    if not docs:
        user_msg = HumanMessage(
            content=f"No context found. Provide general Georgia DV info for:\n\n{question}"
        )
        return llm.invoke([system_msg, user_msg]).content

    context = format_context(docs)
    user_prompt = f"""
Use this context to answer the question. If context is not enough, say so and
offer general guidance and safe next steps only.

Context:
{context}

Question:
{question}
"""
    user_msg = HumanMessage(content=user_prompt)
    return llm.invoke([system_msg, user_msg]).content


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

    retriever, llm = load_retriever_and_llm()

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input(
        "Ask SERAPHIN a question:",
        placeholder="Example: How do I apply for a Temporary Protective Order (TPO) in Fulton County?",
    )

    if st.button("Submit") and question.strip():
        with st.spinner("SERAPHIN is thinking..."):
            answer = answer_question(question.strip(), retriever, llm)
        st.session_state.history.append((question.strip(), answer))

    st.markdown("---")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Seraphin:** {a}")
        st.markdown("---")

    st.caption(
        "Disclaimer: SERAPHIN gives general information only. It does not replace a lawyer, "
        "police, or certified DV advocate. If you are in immediate danger, call 911."
    )


if __name__ == "__main__":
    main()
