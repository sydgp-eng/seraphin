# rag_chain.py
"""
Simple CLI RAG agent for Domestic Violence (DV) information in Georgia.

Uses:
- Chroma vector store in `chroma_db/`
- HuggingFaceEmbeddings (free, local)
- Groq LLM (free API tier)

This tool provides general information, NOT legal advice.
"""

import os
import sys
from typing import List

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage


DB_DIR = "chroma_db"
MODEL_NAME = "llama-3.1-8b-instant"   # or any other Groq chat model
TEMPERATURE = 0.1

DEFAULT_SYSTEM_PROMPT = """
You are a calm, trauma-informed legal information assistant for victims of
domestic violence in Georgia, USA.

Your role:
- Explain Georgia domestic violence laws, procedures, and options in simple language.
- Provide clear, practical next steps (hotlines, shelters, legal help).
- Encourage the user to contact licensed attorneys, police, and certified advocates.
- If there is immediate danger, clearly tell them to call 911 or the local emergency number.

Limits:
- You are NOT a lawyer and do NOT provide legal advice, only general information.
- Do not guess case outcomes.
- When something depends on a judge, court, or attorney strategy, state that clearly.
- Base answers primarily on the provided document context. If context is insufficient,
  say so and give general guidance plus next steps.
""".strip()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_vectorstore() -> Chroma:
    """Load persisted Chroma DB with the SAME embeddings as ingest.py."""
    if not os.path.isdir(DB_DIR):
        raise FileNotFoundError(
            f"Vectorstore directory '{DB_DIR}' not found. "
            f"Run ingest.py first to create it."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    return db


def build_retriever(db: Chroma):
    """Create retriever from Chroma."""
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    return retriever


def build_llm() -> ChatGroq:
    """Build Groq chat model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )
    return llm


def format_context(docs: List[Document]) -> str:
    """Format retrieved documents into context text."""
    chunks = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", d.metadata.get("page_number", ""))
        header = f"[Document {i} | source: {source} | page: {page}]"
        chunks.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n".join(chunks)


def answer_question(question: str, retriever, llm) -> str:
    """Retrieve docs and ask the LLM to answer. Uses `.invoke()` APIs."""
    docs = retriever.invoke(question)

    system_msg = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)

    # If nothing found in the DB
    if not docs:
        user_msg = HumanMessage(
            content=(
                "There were no relevant documents found for this question.\n\n"
                f"User question: {question}\n\n"
                "Provide general, high-level information about domestic violence "
                "support and legal options in Georgia. Make it clear that this is "
                "general information, not legal advice."
            )
        )
        response = llm.invoke([system_msg, user_msg])
        return response.content

    context = format_context(docs)

    user_prompt = f"""
Use ONLY the context below where possible. If the context does not fully answer
the question, say that clearly and provide general guidance and next steps.

Context:
{context}

User question:
{question}

Respond with:
1. A brief summary (2–3 sentences).
2. Step-by-step guidance tailored to the situation.
3. A closing reminder that you are not a lawyer and the user should consult
   a Georgia attorney and a domestic-violence advocate for legal advice.
""".strip()

    user_msg = HumanMessage(content=user_prompt)
    response = llm.invoke([system_msg, user_msg])
    return response.content


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    load_dotenv()

    try:
        db = load_vectorstore()
    except Exception as e:
        print(f"[ERROR] Failed to load vector store: {e}")
        sys.exit(1)

    try:
        llm = build_llm()
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM: {e}")
        sys.exit(1)

    retriever = build_retriever(db)

    print("\n=== Georgia DV RAG Agent (Groq + HuggingFace) ===")
    print("Ask any question about domestic violence information in Georgia.")
    print("Type 'exit', 'quit', or 'q' to stop.\n")

    while True:
        try:
            question = input("Enter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting DV RAG Agent.")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("Exiting DV RAG Agent.")
            break

        if not question:
            continue

        try:
            answer = answer_question(question, retriever, llm)
            print("\nAnswer:\n")
            print(answer)
            print("\n" + "-" * 80 + "\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()
