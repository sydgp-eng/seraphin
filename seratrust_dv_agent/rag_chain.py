import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chains import RetrievalQA
from langchain_groq import ChatGroq

DB_DIR = "chroma_db"


def load_rag_chain():
    # 1. Embeddings (FREE MiniLM)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Load vector store
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 3. Free LLM via Groq (fast + free)
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 4. Create RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    return chain


# ---------------------------
# ⭐ INTERACTIVE CHAT SECTION
# ---------------------------
if __name__ == "__main__":
    chain = load_rag_chain()
    print("\n✅ Georgia DV RAG Agent Loaded Successfully!")
    print("Ask anything about Georgia Domestic Violence laws or protections.")
    print("Type 'exit' to quit.\n")

while True:
    question = input("\nEnter your question: ")
    if question.lower().strip() in ["exit", "quit", "q"]:
        break

    resp = chain({"query": question})
    print("\nAnswer:\n")
    print(resp["result"])
    print("\n----------------------")

