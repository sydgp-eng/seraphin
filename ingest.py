# ingest.py
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_db"


def load_documents():
    docs = []
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".pdf"):
            path = os.path.join(DATA_DIR, f)
            loader = PyPDFLoader(path)
            pages = loader.load()
            for p in pages:
                p.metadata["source"] = f
            docs.extend(pages)
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return splitter.split_documents(docs)


def create_vectorstore(chunks):
    embeddings = FastEmbedEmbeddings()
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    db.persist()
    print("✅ Vector store created successfully in", DB_DIR)


if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"Data directory '{DATA_DIR}' not found.")
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No PDF files found in '{DATA_DIR}'.")
    chunks = split_documents(docs)
    create_vectorstore(chunks)
