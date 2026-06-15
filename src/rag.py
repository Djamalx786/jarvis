import os
from datetime import datetime

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "memory/chroma"
DATA_PATH = "data/notes/"
MEMORY_PATH = "data/notes/memory.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_db = None


def build_db():
    global _db
    print("Baue ChromaDB auf...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    db = get_db()
    existing_ids = db.get()["ids"]
    if existing_ids:
        db.delete(ids=existing_ids)
    db.add_documents(chunks)
    print(f"{len(chunks)} Chunks gespeichert!")
    _db = db
    return db


def get_db():
    global _db
    if _db is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return _db


def search(query: str, k: int = 3) -> str:
    """Search the RAG knowledge base. Returns an empty-safe string, never raises."""
    try:
        db = get_db()
        results = db.similarity_search(query, k=k)
    except Exception as e:
        print(f"RAG search failed: {e}")
        return ""

    if not results:
        return ""

    return "\n\n".join(r.page_content for r in results)


def add_to_memory(entry: str) -> None:
    """Append a timestamped entry to the memory notes and rebuild the index."""
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "a") as f:
        f.write(f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{entry}\n")
    build_db()


def get_doc_count() -> int:
    """Return the number of chunks stored in the RAG knowledge base. Returns 0 on error."""
    try:
        db = get_db()
        return db._collection.count()
    except Exception as e:
        print(f"RAG doc count failed: {e}")
        return 0
