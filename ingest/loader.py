"""
ingest/loader.py — The Document Ingestion Pipeline

Loads all logs and runbooks from data/, splits them into chunks,
embeds each chunk, and stores everything in ChromaDB.

Uses ChromaDB's native Python client directly — bypasses langchain_chroma
which has a known issue passing custom embedding functions on Windows.

Run with:
    python -m ingest.loader
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest.embeddings import LocalEmbeddings
from config import (
    DATA_DIR, CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    CHUNK_SIZE, CHUNK_OVERLAP
)


# ── Step 1: Load documents ─────────────────────────────────────────────────────

def load_documents():
    """
    Load all .txt log files and .md runbooks from data/.
    Returns a list of LangChain Document objects, each with
    .page_content (the text) and .metadata (source filename + doc_type).
    """
    documents = []

    logs_dir = DATA_DIR / "logs"
    for log_file in sorted(logs_dir.glob("*.txt")):
        try:
            loader = TextLoader(str(log_file), encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["doc_type"] = "log"
                doc.metadata["source"] = log_file.name
            documents.extend(loaded)
            print(f"  [log]     {log_file.name}  ({len(loaded[0].page_content):,} chars)")
        except Exception as e:
            print(f"  [ERROR] {log_file.name}: {e}")

    docs_dir = DATA_DIR / "docs"
    for doc_file in sorted(docs_dir.glob("*.md")):
        try:
            loader = TextLoader(str(doc_file), encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["doc_type"] = "runbook"
                doc.metadata["source"] = doc_file.name
            documents.extend(loaded)
            print(f"  [runbook] {doc_file.name}  ({len(loaded[0].page_content):,} chars)")
        except Exception as e:
            print(f"  [ERROR] {doc_file.name}: {e}")

    return documents


# ── Step 2: Split into chunks ──────────────────────────────────────────────────

def split_documents(documents):
    """
    Split documents into overlapping chunks.
    RecursiveCharacterTextSplitter tries paragraph → line → sentence
    boundaries before cutting mid-word.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ── Steps 3 & 4: Embed and store directly in ChromaDB ─────────────────────────

def create_vectorstore(chunks):
    """
    Embeds every chunk and stores it in ChromaDB using the native client.

    Why native client instead of langchain_chroma?
    langchain_chroma's from_documents() has a bug on Windows where it passes
    embeddings=[] to chromadb regardless of what the embedding function returns.
    Using chromadb.PersistentClient directly skips that wrapper entirely.

    The flow:
        chunk texts → LocalEmbeddings.embed_documents() → list of 384-dim vectors
        → chromadb collection.add(embeddings, documents, metadatas, ids)
    """
    embeddings_model = LocalEmbeddings()

    # Wipe any previous ingestion so we start clean
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    # Delete old collection if it exists (lets you re-run ingestion safely)
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("  Deleted existing collection — rebuilding fresh.")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for normalized vectors
    )

    # Process in batches of 50 — avoids memory spikes on large datasets
    batch_size = 50
    total = len(chunks)

    print(f"  Embedding and storing {total} chunks in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]

        texts     = [c.page_content for c in batch]
        metadatas = [c.metadata     for c in batch]
        ids       = [f"chunk_{i + j}" for j in range(len(batch))]

        # This is the call that does the real work
        vectors = embeddings_model.embed_documents(texts)

        # Sanity check — catch empty embeddings before ChromaDB does
        if not vectors or len(vectors) != len(texts):
            print(f"  [ERROR] Embedding batch {i}–{i+len(batch)} returned unexpected result.")
            print(f"          texts={len(texts)}, vectors={len(vectors) if vectors else 0}")
            continue

        collection.add(
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        done = min(i + batch_size, total)
        print(f"  Stored {done}/{total} chunks", end="\r")

    print()  # newline after the \r progress line
    return collection


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_ingestion():
    print("=" * 55)
    print("  OpsMind AI — Document Ingestion Pipeline")
    print("=" * 55)

    print("\nStep 1: Loading documents from data/")
    documents = load_documents()
    if not documents:
        print("\n[ERROR] No documents loaded.")
        return None
    print(f"  → {len(documents)} documents loaded")

    print("\nStep 2: Splitting into chunks")
    chunks = split_documents(documents)
    print(f"  → {len(chunks)} chunks created")
    print(f"    (each up to {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap)")

    if chunks:
        sample = chunks[len(chunks) // 2]
        print(f"\n  Sample chunk [{sample.metadata.get('source')}]:")
        print("  " + "─" * 45)
        for line in sample.page_content[:250].split("\n"):
            print(f"    {line}")
        print("  " + "─" * 45)

    print("\nSteps 3 & 4: Embedding and storing in ChromaDB")
    collection = create_vectorstore(chunks)

    count = collection.count()
    print(f"\n{'=' * 55}")
    print(f"  Ingestion complete! {count} chunks in ChromaDB.")
    print(f"  Location: {CHROMA_PERSIST_DIR}")
    print(f"\n  Next: python -m rag.retriever")
    print(f"{'=' * 55}\n")

    return collection


if __name__ == "__main__":
    run_ingestion()