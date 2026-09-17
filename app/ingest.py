from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_FILE,
    EMBEDDING_MODEL,
    INDEX_DIR,
    RAW_DIR,
)


def load_pages() -> list[dict]:
    """Load crawled book pages from JSON."""

    path = RAW_DIR / "book_pages.json"

    if not path.exists():
        raise FileNotFoundError(
            "Raw book data not found.\n"
            "Run: python -m app.crawler"
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("book_pages.json must contain a list of pages.")

    if not data:
        raise ValueError("book_pages.json is empty.")

    return data


def build_documents(pages: list[dict]) -> list[Document]:
    """Convert crawled pages into LangChain Documents."""

    documents: list[Document] = []

    for page_number, page in enumerate(pages, start=1):

        text = str(page.get("text", "")).strip()

        if not text:
            print(
                f"[WARNING] Empty page skipped: "
                f"{page.get('chapter_name', 'Unknown')}"
            )
            continue

        metadata = {
            "book_name": page.get("book_title", ""),
            "author": page.get("author", ""),
            "edition_year": page.get("edition_year", ""),
            "chapter_name": page.get("chapter_name", ""),
            "section_name": page.get("section_name", ""),
            "source_url": page.get("source_url", ""),
            "page_title": page.get("page_title", ""),
            "page_number": page_number,
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split book pages into retrieval-friendly chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "। ",
            "।",
            "\n",
            " ",
            "",
        ],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Add stable chunk metadata.
    chapter_counter: Counter[str] = Counter()

    for global_index, chunk in enumerate(chunks):

        chapter_name = chunk.metadata.get(
            "chapter_name",
            "unknown",
        )

        chapter_counter[chapter_name] += 1

        chunk.metadata["chunk_id"] = global_index

        chunk.metadata["chapter_chunk_index"] = chapter_counter[
            chapter_name
        ]

        chunk.metadata["chunk_length"] = len(
            chunk.page_content
        )

    return chunks


def print_dataset_summary(
    pages: list[dict],
    documents: list[Document],
    chunks: list[Document],
) -> None:
    """Print useful information before building FAISS."""

    chapters = []

    for page in pages:
        chapter = page.get("chapter_name", "")
        if chapter:
            chapters.append(chapter)

    unique_chapters = list(dict.fromkeys(chapters))

    print()
    print("=" * 70)
    print("BOOK DATASET SUMMARY")
    print("=" * 70)

    print(f"Pages loaded       : {len(pages)}")
    print(f"Documents created  : {len(documents)}")
    print(f"Chunks created     : {len(chunks)}")
    print(f"Chunk size         : {CHUNK_SIZE}")
    print(f"Chunk overlap      : {CHUNK_OVERLAP}")
    print(f"Embedding model    : {EMBEDDING_MODEL}")

    print()
    print(f"Chapters found     : {len(unique_chapters)}")

    for index, chapter in enumerate(unique_chapters, start=1):
        print(f"  {index:02d}. {chapter}")

    print("=" * 70)
    print()


def save_chunks(chunks: list[Document]) -> None:
    """Save generated chunks for debugging and inspection."""

    CHUNKS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CHUNKS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for chunk in chunks:
            record = {
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def clear_old_index() -> None:
    """Remove the previous FAISS index before rebuilding."""

    if INDEX_DIR.exists():

        print(
            f"[INFO] Removing old FAISS index: {INDEX_DIR}"
        )

        if INDEX_DIR.is_dir():
            shutil.rmtree(INDEX_DIR)
        else:
            INDEX_DIR.unlink()

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    """Create the same embedding configuration used during retrieval."""

    print()
    print("[INFO] Loading embedding model...")
    print(f"[INFO] Model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    print("[INFO] Embedding model loaded.")

    return embeddings


def build_index() -> None:
    """Build the complete FAISS vector database."""

    print()
    print("=" * 70)
    print("BUILDING BANGLA BOOK VECTOR INDEX")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load crawled pages
    # ---------------------------------------------------------

    print()
    print("[1/6] Loading crawled pages...")

    pages = load_pages()

    print(
        f"[OK] Loaded {len(pages)} pages."
    )

    # ---------------------------------------------------------
    # 2. Convert pages to Documents
    # ---------------------------------------------------------

    print()
    print("[2/6] Creating LangChain documents...")

    documents = build_documents(pages)

    if not documents:
        raise ValueError(
            "No valid documents were created from book_pages.json."
        )

    print(
        f"[OK] Created {len(documents)} documents."
    )

    # ---------------------------------------------------------
    # 3. Split documents
    # ---------------------------------------------------------

    print()
    print("[3/6] Splitting documents into chunks...")

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError(
            "No chunks were created."
        )

    print(
        f"[OK] Created {len(chunks)} chunks."
    )

    # ---------------------------------------------------------
    # Dataset summary
    # ---------------------------------------------------------

    print_dataset_summary(
        pages,
        documents,
        chunks,
    )

    # ---------------------------------------------------------
    # 4. Save chunks
    # ---------------------------------------------------------

    print()
    print("[4/6] Saving chunks...")

    save_chunks(chunks)

    print(
        f"[OK] Chunks saved to: {CHUNKS_FILE}"
    )

    # ---------------------------------------------------------
    # 5. Build embeddings
    # ---------------------------------------------------------

    print()
    print("[5/6] Creating FAISS vector database...")

    embeddings = build_embeddings()

    # Remove old index first.
    clear_old_index()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    # ---------------------------------------------------------
    # 6. Save FAISS
    # ---------------------------------------------------------

    vectorstore.save_local(
        str(INDEX_DIR)
    )

    print()
    print("[6/6] FAISS index saved.")
    print(
        f"[OK] Index directory: {INDEX_DIR}"
    )

    print()
    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)
    print()
    print(f"Pages      : {len(pages)}")
    print(f"Documents  : {len(documents)}")
    print(f"Chunks     : {len(chunks)}")
    print(f"Index      : {INDEX_DIR}")
    print()


if __name__ == "__main__":
    try:
        build_index()

    except Exception as exc:
        print()
        print("=" * 70)
        print("INDEX BUILD FAILED")
        print("=" * 70)
        print()
        print(f"Error: {exc}")
        print()

        raise