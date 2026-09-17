from __future__ import annotations

import re
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    BOOK_TITLE,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    INDEX_DIR,
    MAX_L2_DISTANCE,
    TOP_K,
)


# ============================================================
# EMBEDDINGS
# ============================================================

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


# ============================================================
# VECTOR STORE
# ============================================================

@lru_cache(maxsize=1)
def get_vectorstore():
    return FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# ============================================================
# LLM
# ============================================================

@lru_cache(maxsize=1)
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
    )


# ============================================================
# TEXT NORMALIZATION & STEMMING
# ============================================================

BENGALI_SUFFIXES = [
    "ের", "র", "দের", "কে", "তে", "ে", "বাবু", "মশায়", "মহাশয়", "সাহেব"
]


def bengali_stem(word: str) -> str:
    """
    Strip common Bengali noun and honorific inflections.
    """
    for suffix in sorted(BENGALI_SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[:-len(suffix)]
    return word


def normalize_bengali(text: str) -> str:
    """
    Normalize Bengali text for lexical matching.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    # Bengali and standard punctuation
    for p in ["।", ",", ";", ":", "?", "!", "-", "—", '"', "'", "(", ")", "[", "]"]:
        text = text.replace(p, " ")

    return text.strip()


# ============================================================
# STOPWORDS
# ============================================================

BENGALI_STOPWORDS = {
    "কি",
    "কী",
    "কে",
    "কেন",
    "কোথায়",
    "কোথায়",
    "কখন",
    "কিভাবে",
    "কীভাবে",
    "হয়",
    "হয়",
    "ছিল",
    "আছে",
    "হল",
    "হলো",
    "এর",
    "এবং",
    "ও",
    "তো",
    "সে",
    "তিনি",
    "তার",
    "তাঁর",
    "একটি",
    "একজন",
}


# ============================================================
# TOKENIZATION
# ============================================================

@lru_cache(maxsize=5000)
def _tokenize_word(word: str) -> tuple[str, ...]:
    if word in BENGALI_STOPWORDS or len(word) <= 1:
        return ()
    res = [word]
    stem = bengali_stem(word)
    if stem != word and len(stem) > 1 and stem not in BENGALI_STOPWORDS:
        res.append(stem)
    return tuple(res)


def tokenize(text: str) -> list[str]:
    """
    Tokenize Bengali text and generate root stems for indexing/matching.
    """
    text = normalize_bengali(text)

    words = re.findall(
        r"[\u0980-\u09FF]+",
        text,
    )

    tokens = []
    for word in words:
        for t in _tokenize_word(word):
            tokens.append(t)

    return tokens


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(question: str) -> list[str]:
    """
    Generate Bengali relationship-focused query variants dynamically.
    """
    q = normalize_bengali(question)
    queries = [q]

    tokens = [w for w in q.split() if w not in BENGALI_STOPWORDS and len(w) > 1]

    rel_keywords = {
        "বাবা", "বাবার", "পিতা", "পিতার", "জনক",
        "মা", "মাতা", "মায়ের", "মাতার", "জননী",
        "ভাই", "ভ্রাতা", "দাদা", "বোন", "দিদি",
        "স্বামী", "স্ত্রী", "ছেলে", "পুত্র", "কন্যা", "মেয়ে", "নাম", "নামের"
    }
    subjects = [w for w in tokens if w not in rel_keywords]
    subject = subjects[0] if subjects else BOOK_TITLE
    base_subject = bengali_stem(subject)

    if any(k in q for k in ["বাবা", "বাবার", "পিতা", "পিতার", "জনক"]):
        queries.extend([
            f"{base_subject} পিতা",
            f"{base_subject} বাবা",
            f"{base_subject} জনক",
            f"{base_subject} পুত্র",
            f"{base_subject}ের পিতা",
            f"{base_subject}ের বাবা",
            f"{base_subject}ের বাবার নাম",
            f"{base_subject}ের পিতার নাম",
            f"জমিদার {base_subject}",
        ])

    if any(k in q for k in ["মা", "মায়ের", "মায়ের", "মাতা", "মাতার", "জননী"]):
        queries.extend([
            f"{base_subject} মা",
            f"{base_subject} মাতা",
            f"{base_subject} জননী",
            f"{base_subject}ের মা",
            f"{base_subject}ের মাতা",
            f"{base_subject}ের জননী",
        ])

    if "ভাই" in q or "দাদা" in q:
        queries.extend([
            f"{base_subject} ভাই",
            f"{base_subject} দাদা",
            f"{base_subject} অগ্রজ",
            f"{base_subject} অনুজ",
            f"{base_subject}ের ভাই",
        ])

    if "ছেলে" in q or "পুত্র" in q:
        queries.extend([
            f"{base_subject} ছেলে",
            f"{base_subject} পুত্র",
            f"{base_subject} সন্তান",
        ])

    if any(k in q for k in ["স্ত্রী", "বউ", "পত্নী", "বিয়ে", "বিবাহ", "স্বামী"]):
        queries.extend([
            f"{base_subject} বিয়ে",
            f"{base_subject} বিবাহ",
            f"{base_subject} স্বামী",
            f"{base_subject} স্ত্রী",
            f"{base_subject}ের বিয়ে",
            f"{base_subject}ের বিবাহ",
            f"{base_subject}ের স্বামী",
        ])

    result = []
    for query in queries:
        clean_q = normalize_bengali(query)
        if clean_q and clean_q not in result:
            result.append(clean_q)

    return result


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieve(
    question: str,
    k: int = TOP_K,
) -> list[tuple[Document, float]]:
    """
    Combine:
    - FAISS semantic similarity
    - Bengali query expansion
    - Lexical matching with stemming
    - Relationship entity prioritization

    Returns:
        [(Document, combined_score), ...]
    """
    vectorstore = get_vectorstore()
    expanded_queries = expand_query(question)

    candidates: dict[str, dict] = {}

    # Semantic retrieval
    semantic_k = max(k * 4, 30)

    for query in expanded_queries:
        try:
            results = vectorstore.similarity_search_with_score(
                query,
                k=semantic_k,
            )
        except Exception:
            continue

        for document, distance in results:
            chunk_id = document.metadata.get(
                "chunk_id",
                f"{document.metadata.get('chapter_name')}::{document.page_content[:80]}",
            )

            semantic_score = 1.0 / (1.0 + float(distance))

            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "document": document,
                    "semantic_score": 0.0,
                    "lexical_score": 0.0,
                }

            candidates[chunk_id]["semantic_score"] = max(
                candidates[chunk_id]["semantic_score"],
                semantic_score,
            )

    # Lexical retrieval
    all_documents = list(
        vectorstore.docstore._dict.values()
    )

    for document in all_documents:
        best_lexical = 0.0
        doc_tokens = set(tokenize(document.page_content))

        for query in expanded_queries:
            q_tokens = tokenize(query)
            if not q_tokens:
                continue
            matched = sum(1.0 for t in q_tokens if t in doc_tokens)
            score = matched / len(q_tokens)
            best_lexical = max(best_lexical, score)

        if best_lexical <= 0:
            continue

        chunk_id = document.metadata.get(
            "chunk_id",
            f"{document.metadata.get('chapter_name')}::{document.page_content[:80]}",
        )

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "document": document,
                "semantic_score": 0.0,
                "lexical_score": 0.0,
            }

        candidates[chunk_id]["lexical_score"] = max(
            candidates[chunk_id]["lexical_score"],
            best_lexical,
        )

    # Dynamic query subject and relationship keywords for any book
    q_norm = normalize_bengali(question)
    q_tokens = [w for w in q_norm.split() if w not in BENGALI_STOPWORDS and len(w) > 1]
    rel_keywords = {
        "বাবা", "বাবার", "পিতা", "পিতার", "জনক",
        "মা", "মাতা", "মায়ের", "মাতার", "জননী",
        "ভাই", "ভ্রাতা", "দাদা", "বোন", "দিদি",
        "স্বামী", "স্ত্রী", "ছেলে", "পুত্র", "কন্যা", "মেয়ে", "নাম", "নামের"
    }
    name_honorifics = {
        "বাবু", "মশায়", "মহাশয়", "সাহেব", "ঠাকুর", "চক্রবর্তী", "মুখুয্যে",
        "ভটচাজ", "জমিদার", "নামক", "নাম", "রায়", "ঘোষ", "মণ্ডল", "চৌধুরী"
    }

    subjects = [w for w in q_tokens if w not in rel_keywords]
    subject_stems = [bengali_stem(s) for s in subjects]
    if not subject_stems and BOOK_TITLE:
        subject_stems = [bengali_stem(BOOK_TITLE)]

    # Combine scores with contextual keyword bonus
    ranked = []

    for item in candidates.values():
        semantic = item["semantic_score"]
        lexical = item["lexical_score"]
        doc_text = item["document"].page_content

        bonus = 0.0
        # Dynamic bonus: boost when chunk mentions queried subject and relationship
        if any(s in doc_text for s in subject_stems):
            if any(r in doc_text for r in rel_keywords):
                bonus += 0.05
                if any(h in doc_text for h in name_honorifics):
                    bonus += 0.05

        combined = (
            semantic * 0.50
            + lexical * 0.50
            + bonus
        )

        ranked.append(
            (
                item["document"],
                combined,
                semantic,
                lexical,
            )
        )

    ranked.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    final_results = []
    for document, combined, semantic, lexical in ranked:
        if semantic == 0 and lexical == 0:
            continue
        if lexical < 0.20 and semantic < 0.35:
            continue

        final_results.append((document, combined))
        if len(final_results) >= k:
            break

    return final_results


# ============================================================
# STRICT SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """তুমি বাংলা সাহিত্য ও বই-ভিত্তিক একজন নির্ভরযোগ্য তথ্যানুগ প্রশ্নোত্তর সহায়ক।
তোমার উত্তরের ভিত্তি হলো নিচে প্রদত্ত CONTEXT।

নির্দেশনা:
1. প্রদত্ত CONTEXT-এর প্রত্যক্ষ তথ্য, চরিত্রের সংলাপ, ঘটনা ও পারিবারিক সম্পর্কের ভিত্তিতে (যেমন: পিতা/মাতা/সন্তানের পরিচয়, সৎকার বা সম্বোধন) সংক্ষেপে ও স্পষ্টভাবে প্রশ্নের উত্তর দেবে।
2. পাঠ্যাংশে চরিত্রের যে নাম বা পরিচয় বর্ণিত রয়েছে (যেমন: জমিদার নারায়ণ মুখুয্যে / নারায়ণবাবু), তা যথাযথভাবে জানাবে।
3. বিমূর্ত বা বিশ্লেষণধর্মী প্রশ্ন (যেমন: জীবনের মূল দ্বন্দ্ব, উদ্দেশ্য, শিক্ষা) অথবা CONTEXT-এ চরিত্রের কোনো তথ্য বা উল্লেখ না থাকলে সরাসরি বলবে:
"দুঃখিত, নির্বাচিত বইটিতে এই প্রশ্নের উত্তর পাওয়া যায়নি।"
"""


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    documents: list[tuple[Document, float]],
) -> str:
    context_parts = []

    for index, (document, score) in enumerate(
        documents,
        start=1,
    ):
        chapter = document.metadata.get(
            "chapter_name",
            "অজানা অধ্যায়",
        )
        section = document.metadata.get(
            "section_name",
            "",
        )
        source_url = document.metadata.get(
            "source_url",
            "",
        )

        context_parts.append(
            f"""
[CONTEXT {index}]
অধ্যায়: {chapter}
অংশ: {section}
উৎস URL: {source_url}

{document.page_content}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# ANSWER
# ============================================================

def answer_question(
    question: str,
) -> dict:
    retrieved = hybrid_retrieve(
        question,
        k=TOP_K,
    )

    if not retrieved:
        return {
            "answer": (
                "দুঃখিত, নির্বাচিত বইটিতে "
                "এই প্রশ্নের উত্তর পাওয়া যায়নি।"
            ),
            "sources": [],
            "documents": [],
        }

    context = build_context(retrieved)
    llm = get_llm()

    user_prompt = f"""CONTEXT:

{context}

==================================================

USER QUESTION:

{question}

==================================================

নির্দেশনা:
প্রদত্ত CONTEXT-এর তথ্যের ভিত্তিতে সংক্ষেপে ও সরাসরি উত্তর দাও।
বইয়ের পাঠ্যাংশে চরিত্রের নাম বা পরিচয় উল্লেখিত থাকলে (যেমন: পিতা/মাতা/সন্তান/ভাই/স্ত্রী) তা যথাযথভাবে উল্লেখ করবে।
যদি CONTEXT-এ প্রশ্নের উত্তর না থাকে, তবে কেবল ঠিক এই বাক্যটি উত্তর দাও:
"দুঃখিত, নির্বাচিত বইটিতে এই প্রশ্নের উত্তর পাওয়া যায়নি।"
"""

    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )

    answer = response.content.strip()

    # Sources
    sources = []
    if "দুঃখিত" not in answer:
        for document, score in retrieved:
            chapter = document.metadata.get("chapter_name")
            source_url = document.metadata.get("source_url")

            if not chapter:
                continue

            source = {
                "chapter_name": chapter,
                "section_name": document.metadata.get("section_name"),
                "source_url": source_url,
            }

            if source not in sources:
                sources.append(source)

    return {
        "answer": answer,
        "sources": sources,
        "documents": retrieved,
    }