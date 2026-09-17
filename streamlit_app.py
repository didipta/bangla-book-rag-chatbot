from __future__ import annotations

import streamlit as st

from app.config import AUTHOR, BOOK_TITLE, EDITION_YEAR, WIKISOURCE_ROOT
from app.rag import answer_question

st.set_page_config(
    page_title=f"{BOOK_TITLE} — Knowledge Base Chatbot",
    page_icon="📚",
    layout="centered",
)

st.title(f"📚 {BOOK_TITLE} — Knowledge Base Chatbot")
st.caption(f"লেখক: {AUTHOR} · সংস্করণ: {EDITION_YEAR}")
st.info(
    "এই chatbot শুধুমাত্র নির্বাচিত বইয়ের retrieved context ব্যবহার করে উত্তর দেয়। "
    "বইয়ে তথ্য না থাকলে সে তা স্পষ্টভাবে জানায়।"
)

with st.sidebar:
    st.subheader("Book")
    st.write(BOOK_TITLE)
    st.write(AUTHOR)
    st.markdown(f"[Bengali Wikisource]({WIKISOURCE_ROOT})")
    st.subheader("Pipeline")
    st.write("Wikisource → Crawl → Clean → Chunk → Embeddings → FAISS → Retrieval → LLM → Citation")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("উৎস")
            for source in message["sources"]:
                chapter = source.get("chapter_name") or source.get("chapter") or "অজানা অধ্যায়"
                section = source.get("section_name") or source.get("section") or ""
                url = source.get("source_url") or source.get("url") or ""
                st.markdown(
                    f"- **{chapter}**" + (f" / {section}" if section else "") + (f" — [Wikisource]({url})" if url else "")
                )

question = st.chat_input("বইটি সম্পর্কে প্রশ্ন লিখুন…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("বইয়ের প্রাসঙ্গিক অংশ খুঁজছি…"):
            try:
                result = answer_question(question)
            except FileNotFoundError:
                st.error("FAISS index পাওয়া যায়নি। আগে `python -m app.crawler` এবং `python -m app.ingest` চালান।")
                st.stop()
            except Exception as exc:
                st.error(f"RAG pipeline error: {exc}")
                st.stop()

        answer_text = result["answer"] if isinstance(result, dict) else getattr(result, "answer", str(result))
        sources_list = result.get("sources", []) if isinstance(result, dict) else getattr(result, "sources", [])

        st.markdown(answer_text)
        if sources_list:
            st.caption("উৎস")
            for source in sources_list:
                chapter = source.get("chapter_name") or source.get("chapter") or "অজানা অধ্যায়"
                section = source.get("section_name") or source.get("section") or ""
                url = source.get("source_url") or source.get("url") or ""
                st.markdown(
                    f"- **{chapter}**" + (f" / {section}" if section else "") + (f" — [Wikisource]({url})" if url else "")
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": sources_list,
        }
    )

