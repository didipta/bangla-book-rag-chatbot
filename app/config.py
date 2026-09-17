from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress verbose unauthenticated HuggingFace Hub warnings
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF Hub.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()

# If user configured HF_TOKEN in .env, export it
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HF_TOKEN"] = hf_token

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

# Selected book for this template. Verify the shared registration sheet first.
BOOK_TITLE = os.getenv("BOOK_TITLE", "দেবদাস")
AUTHOR = os.getenv("BOOK_AUTHOR", "শরৎচন্দ্র চট্টোপাধ্যায়")
EDITION_YEAR = os.getenv("BOOK_EDITION_YEAR", "১৯৪৭")
WIKISOURCE_ROOT = os.getenv(
    "WIKISOURCE_ROOT",
    "https://bn.wikisource.org/wiki/দেবদাস_(শরৎচন্দ্র_চট্টোপাধ্যায়)",
)
WIKISOURCE_API = "https://bn.wikisource.org/w/api.php"

# RAG configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "8"))
MAX_L2_DISTANCE = float(os.getenv("MAX_L2_DISTANCE", "1.30"))

# Current Groq production model; change only if your account exposes another model.
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NO_ANSWER = "দুঃখিত, নির্বাচিত বইটিতে এই প্রশ্নের উত্তর পাওয়া যায়নি।"

for folder in (RAW_DIR, INDEX_DIR):
    folder.mkdir(parents=True, exist_ok=True)
