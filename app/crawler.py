from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import quote, unquote

import requests
from bs4 import BeautifulSoup

from app.config import (
    AUTHOR,
    BOOK_TITLE,
    EDITION_YEAR,
    RAW_DIR,
    WIKISOURCE_API,
    WIKISOURCE_ROOT,
)

HEADERS = {
    "User-Agent": "BanglaBookRAGAssignment/1.0 (educational project)"
}


# ============================================================
# HTTP HELPERS
# ============================================================

def get_response(url: str, params: dict | None = None, max_retries: int = 5) -> requests.Response:
    """Make a GET request with automatic retry and exponential backoff on 429/errors."""

    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=30,
            )
            if response.status_code == 429:
                print(f"        [Rate limited (429)] Waiting {delay:.1f}s before retry (attempt {attempt}/{max_retries})...")
                time.sleep(delay)
                delay *= 2.0
                continue
            response.raise_for_status()
            return response
        except (requests.RequestException, requests.HTTPError) as exc:
            if attempt == max_retries:
                raise
            print(f"        [Network issue: {exc}] Retrying in {delay:.1f}s (attempt {attempt}/{max_retries})...")
            time.sleep(delay)
            delay *= 2.0

    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts.")


def api_get(params: dict) -> dict:
    """Call the Wikisource MediaWiki API."""

    response = get_response(WIKISOURCE_API, params)

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Wikisource API returned an invalid JSON response."
        ) from exc

    if "error" in data:
        raise RuntimeError(
            f"Wikisource API error: {data['error']}"
        )

    return data


# ============================================================
# URL / TITLE HELPERS
# ============================================================

def root_to_prefix(root_url: str) -> str:
    """
    Convert:

    https://bn.wikisource.org/wiki/মেজদিদি_(শরৎচন্দ্র_চট্টোপাধ্যায়,_১৯৬৩)/মেজদিদি

    into:

    মেজদিদি_(শরৎচন্দ্র_চট্টোপাধ্যায়,_১৯৬৩)/মেজদিদি
    """

    if "/wiki/" not in root_url:
        raise ValueError(
            f"Invalid Wikisource root URL: {root_url}"
        )

    title = root_url.split("/wiki/", 1)[1]

    # URL decode Bengali characters if necessary.
    title = unquote(title)

    return title.rstrip("/")


def make_source_url(title: str) -> str:
    """Convert a MediaWiki page title to a Wikisource URL."""

    return (
        "https://bn.wikisource.org/wiki/"
        + quote(title, safe="()/_-—")
    )


# ============================================================
# ROOT PAGE
# ============================================================

def get_root_page_html() -> str:
    """
    Download the selected Wikisource root page.

    We intentionally fetch the actual HTML page instead of depending
    only on the MediaWiki 'links' API.
    """

    root_title = root_to_prefix(WIKISOURCE_ROOT)

    url = make_source_url(root_title)

    print(f"Fetching root page:")
    print(url)

    response = get_response(url)

    return response.text


# ============================================================
# CHAPTER DISCOVERY
# ============================================================

def list_subpages() -> list[str]:
    """
    Find direct chapter pages under the selected Wikisource root.

    Example:

    Root:
        .../মেজদিদি

    Expected chapters:

        .../মেজদিদি/এক
        .../মেজদিদি/দুই
        .../মেজদিদি/তিন
        ...

    Nested pages such as:

        .../আঁধারে আলো/এক

    are ignored.
    """

    root_title = root_to_prefix(WIKISOURCE_ROOT)
    prefix = root_title + "/"

    print()
    print("=" * 60)
    print("Discovering chapter pages")
    print("=" * 60)
    print(f"Root title: {root_title}")
    print(f"Prefix:     {prefix}")
    print()

    html = get_root_page_html()

    soup = BeautifulSoup(html, "html.parser")

    found_titles: list[str] = []

    # --------------------------------------------------------
    # Method 1:
    # Read all links from the root page HTML.
    # --------------------------------------------------------

    for link in soup.find_all("a", href=True):

        href = link.get("href", "").strip()

        if not href.startswith("/wiki/"):
            continue

        # Remove /wiki/
        href_title = href[len("/wiki/"):]

        # URL decode
        href_title = unquote(href_title)

        # Ignore query strings / fragments
        href_title = href_title.split("#", 1)[0]
        href_title = href_title.split("?", 1)[0]

        # Direct child only
        if not href_title.startswith(prefix):
            continue

        remaining = href_title[len(prefix):]

        # Ignore nested pages.
        #
        # Example:
        # মেজদিদি/আঁধারে আলো/এক
        #
        # This contains another "/".
        if "/" in remaining:
            continue

        if not remaining:
            continue

        found_titles.append(href_title)

    # Remove duplicates
    found_titles = sorted(set(found_titles))

    print(f"Chapter links found from HTML: {len(found_titles)}")

    for title in found_titles:
        print(f"  + {title}")

    # --------------------------------------------------------
    # Method 2:
    # If HTML discovery fails, use MediaWiki API.
    # --------------------------------------------------------

    if not found_titles:

        print()
        print("HTML discovery found no chapters.")
        print("Trying MediaWiki API fallback...")

        params = {
            "action": "query",
            "format": "json",
            "titles": root_title,
            "prop": "links",
            "namespace": 0,
            "limit": "max",
        }

        data = api_get(params)

        pages = data.get(
            "query",
            {}
        ).get(
            "pages",
            {}
        )

        for page in pages.values():

            for link in page.get("links", []):

                title = link.get("title", "")

                if not title.startswith(prefix):
                    continue

                remaining = title[len(prefix):]

                if "/" in remaining:
                    continue

                if remaining:
                    found_titles.append(title)

        found_titles = sorted(set(found_titles))

        print(
            f"Chapter links found from API: {len(found_titles)}"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not found_titles:
        print()
        print("ERROR: No chapter pages were discovered.")
        print()
        print("Root URL:")
        print(WIKISOURCE_ROOT)
        print()

        raise RuntimeError(
            "No chapter pages were found. "
            "Check WIKISOURCE_ROOT and Wikisource page structure."
        )

    print()
    print(f"Total chapters found: {len(found_titles)}")

    return found_titles


# ============================================================
# PAGE TEXT
# ============================================================

def page_to_text(title: str) -> str:
    """
    Download a Wikisource page using the MediaWiki parse API
    and convert its HTML to clean text.
    """

    print(f"Fetching chapter: {title}")

    data = api_get(
        {
            "action": "parse",
            "format": "json",
            "page": title,
            "prop": "text",
            "formatversion": 2,
        }
    )

    if "parse" not in data:
        raise RuntimeError(
            f"Could not parse Wikisource page: {title}"
        )

    html = data["parse"]["text"]

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # --------------------------------------------------------
    # Remove UI / navigation elements
    # --------------------------------------------------------

    selectors_to_remove = (
        "table",
        ".mw-editsection",
        ".noprint",
        ".ws-noexport",
        ".ws-header",
        ".ws-footer",
        ".metadata",
        ".reference",
        ".references",
        "sup.reference",
        "style",
        "script",
        "noscript",
        "nav",
    )

    for selector in selectors_to_remove:

        for node in soup.select(selector):
            node.decompose()

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    text = soup.get_text(
        "\n",
        strip=True,
    )

    # Remove invisible Unicode characters
    text = re.sub(
        r"[\u200b\u200c\u200d\ufeff]",
        "",
        text,
    )

    # Normalize spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Normalize excessive newlines
    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean extracted Wikisource text while preserving
    Bengali paragraphs.
    """

    lines: list[str] = []

    ignored_exact_lines = {
        "পড়ুন",
        "উৎস",
        "আলোচনা",
        "সরঞ্জাম",
        "সম্পাদনা",
        "বিষয়শ্রেণী",
    }

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line in ignored_exact_lines:
            continue

        # Remove excessive spaces
        line = re.sub(
            r"\s+",
            " ",
            line,
        )

        lines.append(line)

    cleaned = "\n".join(lines)

    # Remove excessive blank lines
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


# ============================================================
# SORT CHAPTERS
# ============================================================

BENGALI_NUMBERS = {
    "এক": 1,
    "দুই": 2,
    "তিন": 3,
    "চার": 4,
    "পাঁচ": 5,
    "ছয়": 6,
    "ছয়": 6,
    "সাত": 7,
    "আট": 8,
    "নয়": 9,
    "নয়": 9,
    "দশ": 10,
    "এগারো": 11,
    "বারো": 12,
    "তেরো": 13,
    "চৌদ্দ": 14,
    "পনেরো": 15,
    "ষোলো": 16,
    "সতেরো": 17,
    "আঠারো": 18,
    "উনিশ": 19,
    "বিশ": 20,
    "প্রথম": 1,
    "প্রথম_পরিচ্ছেদ": 1,
    "দ্বিতীয়": 2,
    "দ্বিতীয়_পরিচ্ছেদ": 2,
    "তৃতীয়": 3,
    "তৃতীয়_পরিচ্ছেদ": 3,
    "চতুর্থ": 4,
    "চতুর্থ_পরিচ্ছেদ": 4,
    "পঞ্চম": 5,
    "পঞ্চম_পরিচ্ছেদ": 5,
    "ষষ্ঠ": 6,
    "ষষ্ঠ_পরিচ্ছেদ": 6,
    "সপ্তম": 7,
    "সপ্তম_পরিচ্ছেদ": 7,
    "অষ্টম": 8,
    "অষ্টম_পরিচ্ছেদ": 8,
    "নবম": 9,
    "নবম_পরিচ্ছেদ": 9,
    "দশম": 10,
    "দশম_পরিচ্ছেদ": 10,
    "একাদশ": 11,
    "একাদশ_পরিচ্ছেদ": 11,
    "দ্বাদশ": 12,
    "দ্বাদশ_পরিচ্ছেদ": 12,
    "ত্রয়োদশ": 13,
    "ত্রয়োদশ_পরিচ্ছেদ": 13,
    "চতুর্দশ": 14,
    "চতুর্দশ_পরিচ্ছেদ": 14,
    "পঞ্চদশ": 15,
    "পঞ্চদশ_পরিচ্ছেদ": 15,
    "ষোড়শ": 16,
    "ষোড়শ_পরিচ্ছেদ": 16,
    "সপ্তদশ": 17,
    "সপ্তদশ_পরিচ্ছেদ": 17,
    "অষ্টাদশ": 18,
    "অষ্টাদশ_পরিচ্ছেদ": 18,
    "ঊনবিংশ": 19,
    "ঊনবিংশ_পরিচ্ছেদ": 19,
    "বিংশ": 20,
    "বিংশ_পরিচ্ছেদ": 20,
}


def chapter_sort_key(title: str):
    """
    Sort Bengali chapter names naturally.

    Example:

    এক
    দুই
    তিন
    চার
    ...
    """

    short_name = title.rsplit("/", 1)[-1]

    number = BENGALI_NUMBERS.get(
        short_name,
        9999,
    )

    return (
        number,
        short_name,
    )


# ============================================================
# CRAWL BOOK
# ============================================================

def crawl_book() -> Path:
    """
    Crawl the complete selected book and save chapter-level
    JSON records to data/raw/book_pages.json.
    """

    print()
    print("=" * 60)
    print("BANGLA WIKISOURCE BOOK CRAWLER")
    print("=" * 60)
    print(f"Book:       {BOOK_TITLE}")
    print(f"Author:     {AUTHOR}")
    print(f"Edition:    {EDITION_YEAR}")
    print(f"Root URL:   {WIKISOURCE_ROOT}")
    print("=" * 60)

    # --------------------------------------------------------
    # Discover chapters
    # --------------------------------------------------------

    pages = list_subpages()

    pages = sorted(
        pages,
        key=chapter_sort_key,
    )

    if not pages:
        raise RuntimeError(
            "No chapter pages were found."
        )

    print()
    print("=" * 60)
    print(f"Found {len(pages)} chapter pages")
    print("=" * 60)

    for index, title in enumerate(
        pages,
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"{title.rsplit('/', 1)[-1]}"
        )

    # --------------------------------------------------------
    # Crawl chapters
    # --------------------------------------------------------

    records: list[dict] = []

    print()
    print("=" * 60)
    print("Downloading chapters")
    print("=" * 60)

    for index, title in enumerate(
        pages,
        start=1,
    ):

        short_name = title.rsplit(
            "/",
            1,
        )[-1]

        try:

            raw_text = page_to_text(title)

            text = clean_text(raw_text)

            if len(text) < 50:

                print(
                    f"[{index}/{len(pages)}] "
                    f"SKIPPED: {short_name} "
                    f"(too little text)"
                )

                continue

            record = {
                "book_title": BOOK_TITLE,
                "author": AUTHOR,
                "edition_year": EDITION_YEAR,
                "chapter_name": (
                    f"{BOOK_TITLE} — {short_name}"
                ),
                "section_name": short_name,
                "source_url": make_source_url(title),
                "page_title": title,
                "text": text,
            }

            records.append(record)

            print(
                f"[{index}/{len(pages)}] "
                f"OK: {short_name} "
                f"({len(text):,} chars)"
            )

        except Exception as exc:

            print(
                f"[{index}/{len(pages)}] "
                f"ERROR: {short_name}"
            )

            print(
                f"        {exc}"
            )

        # Be polite to Wikisource
        time.sleep(0.5)

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not records:
        raise RuntimeError(
            "No usable chapter text was downloaded."
        )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = RAW_DIR / "book_pages.json"

    output.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_chars = sum(
        len(record["text"])
        for record in records
    )

    print()
    print("=" * 60)
    print("CRAWLING COMPLETE")
    print("=" * 60)
    print(f"Book:             {BOOK_TITLE}")
    print(f"Chapters saved:   {len(records)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Output:           {output}")
    print("=" * 60)

    return output


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    crawl_book()