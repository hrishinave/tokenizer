"""Download India's Wikipedia article in English, Hindi, Telugu, Kannada.

Writes clean plaintext to data/{en,hi,te,kn}.txt using the MediaWiki
`extracts` API (explaintext=1 -> no wiki markup / no reference numbers).
"""
import json
import os
import re
import sys

import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Article titles per language. English drives langlink resolution; the others
# are hardcoded fallbacks in case langlinks lookup fails.
LANGS = ["en", "hi", "te", "kn"]
FALLBACK_TITLES = {
    "en": "India",
    "hi": "भारत",
    "te": "భారతదేశం",
    "kn": "ಭಾರತ",
}

HEADERS = {"User-Agent": "bpe-tokenizer-exercise/1.0 (educational)"}


def resolve_titles():
    """Get the article title in each language via langlinks from en:India."""
    titles = dict(FALLBACK_TITLES)
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": "India",
                "prop": "langlinks",
                "lllang": "|".join(l for l in LANGS if l != "en"),
                "lllimit": "500",
            },
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        for page in pages.values():
            for ll in page.get("langlinks", []):
                if ll["lang"] in titles:
                    titles[ll["lang"]] = ll["*"]
    except Exception as e:  # noqa: BLE001
        print(f"  (langlinks lookup failed: {e}; using fallback titles)")
    return titles


def fetch_plaintext(lang, title):
    """Fetch the full plaintext extract of `title` from {lang}.wikipedia.org."""
    r = requests.get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "titles": title,
            "redirects": "1",
        },
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")


def clean(text):
    """Drop `== Section ==` heading-markup lines and blank-line runs."""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if re.fullmatch(r"=+\s*.*?\s*=+", s):  # heading markup line
            continue
        lines.append(line)
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    titles = resolve_titles()
    print("Resolved titles:", json.dumps(titles, ensure_ascii=False))
    for lang in LANGS:
        title = titles[lang]
        print(f"Fetching {lang}: {title!r} ...", end=" ")
        text = clean(fetch_plaintext(lang, title))
        path = os.path.join(DATA_DIR, f"{lang}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        words = len(text.split())
        chars = len(text)
        print(f"{words} words, {chars} chars -> {path}")
        if words < 500:
            print(f"  WARNING: {lang} extract looks short.", file=sys.stderr)


if __name__ == "__main__":
    main()
