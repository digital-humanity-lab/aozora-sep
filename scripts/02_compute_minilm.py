#!/usr/bin/env python3
"""
Full SEP Comparison
====================
Downloads ALL Stanford Encyclopedia of Philosophy entries,
embeds them, and compares against individual Japanese philosophical texts.
Builds on the caching and embedding infrastructure from build_primary_corpus.py.
"""

import os, sys, time, re, hashlib, zipfile, io
from pathlib import Path
from itertools import combinations

import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent  # repo root
OUTPUT_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / ".cache" / "primary"
SLUG_FILE = REPO_ROOT / "data" / "sep_all_slugs.txt"
MODEL_NAME = os.environ.get("V5_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
MODEL_TAG = os.environ.get("V5_MODEL_TAG", "minilm")  # used in output filenames

AOZORA_BASE = "https://raw.githubusercontent.com/aozorabunko/aozorabunko/master/cards"


# ---------------------------------------------------------------------------
# Text fetching (reused from build_primary_corpus.py)
# ---------------------------------------------------------------------------
def fetch_aozora_zip(card_id, zip_name):
    """Download and extract text from Aozora Bunko ZIP."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"aozora_{card_id}_{zip_name}.txt"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    url = f"{AOZORA_BASE}/{card_id}/files/{zip_name}"
    r = requests.get(url, timeout=20)
    if not r.ok:
        return None

    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in z.namelist():
        if name.endswith(".txt"):
            raw = z.read(name)
            try:
                text = raw.decode("shift_jis")
            except:
                text = raw.decode("utf-8", errors="replace")

            text = re.sub(r"《[^》]+》", "", text)
            text = re.sub(r"［＃[^］]+］", "", text)
            text = re.sub(r"｜", "", text)

            dash_pos = text.find("---\n")
            if dash_pos > 0:
                next_dash = text.find("---\n", dash_pos + 4)
                if next_dash > 0:
                    text = text[next_dash + 4:]
                else:
                    text = text[dash_pos + 4:]

            footer = text.rfind("底本：")
            if footer > 0:
                text = text[:footer]

            text = text.strip()
            if len(text) > 100:
                cache_path.write_text(text, encoding="utf-8")
                return text
    return None


def fetch_sep_section(slug):
    """Fetch and extract text from Stanford Encyclopedia of Philosophy."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"sep_{slug}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    url = f"https://plato.stanford.edu/entries/{slug}/"
    r = requests.get(url, timeout=20)
    if not r.ok:
        return None

    html = r.text
    main = re.search(r'<div id="aueditable">(.*?)</div>\s*<!--', html, re.DOTALL)
    if not main:
        main = re.search(r'<div id="main-text">(.*?)</div>\s*<div', html, re.DOTALL)
    if not main:
        main = re.search(r'<div id="preamble">(.*?)<h2[^>]*id="Bib"', html, re.DOTALL)

    text = main.group(1) if main else html

    text = re.sub(r"<h[1-6][^>]*>(.*?)</h[1-6]>", r"\n\n## \1\n\n", text)
    text = re.sub(r"<p[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) > 500:
        cache_path.write_text(text, encoding="utf-8")
        return text
    return None


def chunk_text(text, max_tokens=120, lang=None):
    """Split text into chunks that fit within the model's token limit.

    Uses sentence-boundary splitting to avoid cutting mid-sentence.
    For Japanese, splits on 。！？; for English/other, splits on .!? followed by space.
    Each chunk is kept under *max_tokens* (estimated) to minimise truncation
    by the 128-token model limit.
    """
    if lang is None:
        # Auto-detect: if >30% of characters are CJK, treat as Japanese
        cjk = sum(1 for c in text[:2000] if '\u3000' <= c <= '\u9fff' or '\uff00' <= c <= '\uffef')
        lang = "ja" if cjk > len(text[:2000]) * 0.15 else "en"

    if lang == "ja":
        sentences = re.split(r'(?<=[。！？])', text)
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)

    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

    if not sentences:
        return [text]

    chunks = []
    current = ""
    for sent in sentences:
        candidate = (current + sent) if lang == "ja" else (current + " " + sent if current else sent)
        # Rough token estimate: Japanese ~1.5 chars/token, English ~0.75 words/token
        if lang == "ja":
            est_tokens = len(candidate) / 1.5
        else:
            est_tokens = len(candidate.split()) * 1.3

        if est_tokens > max_tokens and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------------------------
# Japanese texts to compare
# ---------------------------------------------------------------------------
JP_TEXTS = {
    "Nishida Kitarō / 西田幾多郎": [
        ("000182", "946_ruby_4400.zip", "善の研究 (An Inquiry into the Good, 1911)"),
        ("000182", "1755_ruby_195.zip", "絶対矛盾的自己同一 (Absolute Contradictory Self-Identity, 1939)"),
    ],
    "Kuki Shūzō / 九鬼周造": [
        ("000065", "393_ruby_1764.zip", "「いき」の構造 (The Structure of Iki, 1930)"),
    ],
    "Miki Kiyoshi / 三木清": [
        ("000218", "1914_ruby_63481.zip", "人生論ノート (Notes on the Philosophy of Life, 1941)"),
        ("000218", "43023_ruby_26531.zip", "哲学入門 (Introduction to Philosophy, 1940)"),
        ("000218", "1072_ruby_22574.zip", "語られざる哲学 (The Unspoken Philosophy, 1927)"),
    ],
}


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL SEP COMPARISON")
    print("All SEP entries vs Japanese philosophical texts")
    print("=" * 70)

    model = SentenceTransformer(MODEL_NAME)

    # -----------------------------------------------------------------------
    # Phase 1: Fetch and embed JP texts
    # -----------------------------------------------------------------------
    print("\n--- Phase 1: Japanese texts ---\n")
    jp_works = []  # list of {philosopher, title, centroid, chars}

    for phil_name, aozora_list in JP_TEXTS.items():
        for card_id, zip_name, title in aozora_list:
            print(f"  [{phil_name}] {title}...", end=" ", flush=True)
            text = fetch_aozora_zip(card_id, zip_name)
            if not text or len(text) < 500:
                print("FAILED")
                continue
            chunks = chunk_text(text)
            embs = model.encode(chunks, normalize_embeddings=True,
                                batch_size=64, show_progress_bar=False)
            centroid = np.mean(embs, axis=0)
            centroid /= np.linalg.norm(centroid)
            jp_works.append({
                "philosopher": phil_name,
                "title": title,
                "centroid": centroid,
                "chars": len(text),
                "n_chunks": len(chunks),
            })
            print(f"{len(text):,} chars, {len(chunks)} chunks")

    print(f"\n  Total JP works: {len(jp_works)}")

    # -----------------------------------------------------------------------
    # Phase 2: Fetch and embed ALL SEP entries
    # -----------------------------------------------------------------------
    print("\n--- Phase 2: SEP entries ---\n")

    slugs = SLUG_FILE.read_text().strip().split("\n")
    print(f"  Total SEP slugs: {len(slugs)}")

    # Check how many are already cached
    cached = sum(1 for s in slugs if (CACHE_DIR / f"sep_{s}.txt").exists())
    print(f"  Already cached: {cached}")
    print(f"  To download: {len(slugs) - cached}")

    sep_entries = []  # list of {slug, centroid, chars, n_chunks}
    failed = []
    batch_size = 50  # print progress every N entries

    for idx, slug in enumerate(slugs):
        if (idx + 1) % batch_size == 0 or idx == 0:
            elapsed = time.time() - start
            print(f"  [{idx+1:4d}/{len(slugs)}] ({elapsed:.0f}s) Processing {slug}...")

        text = fetch_sep_section(slug)
        if not text or len(text) < 1000:
            failed.append(slug)
            continue

        chunks = chunk_text(text)
        embs = model.encode(chunks, normalize_embeddings=True,
                            batch_size=64, show_progress_bar=False)
        centroid = np.mean(embs, axis=0)
        centroid /= np.linalg.norm(centroid)
        sep_entries.append({
            "slug": slug,
            "centroid": centroid,
            "chars": len(text),
            "n_chunks": len(chunks),
        })

        # Polite delay for uncached entries
        if not (CACHE_DIR / f"sep_{slug}.txt").exists():
            time.sleep(0.3)

    print(f"\n  Successfully embedded: {len(sep_entries)}")
    print(f"  Failed/too short: {len(failed)}")

    # -----------------------------------------------------------------------
    # Phase 3: Compute similarities
    # -----------------------------------------------------------------------
    print("\n--- Phase 3: Similarity computation ---\n")

    all_rows = []
    for jp in jp_works:
        sims = []
        for sep in sep_entries:
            sim = float(np.dot(jp["centroid"], sep["centroid"]))
            sims.append((sep["slug"], sim, sep["chars"]))
            all_rows.append({
                "jp_philosopher": jp["philosopher"],
                "jp_work": jp["title"],
                "jp_chars": jp["chars"],
                "sep_slug": sep["slug"],
                "sep_chars": sep["chars"],
                "similarity": round(sim, 4),
            })

        sims.sort(key=lambda x: -x[1])
        print(f"  {jp['philosopher']}: {jp['title'][:50]}")
        print(f"  Top 20:")
        for rank, (slug, sim, chars) in enumerate(sims[:20], 1):
            print(f"    {rank:3d}. {sim:.3f}  {slug} ({chars:,} chars)")
        print(f"  Bottom 5:")
        for slug, sim, chars in sims[-5:]:
            print(f"         {sim:.3f}  {slug} ({chars:,} chars)")
        print()

    # Save full results — filename includes MODEL_TAG for cross-model comparison
    suffix = "" if MODEL_TAG == "minilm" else f"_{MODEL_TAG}"
    df = pd.DataFrame(all_rows)
    outpath = OUTPUT_DIR / f"sep_full_similarity{suffix}.csv"
    df.to_csv(outpath, index=False)
    print(f"  Saved: {outpath} ({len(all_rows)} rows)")

    # Save summary (top 50 per JP work)
    summary_rows = []
    for jp_work, g in df.groupby("jp_work"):
        g_sorted = g.sort_values("similarity", ascending=False)
        for rank, (_, r) in enumerate(g_sorted.head(50).iterrows(), 1):
            summary_rows.append({**r.to_dict(), "rank": rank})
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / f"sep_full_top50{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path} ({len(summary_rows)} rows)")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"DONE ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
