#!/usr/bin/env python3
"""
Chunk-level similarity analysis: captures specific conceptual connections
that centroid averaging destroys.

Instead of comparing centroid-to-centroid, computes:
1. Max chunk-pair similarity (strongest local connection)
2. Top-k average chunk-pair similarity
3. Avg-max-per-row (directional: for each JP chunk, best SEP match)

This addresses the James problem: if 善の研究 has specific passages about
"pure experience" that echo James, centroid averaging washes this signal out.
"""

import sys, os, re, time, json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root (flat structure)
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / ".cache" / "primary"
OUTPUT_DIR = PROJECT_ROOT / "results" / "chunk_analysis"
SLUG_FILE = PROJECT_ROOT / "data" / "sep_person_slugs_wikidata.txt"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

AOZORA_BASE = "https://raw.githubusercontent.com/aozorabunko/aozorabunko/master/cards"


# ---------------------------------------------------------------------------
# Text loading (from cache only)
# ---------------------------------------------------------------------------
def fetch_aozora_text(card_id, zip_name):
    """Load cached Aozora text."""
    cache_path = CACHE_DIR / f"aozora_{card_id}_{zip_name}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return None


def fetch_sep_text(slug):
    """Load cached SEP text by slug."""
    cache_path = CACHE_DIR / f"sep_{slug}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return None


# ---------------------------------------------------------------------------
# Sentence-based chunking (same as corrected compare_sep_full.py)
# ---------------------------------------------------------------------------
def chunk_text(text, max_tokens=120, lang=None):
    """Split text into chunks that fit within the model's token limit.

    Uses sentence-boundary splitting to avoid cutting mid-sentence.
    For Japanese, splits on 。！？; for English/other, splits on .!? followed by space.
    """
    if lang is None:
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
# Similarity metrics
# ---------------------------------------------------------------------------
def compute_chunk_similarities(embs_a, embs_b):
    """Compute all pairwise similarities between two sets of chunk embeddings."""
    return embs_a @ embs_b.T  # (n, m) cosine similarities (normalized)


def max_chunk_similarity(sim_matrix):
    """Maximum chunk pair similarity."""
    return float(np.max(sim_matrix))


def topk_chunk_similarity(sim_matrix, k=10):
    """Average of top-k chunk pair similarities."""
    flat = sim_matrix.flatten()
    if len(flat) < k:
        k = len(flat)
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def avg_max_per_row(sim_matrix):
    """For each chunk of text A, find max similarity to any chunk of text B, then average."""
    return float(np.mean(np.max(sim_matrix, axis=1)))


# ---------------------------------------------------------------------------
# Japanese texts
# ---------------------------------------------------------------------------
JP_TEXTS = [
    ("Nishida", "000182", "946_ruby_4400.zip", "善の研究 (An Inquiry into the Good, 1911)"),
    ("Nishida", "000182", "1755_ruby_195.zip", "絶対矛盾的自己同一 (Absolute Contradictory Self-Identity, 1939)"),
]


# ---------------------------------------------------------------------------
# Embedding cache for efficiency
# ---------------------------------------------------------------------------
EMB_CACHE_DIR = PROJECT_ROOT / ".cache" / "chunk_embs"


def cache_path_for(prefix, key):
    """Return path for cached embeddings."""
    safe_key = re.sub(r'[^\w\-.]', '_', key)
    return EMB_CACHE_DIR / f"{prefix}_{safe_key}.npz"


def encode_and_cache(model, chunks, prefix, key):
    """Encode chunks, caching the result."""
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = cache_path_for(prefix, key)
    if cp.exists():
        data = np.load(cp)
        return data["embs"]
    embs = model.encode(chunks, normalize_embeddings=True,
                        batch_size=64, show_progress_bar=False)
    np.savez_compressed(cp, embs=embs)
    return embs


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed")
        sys.exit(1)

    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CHUNK-LEVEL SIMILARITY ANALYSIS")
    print("Centroid vs Max vs Top-k vs Avg-Max")
    print("=" * 70)

    model = SentenceTransformer(MODEL_NAME)

    # -------------------------------------------------------------------
    # Phase 1: Load and embed JP texts
    # -------------------------------------------------------------------
    print("\n--- Phase 1: Japanese texts ---\n")
    jp_data = []
    for phil, card_id, zip_name, title in JP_TEXTS:
        text = fetch_aozora_text(card_id, zip_name)
        if not text:
            print(f"  SKIP: {title} (not cached)")
            continue
        chunks = chunk_text(text, lang="ja")
        embs = encode_and_cache(model, chunks, "jp", f"{card_id}_{zip_name}")
        centroid = np.mean(embs, axis=0)
        centroid /= np.linalg.norm(centroid)
        jp_data.append({
            "phil": phil, "title": title,
            "chunks": chunks, "embs": embs, "centroid": centroid,
            "chars": len(text), "n_chunks": len(chunks),
        })
        print(f"  {title}: {len(text):,} chars, {len(chunks)} chunks")

    # -------------------------------------------------------------------
    # Phase 2: Load philosopher slugs and compute all metrics
    # -------------------------------------------------------------------
    print("\n--- Phase 2: SEP philosophers (Wikidata-filtered) ---\n")
    slugs = SLUG_FILE.read_text().strip().split("\n")
    print(f"  Total philosopher slugs: {len(slugs)}")

    all_rows = []
    for idx, slug in enumerate(slugs):
        sep_text = fetch_sep_text(slug)
        if not sep_text or len(sep_text) < 1000:
            continue

        sep_chunks = chunk_text(sep_text, lang="en")
        sep_embs = encode_and_cache(model, sep_chunks, "sep", slug)
        sep_centroid = np.mean(sep_embs, axis=0)
        sep_centroid /= np.linalg.norm(sep_centroid)

        for jp in jp_data:
            # Centroid similarity
            centroid_sim = float(jp["centroid"] @ sep_centroid)

            # Chunk-level metrics
            sim_mat = compute_chunk_similarities(jp["embs"], sep_embs)
            max_sim = max_chunk_similarity(sim_mat)
            top10_sim = topk_chunk_similarity(sim_mat, k=10)
            top50_sim = topk_chunk_similarity(sim_mat, k=50)
            avg_max_sim = avg_max_per_row(sim_mat)

            # Best chunk pair details
            best_i, best_j = np.unravel_index(np.argmax(sim_mat), sim_mat.shape)

            all_rows.append({
                "jp_work": jp["title"],
                "sep_slug": slug,
                "sep_chars": len(sep_text),
                "sep_n_chunks": len(sep_chunks),
                "centroid": round(centroid_sim, 4),
                "max_chunk": round(max_sim, 4),
                "top10_chunk": round(top10_sim, 4),
                "top50_chunk": round(top50_sim, 4),
                "avg_max": round(avg_max_sim, 4),
                "best_jp_chunk_idx": int(best_i),
                "best_sep_chunk_idx": int(best_j),
                "best_jp_chunk": jp["chunks"][best_i][:200],
                "best_sep_chunk": sep_chunks[best_j][:200],
            })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{idx+1}/{len(slugs)}] ({elapsed:.0f}s)")

    df = pd.DataFrame(all_rows)

    # -------------------------------------------------------------------
    # Phase 3: Output results
    # -------------------------------------------------------------------
    print(f"\n--- Phase 3: Results ---\n")

    # Save full results
    out_path = OUTPUT_DIR / "chunk_level_similarity.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df)} rows)")

    # Print rankings for 善の研究
    zen = df[df["jp_work"].str.contains("善の研究")]
    if len(zen) > 0:
        print("\n" + "=" * 100)
        print("善の研究: CENTROID vs CHUNK-LEVEL RANKINGS")
        print("=" * 100)

        for metric, label in [
            ("centroid", "Centroid"),
            ("max_chunk", "Max Chunk"),
            ("top10_chunk", "Top-10 Chunk"),
            ("avg_max", "Avg-Max"),
        ]:
            ranked = zen.sort_values(metric, ascending=False).reset_index(drop=True)
            james_row = ranked[ranked["sep_slug"] == "james"]
            james_rank = james_row.index[0] + 1 if len(james_row) > 0 else "N/A"
            james_val = james_row[metric].values[0] if len(james_row) > 0 else "N/A"

            print(f"\n--- {label} ---")
            print(f"  James rank: {james_rank}/{len(ranked)} (score: {james_val})")
            print(f"  Top 15:")
            for i, (_, r) in enumerate(ranked.head(15).iterrows(), 1):
                marker = " <<<" if r["sep_slug"] == "james" else ""
                print(f"    {i:3d}. {r[metric]:.4f}  {r['sep_slug']}{marker}")

            # Also show James's neighborhood
            if isinstance(james_rank, int) and james_rank > 15:
                print(f"  Around James (#{james_rank}):")
                start_idx = max(0, james_rank - 3)
                end_idx = min(len(ranked), james_rank + 2)
                for i in range(start_idx, end_idx):
                    r = ranked.iloc[i]
                    marker = " <<<" if r["sep_slug"] == "james" else ""
                    print(f"    {i+1:3d}. {r[metric]:.4f}  {r['sep_slug']}{marker}")

        # Show best chunk pair for James
        james_data = zen[zen["sep_slug"] == "james"]
        if len(james_data) > 0:
            row = james_data.iloc[0]
            print(f"\n--- Best chunk pair: 善の研究 × James ---")
            print(f"  Max chunk similarity: {row['max_chunk']:.4f}")
            print(f"  JP chunk [{row['best_jp_chunk_idx']}]: {row['best_jp_chunk']}")
            print(f"  SEP chunk [{row['best_sep_chunk_idx']}]: {row['best_sep_chunk']}")

        # Key comparison table
        key_slugs = ["james", "spinoza", "johann-herbart", "nagarjuna",
                     "husserl", "nishida-kitaro", "bergson", "kant",
                     "shankara", "broad", "rosmini"]
        print(f"\n--- Key philosophers comparison ---")
        print(f"{'Slug':<22s}  {'Centroid':>8s} {'Rank':>5s}  {'MaxChk':>8s} {'Rank':>5s}  "
              f"{'Top10':>8s} {'Rank':>5s}  {'AvgMax':>8s} {'Rank':>5s}")
        print("-" * 105)

        for metric in ["centroid", "max_chunk", "top10_chunk", "avg_max"]:
            zen[f"rank_{metric}"] = zen[metric].rank(ascending=False).astype(int)

        for slug in key_slugs:
            row = zen[zen["sep_slug"] == slug]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"{slug:<22s}  {r['centroid']:8.4f} {r['rank_centroid']:5d}  "
                  f"{r['max_chunk']:8.4f} {r['rank_max_chunk']:5d}  "
                  f"{r['top10_chunk']:8.4f} {r['rank_top10_chunk']:5d}  "
                  f"{r['avg_max']:8.4f} {r['rank_avg_max']:5d}")

    # Same for 絶対矛盾的自己同一
    zettai = df[df["jp_work"].str.contains("絶対矛盾")]
    if len(zettai) > 0:
        print("\n\n" + "=" * 100)
        print("絶対矛盾的自己同一: KEY PHILOSOPHERS")
        print("=" * 100)

        key_slugs_z = ["james", "husserl", "dilthey", "nagarjuna", "spinoza",
                       "nishida-kitaro", "heidegger", "bergson"]
        for metric in ["centroid", "max_chunk", "top10_chunk", "avg_max"]:
            zettai[f"rank_{metric}"] = zettai[metric].rank(ascending=False).astype(int)

        print(f"{'Slug':<22s}  {'Centroid':>8s} {'Rank':>5s}  {'MaxChk':>8s} {'Rank':>5s}  "
              f"{'Top10':>8s} {'Rank':>5s}  {'AvgMax':>8s} {'Rank':>5s}")
        print("-" * 105)
        for slug in key_slugs_z:
            row = zettai[zettai["sep_slug"] == slug]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"{slug:<22s}  {r['centroid']:8.4f} {r['rank_centroid']:5d}  "
                  f"{r['max_chunk']:8.4f} {r['rank_max_chunk']:5d}  "
                  f"{r['top10_chunk']:8.4f} {r['rank_top10_chunk']:5d}  "
                  f"{r['avg_max']:8.4f} {r['rank_avg_max']:5d}")

    elapsed = time.time() - start
    print(f"\nDONE ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
