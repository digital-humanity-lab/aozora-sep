#!/usr/bin/env python3
"""
Primary Text Corpus Builder
============================
Downloads actual primary philosophical texts from Aozora Bunko (JP)
and uses SEP + manual key passages for Western philosophers.
Embeds with sentence-transformers and computes thought affinities.
"""

import os, sys, time, re, hashlib, zipfile, io
from pathlib import Path
from itertools import combinations

import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root (flat structure)
OUTPUT_DIR = PROJECT_ROOT / "corpus"                   # JP corpus build output
CACHE_DIR = PROJECT_ROOT / ".cache" / "primary"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

AOZORA_BASE = "https://raw.githubusercontent.com/aozorabunko/aozorabunko/master/cards"

# ---------------------------------------------------------------------------
# Aozora Bunko text extraction
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
        print(f"    FAILED: {url} ({r.status_code})")
        return None

    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in z.namelist():
        if name.endswith(".txt"):
            raw = z.read(name)
            try:
                text = raw.decode("shift_jis")
            except:
                text = raw.decode("utf-8", errors="replace")

            # Clean Aozora markup
            text = re.sub(r"《[^》]+》", "", text)  # Ruby
            text = re.sub(r"［＃[^］]+］", "", text)  # Formatting
            text = re.sub(r"｜", "", text)  # Ruby markers

            # Remove header (up to first ---)
            dash_pos = text.find("---\n")
            if dash_pos > 0:
                next_dash = text.find("---\n", dash_pos + 4)
                if next_dash > 0:
                    text = text[next_dash + 4:]
                else:
                    text = text[dash_pos + 4:]

            # Remove footer (from 底本：)
            footer = text.rfind("底本：")
            if footer > 0:
                text = text[:footer]

            text = text.strip()
            if len(text) > 100:
                cache_path.write_text(text, encoding="utf-8")
                return text
    return None


def fetch_sep_section(slug, section_ids=None):
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
    # Extract main article content
    main = re.search(r'<div id="aueditable">(.*?)</div>\s*<!--', html, re.DOTALL)
    if not main:
        main = re.search(r'<div id="main-text">(.*?)</div>\s*<div', html, re.DOTALL)
    if not main:
        # Fallback: get everything between preamble and bibliography
        main = re.search(r'<div id="preamble">(.*?)<h2[^>]*id="Bib"', html, re.DOTALL)

    text = main.group(1) if main else html

    # Remove HTML tags but keep structure
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


# ---------------------------------------------------------------------------
# Corpus definition — PRIMARY TEXTS
# ---------------------------------------------------------------------------
CORPUS = {
    "JP": {
        "Nishida Kitarō / 西田幾多郎": {
            "aozora": [
                ("000182", "946_ruby_4400.zip", "善の研究 (An Inquiry into the Good, 1911)"),
                ("000182", "1755_ruby_195.zip", "絶対矛盾的自己同一 (Absolute Contradictory Self-Identity, 1939)"),
            ],
        },
        "Kuki Shūzō / 九鬼周造": {
            "aozora": [
                ("000065", "393_ruby_1764.zip", "「いき」の構造 (The Structure of Iki, 1930)"),
            ],
        },
        "Miki Kiyoshi / 三木清": {
            "aozora": [
                ("000218", "1914_ruby_63481.zip", "人生論ノート (Notes on the Philosophy of Life, 1941)"),
                ("000218", "43023_ruby_26531.zip", "哲学入門 (Introduction to Philosophy, 1940)"),
                ("000218", "1072_ruby_22574.zip", "語られざる哲学 (The Unspoken Philosophy, 1927)"),
            ],
        },
        "Watsuji Tetsurō / 和辻哲郎": {
            "aozora": [
                ("001395", "49881_ruby_45902.zip", "埋もれた日本 (Buried Japan)"),
                ("001395", "49873_ruby_42489.zip", "「自然」を深めよ (Deepen 'Nature')"),
                ("001395", "49878_ruby_42772.zip", "生きること作ること (Living and Creating)"),
                ("001395", "49874_ruby_42525.zip", "『偶像再興』序言 (Preface to Revival of Idols)"),
            ],
            "manual": [
                ("風土 (Fūdo/Climate, 1935) — 概要",
                 """風土は単なる自然環境ではない。それは人間の自己了解の契機である。
人間は風土のうちに自己を見出す。モンスーン的風土における人間は受容的・忍従的であり、
砂漠的風土においては戦闘的・服従的であり、牧場的風土においては合理的・支配的である。
しかしこの類型論は決定論ではない。風土は人間存在の具体的な在り方を規定する条件であるが、
同時に人間は風土を通じて自己を了解し、自己を形成するのである。
ハイデガーは時間性を人間存在の根本構造として分析したが、空間性・風土性もまた等根源的に重要である。"""),
                ("倫理学 (Rinrigaku/Ethics, 1937-49) — 概要",
                 """人間の学としての倫理学。倫理とは人と人との間柄の道理である。
人間とは世の中における人であり、社会における個人である。
人間という言葉はその中に「間」を含んでいる。これは人間存在の根本構造が「間柄」にあることを示している。
人間は孤立した個人ではなく、つねに他者との関係のうちに存在する。
この関係の在り方が倫理の基礎をなす。"""),
            ],
        },
        "Nishitani Keiji / 西谷啓治": {
            "sep": "kyoto-school",  # Nishitani sections within Kyoto School entry
            "manual": [
                ("宗教とは何か (Religion and Nothingness, 1961) — 概要",
                 """ニヒリズムの問題は近代の根本問題である。ニーチェが「神は死んだ」と宣言したとき、
それは単に信仰の問題ではなく、存在そのものの意味が問われることになった。
虚無の深淵が開かれた。しかし虚無を単に否定的に捉えることは、虚無の本質を見失うことである。
空の立場においては、虚無は存在と表裏一体のものとして了解される。
空とは有でも無でもなく、有と無の二元対立を超えた根源的な場である。
この空の立場に立つとき、あらゆる存在者はそのあるがままの姿において現れる。
ハイデガーの「無」の概念は、この方向への重要な一歩であったが、
なお存在の側からの問いにとどまっている。空の立場は、存在の問い自体を根底から転換するものである。"""),
            ],
        },
        "Tanabe Hajime / 田辺元": {
            "sep": "kyoto-school",
            "manual": [
                ("懺悔道としての哲学 (Philosophy as Metanoetics, 1946) — 概要",
                 """哲学は自力の理性的思索によっては究極の真理に到達しえない。
理性の行き詰まりにおいて、理性は自己の無力を自覚し、懺悔する。
この懺悔を通じて、他力の働きが現れる。これが懺悔道としての哲学である。
ヘーゲルの弁証法は絶対精神の自己展開として歴史を捉えるが、
それは理性の自力的立場にとどまっている。
絶対弁証法は、理性の自己否定を通じて、理性を超えたものの働きを受けることによって成立する。"""),
                ("種の論理 (Logic of Species) — 概要",
                 """種の論理の問題は、個と普遍の媒介の問題である。
現実の社会的存在においては、個人は直接に普遍に関わるのではなく、
種（家族、民族、国家）を媒介として普遍に参与する。
この種の媒介は単に概念的な問題ではなく、存在の根本構造に関わる問題である。
種は個を基礎づけると同時に、個を制限し、否定する。"""),
            ],
        },
    },
    "DE": {
        "Martin Heidegger": {"sep": "heidegger"},
        "Edmund Husserl": {"sep": "husserl"},
        "Hans-Georg Gadamer": {"sep": "gadamer"},
    },
    "FR": {
        "Maurice Merleau-Ponty": {"sep": "merleau-ponty"},
        "Emmanuel Levinas": {"sep": "levinas"},
        "Jacques Derrida": {"sep": "derrida"},
        "Paul Ricoeur": {"sep": "ricoeur"},
        "Jean-Paul Sartre": {"sep": "sartre"},
    },
    "CTRL": {
        "Immanuel Kant": {"sep": "kant"},
        "John Stuart Mill": {"sep": "mill"},
        "John Dewey": {"sep": "dewey"},
        "Aristotle": {"sep": "aristotle"},
        "Ludwig Wittgenstein": {"sep": "wittgenstein"},
        "Confucius": {"sep": "confucius"},
    },
}


def chunk_text(text, max_words=200, overlap=50):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + max_words]))
        i += max_words - overlap
    return chunks


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRIMARY TEXT ANALYSIS")
    print("Aozora Bunko originals + Stanford Encyclopedia of Philosophy")
    print("=" * 70)

    model = SentenceTransformer(MODEL_NAME)

    # Phase 1: Fetch all texts
    all_phils = []
    for cc, philosophers in CORPUS.items():
        for name, sources in philosophers.items():
            print(f"\n  [{cc}] {name}")
            texts = []

            # Aozora Bunko primary texts
            for card_id, zip_name, title in sources.get("aozora", []):
                print(f"    [aozora] {title}...", end=" ", flush=True)
                text = fetch_aozora_zip(card_id, zip_name)
                if text and len(text) > 500:
                    texts.append({"title": title, "text": text,
                                  "type": "primary_aozora", "chars": len(text)})
                    print(f"{len(text):,} chars")
                else:
                    print("FAILED")
                time.sleep(0.2)

            # SEP article
            sep_slug = sources.get("sep")
            if sep_slug:
                print(f"    [SEP] {sep_slug}...", end=" ", flush=True)
                text = fetch_sep_section(sep_slug)
                if text and len(text) > 1000:
                    texts.append({"title": f"SEP: {sep_slug}", "text": text,
                                  "type": "sep_article", "chars": len(text)})
                    print(f"{len(text):,} chars")
                else:
                    print("FAILED")

            # Manual passages
            for title, text in sources.get("manual", []):
                texts.append({"title": title, "text": text,
                              "type": "manual_summary", "chars": len(text)})
                print(f"    [manual] {title[:50]}... {len(text)} chars")

            all_phils.append({"country": cc, "name": name, "texts": texts})

    # Phase 2: Embed
    print(f"\n{'='*70}\nEmbedding with chunking...\n")

    TYPE_WEIGHT = {
        "primary_aozora": 5.0,
        "sep_article": 3.0,
        "manual_summary": 4.0,
    }

    centroids = {}
    text_embs = {}

    for phil in all_phils:
        key = (phil["country"], phil["name"])
        all_chunks = []
        all_weights = []
        chunk_sources = []

        for t in phil["texts"]:
            if not t["text"] or len(t["text"]) < 50:
                continue
            chunks = chunk_text(t["text"])
            w = TYPE_WEIGHT.get(t["type"], 1.0)
            for c in chunks:
                all_chunks.append(c)
                all_weights.append(w)
                chunk_sources.append(t)

        if not all_chunks:
            print(f"  {cc}:{phil['name']} — no texts!")
            continue

        embs = model.encode(all_chunks, normalize_embeddings=True,
                           batch_size=64, show_progress_bar=False)
        weights = np.array(all_weights)
        centroid = np.average(embs, axis=0, weights=weights)
        centroid /= np.linalg.norm(centroid)
        centroids[key] = centroid

        # Aggregate per-text
        text_level = {}
        for i, t in enumerate(chunk_sources):
            tid = id(t)
            if tid not in text_level:
                text_level[tid] = {"meta": t, "embs": []}
            text_level[tid]["embs"].append(embs[i])

        text_embs[key] = []
        for info in text_level.values():
            avg = np.mean(info["embs"], axis=0)
            avg /= np.linalg.norm(avg)
            text_embs[key].append((info["meta"], avg))

        total_chars = sum(t["chars"] for t in phil["texts"])
        n_primary = sum(1 for t in phil["texts"] if t["type"] == "primary_aozora")
        n_sep = sum(1 for t in phil["texts"] if t["type"] == "sep_article")
        print(f"  {cc}:{phil['name']:35s} {len(all_chunks):5d} chunks | "
              f"{total_chars:>9,} chars | {n_primary} primary, {n_sep} SEP")

    # Phase 3: Affinity
    print(f"\n{'='*70}")
    print("PHILOSOPHER THOUGHT AFFINITY (primary texts)\n")

    rows = []
    for ki, kj in combinations(centroids.keys(), 2):
        if ki[0] == kj[0]:
            continue
        sim = float(np.dot(centroids[ki], centroids[kj]))
        rows.append({
            "philosopher_a": ki[1], "country_a": ki[0],
            "philosopher_b": kj[1], "country_b": kj[0],
            "thought_similarity": round(sim, 4),
        })

    affinity = pd.DataFrame(rows).sort_values("thought_similarity", ascending=False)
    affinity.to_csv(OUTPUT_DIR / "thought_affinity.csv", index=False)

    for _, r in affinity.head(25).iterrows():
        print(f"  {r['thought_similarity']:.3f}  "
              f"{r['country_a']}:{r['philosopher_a'][:30]:30s} <-> "
              f"{r['country_b']}:{r['philosopher_b'][:30]}")

    # Phase 4: JP-focused work-level bridges
    print(f"\n{'='*70}")
    print("WORK-LEVEL CONCEPTUAL BRIDGES\n")

    bridge_rows = []
    for phil in all_phils:
        if phil["country"] != "JP":
            continue
        jp_key = (phil["country"], phil["name"])
        if jp_key not in text_embs:
            continue

        print(f"--- {phil['name']} ---\n")

        for other in all_phils:
            if other["country"] == "JP":
                continue
            other_key = (other["country"], other["name"])
            if other_key not in text_embs:
                continue

            best_sim = 0
            best_pair = None
            for jp_t, jp_e in text_embs[jp_key]:
                for ot_t, ot_e in text_embs[other_key]:
                    sim = float(np.dot(jp_e, ot_e))
                    if sim > best_sim and sim < 0.98:
                        best_sim = sim
                        best_pair = (jp_t, ot_t)

            if best_pair and best_sim > 0.3:
                jp_t, ot_t = best_pair
                print(f"  {best_sim:.3f}  <-> {other['country']}:{other['name'][:30]}")
                print(f"    JP: {jp_t['title'][:60]} [{jp_t['type']}]")
                jp_preview = jp_t["text"][:150].replace("\n", " ")
                print(f"      \"{jp_preview}...\"")
                print(f"    Other: {ot_t['title'][:60]} [{ot_t['type']}]")
                ot_preview = ot_t["text"][:150].replace("\n", " ")
                print(f"      \"{ot_preview}...\"")
                print()
                bridge_rows.append({
                    "jp_philosopher": phil["name"],
                    "jp_work": jp_t["title"],
                    "jp_type": jp_t["type"],
                    "jp_chars": jp_t["chars"],
                    "other_country": other["country"],
                    "other_philosopher": other["name"],
                    "other_work": ot_t["title"],
                    "other_type": ot_t["type"],
                    "similarity": round(best_sim, 4),
                })

    pd.DataFrame(bridge_rows).to_csv(OUTPUT_DIR / "work_bridges.csv", index=False)

    # Phase 4b: Per-text similarity — each JP text vs all non-JP philosopher centroids
    print(f"\n{'='*70}")
    print("PER-TEXT SIMILARITY (each JP text vs non-JP philosopher centroids)\n")

    per_text_rows = []
    non_jp_keys = [k for k in centroids if k[0] != "JP"]

    for phil in all_phils:
        if phil["country"] != "JP":
            continue
        jp_key = (phil["country"], phil["name"])
        if jp_key not in text_embs:
            continue

        print(f"--- {phil['name']} ---")
        for t_meta, t_emb in text_embs[jp_key]:
            print(f"\n  [{t_meta['type']}] {t_meta['title'][:60]} ({t_meta['chars']:,} chars)")
            sims = []
            for nk in non_jp_keys:
                sim = float(np.dot(t_emb, centroids[nk]))
                sims.append((nk, sim))
                per_text_rows.append({
                    "jp_philosopher": phil["name"],
                    "jp_work": t_meta["title"],
                    "jp_type": t_meta["type"],
                    "jp_chars": t_meta["chars"],
                    "other_country": nk[0],
                    "other_philosopher": nk[1],
                    "similarity": round(sim, 4),
                })
            sims.sort(key=lambda x: -x[1])
            for (cc, name), sim in sims[:5]:
                print(f"    {sim:.3f}  {cc}:{name[:35]}")

    per_text_df = pd.DataFrame(per_text_rows)
    per_text_df.to_csv(OUTPUT_DIR / "per_text_similarity.csv", index=False)
    print(f"\n  -> Saved per_text_similarity.csv ({len(per_text_rows)} rows)")

    # Phase 5: JP internal affinity (for reference)
    print(f"\n{'='*70}")
    print("JP INTERNAL AFFINITY\n")
    jp_keys = [k for k in centroids if k[0] == "JP"]
    for ki, kj in combinations(jp_keys, 2):
        sim = float(np.dot(centroids[ki], centroids[kj]))
        print(f"  {sim:.3f}  {ki[1][:30]:30s} <-> {kj[1][:30]}")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"DONE ({elapsed:.0f}s)")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
