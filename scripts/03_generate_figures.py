#!/usr/bin/env python3
"""
Generate figures for paper_jjdh final (v6/final).
Derived from paper/jjdh_v5/scripts/03_generate_figures.py.

Modifications applied (per 2026-04-29 reviewer comments):
  - Comment #2 (Fig.2 = all_works_top5):
      (a) Label overlap mitigated by enlarging left-margin / xlim
      (b) Ryle: "Ryle (Gilbert Ryle)" → "ライル (Gilbert Ryle)" (Katakana)
      (c) Subtitles use 「...」 instead of 『...』 for 絶対矛盾的自己同一 / 語られざる哲学
      (d) Miki's three works ordered by publication year: 1919 → 1940 → 1941
  - Comment #3 (Fig.3 = nishida_contrast):
      Title in figure: 『絶対矛盾的自己同一』 → 「絶対矛盾的自己同一」
  - Comment #5/#6/#7 (Fig.4/5/6):
      (a) 「語られざる哲学」 year corrected: 1927 → 1919
      (b) Miki's three works ordered by publication year (1919 → 1940 → 1941)
      (c) Fig.5 (heatmap) Ryle → ライル
Outputs PDFs to ./figures/ (relative to this script's directory).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# Japanese font setup
JP_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
for fp in JP_FONTS:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        break

plt.rcParams.update({
    "font.family": ["Noto Serif CJK JP", "Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent                               # repo root
DATA_DIR = PROJECT_ROOT / "results"                            # input similarity CSV
OUT_DIR = PROJECT_ROOT / "figures"                             # output figures
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping slugs to display names (Japanese)
SLUG_TO_JP = {
    "johann-herbart": "ヘルバルト",
    "nagarjuna": "ナーガールジュナ",
    "shankara": "シャンカラ",
    "paul-venice": "P.ヴェネトゥス",
    "hermann-lotze": "ロッツェ",
    "sriharsa": "シュリーハルシャ",
    "meinong": "マイノング",
    "burley": "バーリー",
    "wilhelm-wundt": "ヴント",
    "penbygull": "ペンビグル",
    "broad": "ブロード",
    "olivi": "オリヴィ",
    "mary-shepherd": "M.シェパード",
    "henry-ghent": "ヘンリクス",
    "robert-kilwardby": "キルワードビ",
    "plotinus": "プロティノス",
    "antonio-rosmini": "ロスミーニ",
    "husserl": "フッサール",
    "gangesa": "ガンゲーシャ",
    "condillac": "コンディヤック",
    "kant": "カント",
    "mill": "ミル",
    "heidegger": "ハイデガー",
    "merleau-ponty": "メルロ＝ポンティ",
    "levinas": "レヴィナス",
    "derrida": "デリダ",
    "ricoeur": "リクール",
    "sartre": "サルトル",
    "dewey": "デューイ",
    "aristotle": "アリストテレス",
    "wittgenstein": "ウィトゲンシュタイン",
    "confucius": "孔子",
    "lesniewski": "レシニェフスキ",
    "frege": "フレーゲ",
    "boole": "ブール",
    "goedel": "ゲーデル",
    "kumaarila": "クマーリラ",
    "church": "チャーチ",
    "saantarak-sita": "シャーンタラクシタ",
    "jayaraasi": "ジャヤラーシ",
    "marcel": "マルセル",
    "scheler": "シェーラー",
    "spinoza": "スピノザ",
    "maimonides-islamic": "マイモニデス",
    "stein": "シュタイン",
    "kukai": "空海",
    "dharmakiirti": "ダルマキールティ",
    "shantideva": "シャーンティデーヴァ",
    "gadamer": "ガダマー",
    "hutcheson": "ハチスン",
    "ludwig-feuerbach": "フォイエルバッハ",
    "schiller": "シラー",
    "edwards": "エドワーズ",
    "reid": "リード",
    "dilthey": "ディルタイ",
    "james": "ジェイムズ",
    "bergson": "ベルクソン",
    "nishida-kitaro": "西田幾多郎",
    "gasset": "オルテガ",
    "heinrich-rickert": "リッケルト",
    "buddha": "仏陀",
    "hare": "ヘア",
    "hume": "ヒューム",
    "astell": "アステル",
    "maimonides": "マイモニデス",
    "berkeley": "バークリー",
    # --- Added 2026-04-29 (Comment #2/#6) ---
    "gilbert-ryle": "ライル",
    "ryle": "ライル",
}

def get_jp_name(slug):
    return SLUG_TO_JP.get(slug, slug)

# Category colors
CAT_COLORS = {
    "phenomenology": "#2166AC",  # blue
    "control": "#878787",        # gray
    "indian": "#B2182B",         # red
    "german_psych": "#D6604D",   # light red/orange
    "medieval": "#92C5DE",       # light blue
    "other": "#4DAF4A",          # green
}

PHENO_SLUGS = {"husserl", "heidegger", "gadamer", "merleau-ponty", "levinas",
               "derrida", "ricoeur", "sartre"}
CTRL_SLUGS = {"kant", "mill", "dewey", "aristotle", "wittgenstein", "confucius"}
INDIAN_SLUGS = {"nagarjuna", "shankara", "sriharsa", "gangesa", "kumaarila",
                "saantarak-sita", "dharmakiirti", "jayaraasi", "kukai", "shantideva"}
GERMAN_PSYCH_SLUGS = {"johann-herbart", "hermann-lotze", "wilhelm-wundt", "meinong"}

def get_category(slug):
    if slug in PHENO_SLUGS: return "phenomenology"
    if slug in CTRL_SLUGS: return "control"
    if slug in INDIAN_SLUGS: return "indian"
    if slug in GERMAN_PSYCH_SLUGS: return "german_psych"
    return "other"

def get_cat_color(slug):
    return CAT_COLORS[get_category(slug)]


def fig1_nishida_zen_ranking():
    """善の研究 × 459名 top 20 horizontal bar chart."""
    df = pd.read_csv(DATA_DIR / "sep_person_similarity.csv")
    zen = df[df["jp_work"].str.contains("善の研究")]
    zen = zen.sort_values("similarity", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [get_cat_color(s) for s in zen["sep_slug"]]
    labels = [f"{get_jp_name(s)} ({s})" for s in zen["sep_slug"]]

    y = range(len(zen))
    ax.barh(y, zen["similarity"].values, color=colors, height=0.7, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("コサイン類似度")
    ax.set_title("西田幾多郎『善の研究』× SEP哲学者459名（上位20名）")
    ax.set_xlim(0.7, 0.87)
    ax.axvline(0.038, color="green", linestyle=":", alpha=0.5, label="ランダムベースライン")

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=CAT_COLORS["german_psych"], label="19世紀独心理学"),
        Patch(facecolor=CAT_COLORS["indian"], label="インド哲学"),
        Patch(facecolor=CAT_COLORS["phenomenology"], label="現象学"),
        Patch(facecolor=CAT_COLORS["control"], label="対照群（v1）"),
        Patch(facecolor=CAT_COLORS["other"], label="その他"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "nishida_zen_ranking.pdf")
    print(f"  Saved nishida_zen_ranking.pdf")
    plt.close(fig)


def fig2_nishida_contrast():
    """善の研究 vs 絶対矛盾的自己同一 side-by-side ranking."""
    df = pd.read_csv(DATA_DIR / "sep_person_similarity.csv")

    zen = df[df["jp_work"].str.contains("善の研究")].sort_values("similarity", ascending=False).head(15)
    abs_ = df[df["jp_work"].str.contains("絶対矛盾")].sort_values("similarity", ascending=False).head(15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # 善の研究
    colors1 = [get_cat_color(s) for s in zen["sep_slug"]]
    labels1 = [get_jp_name(s) for s in zen["sep_slug"]]
    ax1.barh(range(len(zen)), zen["similarity"].values, color=colors1, height=0.7)
    ax1.set_yticks(range(len(zen)))
    ax1.set_yticklabels(labels1, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("コサイン類似度")
    ax1.set_title("『善の研究』(1911)")
    ax1.set_xlim(0.0, 0.90)

    # 絶対矛盾的自己同一  -- Comment #3: 『...』 → 「...」 in figure title
    colors2 = [get_cat_color(s) for s in abs_["sep_slug"]]
    labels2 = [get_jp_name(s) for s in abs_["sep_slug"]]
    ax2.barh(range(len(abs_)), abs_["similarity"].values, color=colors2, height=0.7)
    ax2.set_yticks(range(len(abs_)))
    ax2.set_yticklabels(labels2, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("コサイン類似度")
    ax2.set_title("「絶対矛盾的自己同一」(1939)")
    ax2.set_xlim(0.0, 0.90)

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=CAT_COLORS["german_psych"], label="19世紀独心理学"),
        Patch(facecolor=CAT_COLORS["indian"], label="インド哲学"),
        Patch(facecolor=CAT_COLORS["phenomenology"], label="現象学"),
        Patch(facecolor=CAT_COLORS["other"], label="その他"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("西田幾多郎の初期・後期テキストにおける類似度パターンの対比", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nishida_contrast.pdf")
    print(f"  Saved nishida_contrast.pdf")
    plt.close(fig)


def fig3_boxplot_works():
    """6 JP works × 459 philosopher similarity distributions."""
    df = pd.read_csv(DATA_DIR / "sep_person_similarity.csv")

    # Comment #5: 「語られざる哲学」 1927 → 1919, and order Miki by publication year
    # Miki order: 「語られざる哲学」(1919) → 『哲学入門』(1940) → 『人生論ノート』(1941)
    work_order = [
        "善の研究",
        "絶対矛盾的自己同一",
        "「いき」の構造",
        "語られざる哲学",
        "哲学入門",
        "人生論ノート",
    ]
    work_labels = [
        "善の研究\n(西田 1911)",
        "「絶対矛盾的\n自己同一」\n(西田 1939)",
        "「いき」の\n構造\n(九鬼 1930)",
        "「語られざる\n哲学」\n(三木 1919)",
        "哲学入門\n(三木 1940)",
        "人生論ノート\n(三木 1941)",
    ]
    work_colors = ["#D6604D", "#D6604D", "#2166AC", "#4DAF4A", "#4DAF4A", "#4DAF4A"]

    data = []
    for w in work_order:
        subset = df[df["jp_work"].str.contains(w)]
        data.append(subset["similarity"].values)

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=work_labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], work_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0.038, color="green", linestyle=":", alpha=0.5, label="ランダムベースライン")
    ax.set_ylabel("コサイン類似度")
    ax.set_title("日本哲学6著作 × SEP哲学者459名の類似度分布")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "boxplot_works.pdf")
    print(f"  Saved boxplot_works.pdf")
    plt.close(fig)


def fig4_heatmap_6works():
    """Heatmap of 6 JP works mutual similarity (optional)."""
    # Compute from per_text_similarity or just skip if not needed
    pass


def fig5_all_works_top5():
    """Summary figure: each of 6 works with top 5 philosophers.
    Comment #2: (a) label/value overlap fix, (b) Ryle→ライル via SLUG_TO_JP map,
    (c) 『...』 → 「...」 for 絶対矛盾的自己同一 / 語られざる哲学,
    (d) Miki order = publication year (1919→1940→1941).
    """
    df = pd.read_csv(DATA_DIR / "sep_person_similarity.csv")

    # Reordered: 三木の三著作を出版年順 (1919, 1940, 1941)
    works = [
        ("善の研究", "西田『善の研究』(1911)"),
        ("絶対矛盾的自己同一", "西田「絶対矛盾的自己同一」(1939)"),
        ("「いき」の構造", "九鬼『「いき」の構造』(1930)"),
        ("語られざる哲学", "三木「語られざる哲学」(1919)"),
        ("哲学入門", "三木『哲学入門』(1940)"),
        ("人生論ノート", "三木『人生論ノート』(1941)"),
    ]

    # Wider figure to give labels space; slight value-text padding to avoid overlap.
    # Extra height (8.5 → 9.6) reserves a clear blank row between subplots and the
    # bottom-centered legend (Comment 2026-04-29: legend was visually overlapping
    # with subplot x-axis labels).
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.6))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(works):
        ax = axes[idx]
        subset = df[df["jp_work"].str.contains(key)].sort_values("similarity", ascending=False).head(5)
        colors = [get_cat_color(s) for s in subset["sep_slug"]]
        labels = [get_jp_name(s) for s in subset["sep_slug"]]
        sims = subset["similarity"].values

        bars = ax.barh(range(len(subset)), sims, color=colors, height=0.6)
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        # Comment #2(a): give value labels a clear right-margin so numbers don't overlap label text
        ax.set_xlim(0, 1.00)
        ax.tick_params(axis="x", labelsize=8)
        # Annotate each bar with its similarity value at the right of the bar
        for bar, v in zip(bars, sims):
            ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", ha="left", fontsize=7.5, color="#222")

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=CAT_COLORS["german_psych"], label="19世紀独心理学"),
        Patch(facecolor=CAT_COLORS["indian"], label="インド哲学"),
        Patch(facecolor=CAT_COLORS["phenomenology"], label="現象学"),
        Patch(facecolor=CAT_COLORS["other"], label="その他"),
    ]
    fig.suptitle("日本哲学6著作 × SEP哲学者459名：各著作の上位5名", fontsize=13, y=0.995)
    # tight_layout first, then reserve bottom space for the legend with clear gap
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    fig.legend(handles=legend_items, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))
    fig.savefig(OUT_DIR / "all_works_top5.pdf")
    print(f"  Saved all_works_top5.pdf")
    plt.close(fig)


def fig6_heatmap_works_philosophers():
    """Heatmap: 6 JP works × top philosophers (union of top 8 per work)."""
    from matplotlib.colors import LinearSegmentedColormap
    df = pd.read_csv(DATA_DIR / "sep_person_similarity.csv")

    # Comment #6: Miki ordered by publication year, year 1927→1919 for 語られざる哲学
    work_keys = [
        ("善の研究", "善の研究 (西田 1911)"),
        ("絶対矛盾的自己同一", "「絶対矛盾的自己同一」 (西田 1939)"),
        ("「いき」の構造", "「いき」の構造 (九鬼 1930)"),
        ("語られざる哲学", "「語られざる哲学」 (三木 1919)"),
        ("哲学入門", "哲学入門 (三木 1940)"),
        ("人生論ノート", "人生論ノート (三木 1941)"),
    ]

    # Collect union of top 8 per work
    top_slugs = set()
    for key, _ in work_keys:
        subset = df[df["jp_work"].str.contains(key)].sort_values("similarity", ascending=False).head(8)
        top_slugs.update(subset["sep_slug"].tolist())
    top_slugs = sorted(top_slugs)

    # Build matrix
    matrix = np.zeros((len(work_keys), len(top_slugs)))
    for i, (key, _) in enumerate(work_keys):
        subset = df[df["jp_work"].str.contains(key)].set_index("sep_slug")
        for j, slug in enumerate(top_slugs):
            if slug in subset.index:
                matrix[i, j] = subset.loc[slug, "similarity"]

    # Sort columns by max value across works
    col_max = matrix.max(axis=0)
    sort_idx = np.argsort(-col_max)
    matrix = matrix[:, sort_idx]
    top_slugs = [top_slugs[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = LinearSegmentedColormap.from_list("custom", ["#FFFFFF", "#FEE08B", "#F46D43", "#A50026"])
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=0.87)

    ax.set_xticks(range(len(top_slugs)))
    ax.set_xticklabels([get_jp_name(s) for s in top_slugs], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(work_keys)))
    ax.set_yticklabels([label for _, label in work_keys], fontsize=9)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if v > 0.01:
                color = "white" if v > 0.65 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="コサイン類似度")
    ax.set_title("日本哲学6著作 × SEP上位哲学者の類似度ヒートマップ", fontsize=12)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_works_philosophers.pdf")
    print(f"  Saved heatmap_works_philosophers.pdf")
    plt.close(fig)


def fig7_works_dendrogram():
    """Dendrogram: cluster 6 JP works by their 459-dim similarity profiles."""
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    df = pd.read_csv(DATA_DIR / "sep_person_similarity.csv")

    # Comment #7: 「語られざる哲学」 1927→1919, Miki publication-year order
    work_keys = [
        ("善の研究", "善の研究\n(西田 1911)"),
        ("絶対矛盾的自己同一", "「絶対矛盾的\n自己同一」\n(西田 1939)"),
        ("「いき」の構造", "「いき」の\n構造\n(九鬼 1930)"),
        ("語られざる哲学", "「語られざる\n哲学」\n(三木 1919)"),
        ("哲学入門", "哲学入門\n(三木 1940)"),
        ("人生論ノート", "人生論ノート\n(三木 1941)"),
    ]
    work_colors_map = {
        "善の研究": "#D6604D",
        "絶対矛盾的自己同一": "#D6604D",
        "「いき」の構造": "#2166AC",
        "語られざる哲学": "#4DAF4A",
        "哲学入門": "#4DAF4A",
        "人生論ノート": "#4DAF4A",
    }

    # Get all philosopher slugs
    all_slugs = sorted(df["sep_slug"].unique())

    # Build profile matrix: 6 works × N slugs
    profiles = np.zeros((len(work_keys), len(all_slugs)))
    slug_to_idx = {s: i for i, s in enumerate(all_slugs)}

    for i, (key, _) in enumerate(work_keys):
        subset = df[df["jp_work"].str.contains(key)]
        for _, row in subset.iterrows():
            j = slug_to_idx.get(row["sep_slug"])
            if j is not None:
                profiles[i, j] = row["similarity"]

    # Compute distances and linkage
    dist = pdist(profiles, metric="cosine")
    Z = linkage(dist, method="average")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.5]})

    # Dendrogram
    labels = [label for _, label in work_keys]
    dn = dendrogram(Z, labels=labels, orientation="left", ax=ax1,
                    leaf_font_size=9, color_threshold=0)
    ax1.set_xlabel("コサイン距離（類似度プロファイル間）")
    ax1.set_title("(a) 著作間クラスタリング", fontsize=11)

    # Profile correlation heatmap
    from matplotlib.colors import LinearSegmentedColormap
    corr = np.corrcoef(profiles)
    # Reorder by dendrogram leaves
    order = dn["leaves"]
    corr_ordered = corr[np.ix_(order, order)]
    labels_ordered = [labels[i] for i in order]

    cmap = LinearSegmentedColormap.from_list("custom", ["#2166AC", "#F7F7F7", "#B2182B"])
    im = ax2.imshow(corr_ordered, cmap=cmap, vmin=-0.2, vmax=1.0)
    ax2.set_xticks(range(len(labels_ordered)))
    ax2.set_xticklabels(labels_ordered, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(labels_ordered)))
    ax2.set_yticklabels(labels_ordered, fontsize=8)

    for i in range(len(labels_ordered)):
        for j in range(len(labels_ordered)):
            v = corr_ordered[i, j]
            color = "white" if abs(v) > 0.7 else "black"
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax2, shrink=0.8, label="プロファイル相関")
    ax2.set_title("(b) 類似度プロファイルの相関行列", fontsize=11)

    fig.suptitle("6著作の意味的近接性プロファイルに基づくクラスタリング", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "works_clustering.pdf")
    print(f"  Saved works_clustering.pdf")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_nishida_zen_ranking()
    fig2_nishida_contrast()
    fig3_boxplot_works()
    fig5_all_works_top5()
    fig6_heatmap_works_philosophers()
    fig7_works_dendrogram()
    print("Done.")
