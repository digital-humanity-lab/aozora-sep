# aozora-sep

Reproduction package for:

> **多言語文埋め込みによる日本哲学テキストとSEP全哲学者記事の意味的近接性の探索的計量分析**
> *(An Exploratory Quantitative Analysis of Semantic Proximity between Japanese Philosophy Texts and SEP Philosopher Articles via Multilingual Sentence Embeddings)*
>
> 犬塚 悠（名古屋工業大学）, 松井 佑介（名古屋大学）
>
> Submitted to *Japanese Journal of Digital Humanities* (JJDH), 2026.

This repository contains the **reproduction package** — analysis code, input data, intermediate results, and final figures — for the above study, which compares 6 Japanese primary philosophical texts (Aozora Bunko) against 458 SEP philosopher articles (after applying a 1,000-character minimum filter to the 459 entries originally identified via Wikidata) using a multilingual sentence-embedding model.

## Repository structure

```
aozora-sep/
├── README.md                  ← this file
├── LICENSE                    ← MIT
├── Makefile                   ← reproduction targets
├── requirements.txt           ← Python dependencies
├── scripts/
│   ├── 01_build_jp_corpus.py        ← fetch JP texts from Aozora Bunko
│   ├── 02_compute_minilm.py         ← compute MiniLM similarities (main)
│   ├── 03_generate_figures.py       ← regenerate paper figures
│   ├── 05_compute_labse.py          ← LaBSE robustness check
│   └── 06_chunk_level_analysis.py   ← chunk-level supplementary analysis
├── data/                      ← inputs (committed)
│   ├── sep_all_slugs.txt              ← all 1,856 SEP entry slugs
│   ├── sep_person_slugs_wikidata.txt  ← 459 person slugs (Wikidata-filtered)
│   └── sep_person_wikidata_mapping.json  ← slug → Wikidata QID
├── results/                   ← computed outputs (committed for quick reproduction)
│   ├── sep_full_similarity.csv        ← 6 JP works × 1,856 SEP entries
│   ├── sep_full_top50.csv             ← top-50 extracts
│   ├── sep_person_similarity.csv      ← 6 JP works × 458 philosophers (after 1,000-char filter)
│   ├── sep_person_top50.csv           ← top-50 extracts
│   └── bootstrap_ci.csv               ← bootstrap confidence intervals (top-5 per work)
└── figures/                   ← final paper figures (PDF + PNG)
    ├── nishida_zen_ranking.{pdf,png}              ← Fig. 1
    ├── all_works_top5.{pdf,png}                   ← Fig. 2
    ├── nishida_contrast.{pdf,png}                 ← Fig. 3
    ├── boxplot_works.{pdf,png}                    ← Fig. 4
    ├── heatmap_works_philosophers.{pdf,png}       ← Fig. 5
    └── works_clustering.{pdf,png}                 ← Fig. 6
```

> **Note**: The paper PDF/DOCX itself is not included in this repository. Please refer to the published version in *Japanese Journal of Digital Humanities*.

## Quick reproduction

The repo ships with `results/` already populated, so the most common task — regenerating the paper's figures — takes only seconds:

```bash
pip install -r requirements.txt
make figures      # regenerates figures/ from results/sep_person_similarity.csv
```

## Full reproduction from sources

Re-running the entire pipeline (~50 min on CPU, ~10 min on GPU) downloads the Japanese primary texts from Aozora Bunko, fetches SEP articles, computes embeddings with `paraphrase-multilingual-MiniLM-L12-v2`, and writes both `results/` and `figures/`:

```bash
pip install -r requirements.txt
export OPENALEX_MAILTO="your-email@example.com"   # politeness for any future API calls
make all
# = make corpus  (Aozora Bunko download)
#   make minilm  (SEP fetch + similarity computation, ~50 min)
#   make figures
```

Optional robustness check with LaBSE:

```bash
make labse        # ~50 min, writes results/sep_full_similarity_labse.csv
```

Chunk-level supplementary analysis (for §4.2 of the paper):

```bash
make chunk        # writes results/chunk_analysis/
```

## Method summary

- **Japanese corpus** (6 works, ~460,000 chars total): 西田幾多郎『善の研究』(1911) ・「絶対矛盾的自己同一」(1939) ／ 九鬼周造『「いき」の構造』(1930) ／ 三木清「語られざる哲学」(1919) ・『哲学入門』(1940) ・『人生論ノート』(1941). All from Aozora Bunko.
- **SEP corpus**: 459 philosopher entries identified by Wikidata SPARQL query (P3123 × P31=Q5). After filtering articles <1,000 chars (excluded `neurath` only), **458 philosopher articles** were used in the similarity computation.
- **Embedding model**: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 128-token max input). Sentence-level chunking up to 120 tokens; centroid per work/article = L2-normalized mean of chunk embeddings; pairwise cosine similarity.
- **Baselines**: random (0.038), literary (0.312), SEP concept articles (0.498) — see paper §2.4.

## Data sources

- Japanese texts: [Aozora Bunko](https://www.aozora.gr.jp/) ([GitHub mirror](https://github.com/aozorabunko/aozorabunko))
- Philosopher articles: [Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/) (fetched per SEP terms of use)
- Philosopher identification: [Wikidata](https://www.wikidata.org/) (P3123 × P31=Q5 SPARQL query, see paper §2.2)

## Citation

```bibtex
@article{inutsuka_matsui_2026,
  author  = {Inutsuka, Yu and Matsui, Yusuke},
  title   = {{多言語文埋め込みによる日本哲学テキストとSEP全哲学者記事の意味的近接性の探索的計量分析}},
  journal = {Japanese Journal of Digital Humanities},
  year    = {2026}
}
```

## License

[MIT License](LICENSE) — Copyright © 2026 Yusuke Matsui.
The included Japanese primary texts originate from Aozora Bunko (public domain in Japan); SEP article text is *not* redistributed in this repository (only slugs are committed; full text is fetched and cached locally at run-time, in compliance with SEP's terms of use).
