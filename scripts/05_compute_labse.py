#!/usr/bin/env python3
"""
LaBSE robustness check
=======================
Re-runs the SEP-vs-Japanese-philosophy similarity computation using
Google's LaBSE (Language-agnostic BERT Sentence Embedding) model
in place of paraphrase-multilingual-MiniLM-L12-v2.

LaBSE produces 768-dim embeddings (vs 384 for MiniLM) and was trained
specifically for cross-lingual semantic alignment, so it serves as
an independent check on whether the rankings reported in the paper
are an artefact of MiniLM's specific training objective.

Outputs:
    results/sep_full_similarity_labse.csv
    results/sep_full_top50_labse.csv

Runtime: ~50min on CPU, ~10min on a single GPU.

Usage:
    python scripts/05_compute_labse.py
"""

import os
import runpy
from pathlib import Path

os.environ["V5_MODEL"] = "sentence-transformers/LaBSE"
os.environ["V5_MODEL_TAG"] = "labse"

THIS_DIR = Path(__file__).resolve().parent
runpy.run_path(str(THIS_DIR / "02_compute_minilm.py"), run_name="__main__")
