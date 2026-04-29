# aozora-sep — reproduction Makefile
#
# Quick start (uses bundled results/ and figures/):
#   make figures      ← regenerate figures from results/sep_person_similarity.csv
#
# Full re-run (downloads SEP, embeds with MiniLM ~50 min CPU):
#   make all
#
# Optional:
#   make labse        ← LaBSE robustness check (~50 min)
#   make chunk        ← chunk-level supplementary analysis

PYTHON ?= python3
SCRIPT_DIR = scripts

.PHONY: all corpus minilm figures labse chunk clean help

help:
	@echo "Targets:"
	@echo "  make figures  : regenerate figures/ from results/sep_person_similarity.csv"
	@echo "  make corpus   : download Japanese primary texts from Aozora Bunko"
	@echo "  make minilm   : compute SEP × JP similarity with MiniLM (~50 min)"
	@echo "  make labse    : LaBSE robustness recompute (~50 min)"
	@echo "  make chunk    : chunk-level supplementary analysis"
	@echo "  make all      : corpus → minilm → figures (full pipeline)"
	@echo "  make clean    : remove caches and intermediate outputs"

corpus:
	$(PYTHON) $(SCRIPT_DIR)/01_build_jp_corpus.py

minilm:
	$(PYTHON) $(SCRIPT_DIR)/02_compute_minilm.py

figures:
	$(PYTHON) $(SCRIPT_DIR)/03_generate_figures.py

labse:
	$(PYTHON) $(SCRIPT_DIR)/05_compute_labse.py

chunk:
	$(PYTHON) $(SCRIPT_DIR)/06_chunk_level_analysis.py

all: corpus minilm figures

clean:
	rm -rf .cache corpus
