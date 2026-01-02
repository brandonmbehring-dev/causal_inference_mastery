# Makefile for Causal Inference Mastery - Volume 1 (Native LaTeX)
#
# Build professional PDFs with native LaTeX equation rendering

.PHONY: all clean help view install-deps check-errors part1 part2

# Main targets
MAIN = main
PDF = $(MAIN).pdf

# LaTeX compilation commands
LATEX = pdflatex -shell-escape -interaction=nonstopmode -file-line-error
BIBTEX = bibtex

# Chapter files
PART1_CHAPTERS = chapters/part1_foundations/ch01_potential_outcomes.tex \
                 chapters/part1_foundations/ch02_randomized_experiments.tex \
                 chapters/part1_foundations/ch03_statistical_inference.tex

PART2_CHAPTERS = chapters/part2_selection_observables/ch04_propensity_score.tex \
                 chapters/part2_selection_observables/ch05_weighting.tex \
                 chapters/part2_selection_observables/ch06_cate.tex \
                 chapters/part2_selection_observables/ch07_causal_forests.tex \
                 chapters/part2_selection_observables/ch08_sensitivity.tex

APPENDICES = chapters/appendices/app_julia.tex

ALL_CHAPTERS = $(PART1_CHAPTERS) $(PART2_CHAPTERS) $(APPENDICES)

# Default target: build complete book
all: $(PDF)

# Build complete book (all chapters)
$(PDF): $(MAIN).tex $(ALL_CHAPTERS) bibliography.bib
	@echo "=========================================="
	@echo "Building Causal Inference Mastery..."
	@echo "=========================================="
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	@echo ""
	@echo "Build complete: $(PDF)"
	@ls -lh $(PDF)
	@echo ""

# View PDF in default viewer
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	else \
		echo "No PDF viewer found. Please open $(PDF) manually."; \
	fi

# Install required dependencies
install-deps:
	@echo "Installing Python dependencies for minted..."
	pip install Pygments
	@echo ""
	@echo "Dependencies installed"
	@echo ""
	@echo "LaTeX packages required (install via TeX distribution):"
	@echo "  - amsbook (usually included)"
	@echo "  - minted"
	@echo "  - hyperref"
	@echo "  - booktabs"
	@echo ""

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).bbl $(MAIN).blg
	rm -f $(MAIN).synctex.gz $(MAIN).fdb_latexmk $(MAIN).fls
	rm -f chapters/*/*.aux chapters/appendices/*.aux
	rm -rf _minted-$(MAIN)/
	@echo "Cleaned"

# Clean everything including PDF
distclean: clean
	@echo "Removing PDF..."
	rm -f $(PDF)
	@echo "All generated files removed"

# Check for LaTeX errors in log
check-errors:
	@if [ -f $(MAIN).log ]; then \
		echo "Checking for errors in $(MAIN).log..."; \
		if grep -E "^!" $(MAIN).log; then \
			echo ""; \
			echo "Errors found in LaTeX compilation"; \
			exit 1; \
		else \
			echo "No errors found"; \
		fi; \
		echo ""; \
		echo "Checking for warnings..."; \
		if grep -i "warning" $(MAIN).log | grep -v "Font shape" | head -10; then \
			echo ""; \
			echo "Warnings found (review log)"; \
		else \
			echo "No critical warnings found"; \
		fi; \
	else \
		echo "No log file found. Run 'make' first."; \
	fi

# Help message
help:
	@echo "Causal Inference Mastery - Volume 1 - LaTeX Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all            Build complete book (default)"
	@echo "  view           Build and open PDF in viewer"
	@echo "  install-deps   Install Python dependencies (Pygments)"
	@echo "  clean          Remove auxiliary files (keep PDF)"
	@echo "  distclean      Remove all generated files including PDF"
	@echo "  check-errors   Check LaTeX log for errors/warnings"
	@echo "  help           Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex (TeX Live or MiKTeX)"
	@echo "  - bibtex"
	@echo "  - Python + Pygments (for minted code highlighting)"
	@echo ""
	@echo "Example usage:"
	@echo "  make               # Build complete book"
	@echo "  make view          # Build and open PDF"
	@echo "  make clean         # Clean auxiliary files"
	@echo ""
	@echo "Compilation process:"
	@echo "  1. pdflatex (first pass)"
	@echo "  2. bibtex (process bibliography)"
	@echo "  3. pdflatex (second pass, resolve citations)"
	@echo "  4. pdflatex (third pass, resolve cross-references)"
	@echo ""
