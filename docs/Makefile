# SLF Detection System Documentation Makefile
TEX_FILE = report.tex
PDF_FILE = report.pdf
BIB_FILE = report.bib

# LaTeX settings
LATEX = pdflatex
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

all: $(PDF_FILE)

$(PDF_FILE): $(TEX_FILE) $(BIB_FILE)
	@echo "Compiling SLF technical report..."
	$(LATEX) $(LATEX_FLAGS) $(TEX_FILE)
	bibtex $(TEX_FILE:.tex=)
	$(LATEX) $(LATEX_FLAGS) $(TEX_FILE)
	$(LATEX) $(LATEX_FLAGS) $(TEX_FILE)
	@echo "Report compiled: $(PDF_FILE)"

quick: $(TEX_FILE)
	$(LATEX) $(LATEX_FLAGS) $(TEX_FILE)

clean:
	rm -f *.aux *.log *.bbl *.blg *.toc *.lof *.lot *.out *.synctex.gz

view: $(PDF_FILE)
	open $(PDF_FILE)  # macOS
	# evince $(PDF_FILE)  # Linux

.PHONY: all quick clean view
