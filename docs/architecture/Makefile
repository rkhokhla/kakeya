all: pdf

pdf:
	pdflatex -interaction=nonstopmode asv_whitepaper_natbib.tex
	bibtex asv_whitepaper_natbib || true
	pdflatex -interaction=nonstopmode asv_whitepaper_natbib.tex
	pdflatex -interaction=nonstopmode asv_whitepaper_natbib.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk

arxiv:
	# Creates an arXiv-ready zip with .tex, .bbl (if present), and .bib as backup
	zip -r asv_arxiv.zip asv_whitepaper_natbib.tex asv_refs.bib *.bbl

