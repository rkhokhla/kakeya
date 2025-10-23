#!/usr/bin/env bash
set -e
echo "Compiling LaTeX with BibTeX..."
pdflatex -interaction=nonstopmode asv_whitepaper_natbib.tex
bibtex asv_whitepaper_natbib || true
pdflatex -interaction=nonstopmode asv_whitepaper_natbib.tex
pdflatex -interaction=nonstopmode asv_whitepaper_natbib.tex
echo "Done. Output: asv_whitepaper_natbib.pdf"
