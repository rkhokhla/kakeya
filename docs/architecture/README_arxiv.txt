ASV Whitepaper — arXiv submission bundle
========================================

Files included:
- asv_whitepaper_natbib.tex     (LaTeX source using natbib + BibTeX)
- asv_refs.bib                  (BibTeX database)
- Makefile                      (optional local build)
- build.sh                      (optional local build script)
- asv_whitepaper_preview.pdf    (quick preview rendered from Markdown; not TeX-compiled)

How to compile locally (recommended):
1) Ensure TeX Live 2023 or 2025 is installed (arXiv currently supports both; 2025 is default).
2) Run:    pdflatex asv_whitepaper_natbib.tex
           bibtex asv_whitepaper_natbib
           pdflatex asv_whitepaper_natbib.tex
           pdflatex asv_whitepaper_natbib.tex
   This produces asv_whitepaper_natbib.pdf and asv_whitepaper_natbib.bbl.

arXiv tips (short):
- Prefer natbib+BibTeX for widest compatibility.
- Upload .tex, .bbl, and figures. Including .bib is fine as backup. If arXiv processing differs,
  the included .bbl ensures citations render as expected.
- TeX Live version can be chosen during submission (2023 or 2025). Verify the compiled PDF.
- Avoid loading hyperref with custom options; set options via \hypersetup to prevent clashes.

References:
- arXiv TeX/LaTeX submission help: https://info.arxiv.org/help/submit_tex.html
- arXiv blog (TeX Live 2025 available): https://blog.arxiv.org/2025/09/10/tex-live-2025-on-arxiv/
- Overleaf natbib guide: https://www.overleaf.com/learn/latex/Bibliography_management_with_natbib

