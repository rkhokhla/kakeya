#!/usr/bin/env python3
"""
Convert markdown to beautiful academic PDF using weasyprint.
"""

import sys
import re
from pathlib import Path
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def convert_latex_math(text):
    """Convert LaTeX-style math to HTML with Unicode/styled text."""
    # Replace display math $$...$$ with styled div
    text = re.sub(r'\$\$(.*?)\$\$', r'<div class="math-block">\1</div>', text, flags=re.DOTALL)

    # Replace inline math $...$ with styled span
    text = re.sub(r'\$(.*?)\$', r'<span class="math-inline">\1</span>', text)

    # Convert common LaTeX symbols to Unicode
    latex_to_unicode = {
        r'\\alpha': 'α', r'\\beta': 'β', r'\\gamma': 'γ', r'\\delta': 'δ',
        r'\\epsilon': 'ε', r'\\zeta': 'ζ', r'\\eta': 'η', r'\\theta': 'θ',
        r'\\lambda': 'λ', r'\\mu': 'μ', r'\\sigma': 'σ', r'\\tau': 'τ',
        r'\\Sigma': 'Σ', r'\\leq': '≤', r'\\geq': '≥', r'\\approx': '≈',
        r'\\times': '×', r'\\cdot': '·', r'\\pm': '±', r'\\infty': '∞',
        r'\\in': '∈', r'\\subset': '⊂', r'\\subseteq': '⊆', r'\\cap': '∩',
        r'\\cup': '∪', r'\\emptyset': '∅', r'\\forall': '∀', r'\\exists': '∃',
        r'\\neg': '¬', r'\\wedge': '∧', r'\\vee': '∨', r'\\rightarrow': '→',
        r'\\Rightarrow': '⇒', r'\\leftrightarrow': '↔', r'\\equiv': '≡',
        r'\\neq': '≠', r'\\sim': '∼', r'\\propto': '∝',
    }

    for latex, unicode_char in latex_to_unicode.items():
        text = text.replace(latex, unicode_char)

    # Handle subscripts and superscripts
    text = re.sub(r'_\{([^}]+)\}', lambda m: ''.join(chr(0x2080 + int(c)) if c.isdigit() else c for c in m.group(1)), text)
    text = re.sub(r'\^\{([^}]+)\}', lambda m: m.group(1), text)  # Keep superscripts as-is for now

    return text

def create_html_template(md_content, title="Fractal LBA: Formal Verification for LLM Output Quality"):
    """Create full HTML document with styling."""

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=[
        'extra',  # tables, fenced code, etc.
        'codehilite',  # syntax highlighting
        'toc',  # table of contents
        'footnotes',
        'attr_list',
    ])

    # Convert LaTeX math before markdown processing
    md_content = convert_latex_math(md_content)

    html_content = md.convert(md_content)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        @page {{
            size: letter;
            margin: 1in;
            @bottom-center {{
                content: counter(page);
                font-family: 'Times New Roman', serif;
                font-size: 10pt;
            }}
        }}

        body {{
            font-family: 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #000;
            max-width: 7.5in;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 18pt;
            font-weight: bold;
            text-align: center;
            margin: 24pt 0 12pt 0;
            page-break-after: avoid;
        }}

        h2 {{
            font-size: 14pt;
            font-weight: bold;
            margin: 18pt 0 12pt 0;
            page-break-after: avoid;
        }}

        h3 {{
            font-size: 12pt;
            font-weight: bold;
            margin: 14pt 0 8pt 0;
            page-break-after: avoid;
        }}

        h4 {{
            font-size: 11pt;
            font-weight: bold;
            font-style: italic;
            margin: 12pt 0 6pt 0;
            page-break-after: avoid;
        }}

        p {{
            margin: 0 0 8pt 0;
            text-align: justify;
            text-indent: 0;
        }}

        /* First paragraph after heading should not be indented */
        h1 + p, h2 + p, h3 + p, h4 + p {{
            text-indent: 0;
        }}

        /* Abstract styling */
        blockquote {{
            margin: 12pt 0.5in;
            padding: 12pt;
            background: #f9f9f9;
            border-left: 3pt solid #333;
            font-size: 10pt;
        }}

        /* Math blocks */
        .math-block {{
            margin: 12pt 0.25in;
            padding: 8pt;
            background: #f5f5f5;
            border-left: 2pt solid #999;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            line-height: 1.4;
            white-space: pre-wrap;
            overflow-x: auto;
        }}

        .math-inline {{
            font-family: 'Times New Roman', serif;
            font-style: italic;
            background: #f9f9f9;
            padding: 1pt 3pt;
            border-radius: 2pt;
        }}

        /* Code blocks */
        pre {{
            margin: 12pt 0.25in;
            padding: 8pt;
            background: #f5f5f5;
            border: 1pt solid #ccc;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            line-height: 1.4;
            overflow-x: auto;
            page-break-inside: avoid;
        }}

        code {{
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            background: #f5f5f5;
            padding: 1pt 3pt;
        }}

        /* Tables */
        table {{
            margin: 12pt auto;
            border-collapse: collapse;
            width: 90%;
            page-break-inside: avoid;
        }}

        th {{
            background: #333;
            color: white;
            padding: 6pt 8pt;
            text-align: left;
            font-weight: bold;
            border: 1pt solid #000;
        }}

        td {{
            padding: 4pt 8pt;
            border: 1pt solid #999;
        }}

        tr:nth-child(even) {{
            background: #f9f9f9;
        }}

        /* Lists */
        ul, ol {{
            margin: 8pt 0;
            padding-left: 0.3in;
        }}

        li {{
            margin: 4pt 0;
        }}

        /* Strong emphasis */
        strong {{
            font-weight: bold;
        }}

        em {{
            font-style: italic;
        }}

        /* Links */
        a {{
            color: #0066cc;
            text-decoration: none;
        }}

        /* Horizontal rules */
        hr {{
            border: none;
            border-top: 1pt solid #333;
            margin: 24pt 0;
        }}

        /* Theorem/proof boxes */
        .theorem, .proof {{
            margin: 12pt 0;
            padding: 8pt;
            border: 1pt solid #333;
            page-break-inside: avoid;
        }}

        .theorem {{
            background: #f0f0ff;
        }}

        .proof {{
            background: #fff5f0;
        }}

        /* Caption styles */
        .caption {{
            font-size: 9pt;
            text-align: center;
            margin: 6pt 0;
            font-style: italic;
        }}

        /* Page breaks */
        .page-break {{
            page-break-before: always;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""

    return html

def main():
    if len(sys.argv) != 3:
        print("Usage: python md_to_pdf.py input.md output.pdf")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    print(f"Reading {input_file}...")
    md_content = input_file.read_text(encoding='utf-8')

    print("Converting markdown to HTML...")
    html = create_html_template(md_content)

    print("Generating PDF...")
    font_config = FontConfiguration()

    HTML(string=html).write_pdf(
        output_file,
        font_config=font_config,
    )

    print(f"✓ PDF created: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")

if __name__ == '__main__':
    main()
