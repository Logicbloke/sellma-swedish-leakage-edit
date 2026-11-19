"""Render report/note.md to report/note.pdf (Markdown -> HTML -> WeasyPrint). Figures resolve relative to report/."""
import pathlib, markdown, weasyprint

HERE = pathlib.Path(__file__).resolve().parent
CSS = """
@page { size: A4; margin: 18mm 17mm; @bottom-center { content: counter(page); font-size: 8pt; color: #777; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.5pt; line-height: 1.4; color: #111; }
h1 { font-size: 15pt; margin: 0 0 6pt; } h2 { font-size: 11.5pt; margin: 12pt 0 4pt; border-bottom: 1px solid #ddd; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 8.5pt; } th, td { border: 1px solid #ccc; padding: 2pt 5pt; }
th { background: #f3f3f1; } code { font-size: 8.5pt; background: #f4f4f2; padding: 0 2pt; }
img { max-width: 100%; display: block; margin: 6pt auto; } p, li { margin: 3pt 0; } hr { border: 0; border-top: 1px solid #ddd; }
"""
md = (HERE / "note.md").read_text(encoding="utf-8")
html = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
doc = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
weasyprint.HTML(string=doc, base_url=str(HERE)).write_pdf(HERE / "note.pdf")
print("wrote", HERE / "note.pdf")
