import os
from pypdf import PdfReader
from pathlib import Path

# Get script's directory, then navigate to docs
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
docs_dir = PROJECT_ROOT / "pdfs"

# Only these 3 PDFs for now
test_pdfs = [
    "swm_2016.pdf",
    "urdpfi_vol1.pdf",
    "urdpfi_vol2.pdf"
]

print("📚 Documents for initial testing:\n")

total_pages = 0
for pdf_name in test_pdfs:
    pdf_path = os.path.join(docs_dir, pdf_name)

    if not os.path.exists(pdf_path):
        print(f"  ❌ {pdf_name}: NOT FOUND")
        continue

    reader = PdfReader(pdf_path)
    pages = len(reader.pages)
    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

    print(f"  ✅ {pdf_name}")
    print(f"     Pages: {pages}")
    print(f"     Size: {size_mb:.1f} MB\n")

    total_pages += pages

print(f"📊 Total pages to embed: {total_pages}")
print(f"⏱️  Estimated embedding time (CPU): {total_pages * 15 / 3600:.1f} - {total_pages * 20 / 3600:.1f} hours")
