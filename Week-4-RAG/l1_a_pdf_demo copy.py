from pathlib import Path
import PyPDF2

pdf_path = Path(__file__).parent / "test_documents" / "financial_report.pdf"

def extract_pypdf2(pdf_path: Path) -> str:
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

print(extract_pypdf2(pdf_path))