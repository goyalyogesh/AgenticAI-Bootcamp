from pathlib import Path
import PyPDF2
import pdfplumber
import re

pdf_path = Path(__file__).parent / "test_documents" / "1810.04805v2.pdf"

def extract_pypdf2(pdf_path: Path) -> str:
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
# print("PyPDF2 extraction:")
# print(extract_pypdf2(pdf_path))
# print("--------------------------------")


# 50 ms per document as latency

# def extract_pdfplumber(pdf_path: Path) -> str:
#     quality_score = 1.0
#     with open(pdf_path, "rb") as f:
#         pdf = pdfplumber.open(f)
#         text = ""
#         for page in pdf.pages:
#             text += page.extract_text()

#         non_ascii_ratio = sum(1 for c in text if ord( c) > 127 )/ len(text)

#         if len(text) == 0:
#             #return "No text found"
#             quality_score -= 0.3
#         if len(text) < 100:
#             #return "Text is too short"
#             quality_score -= 0.3
#         #return text

#         if non_ascii_ratio > 0.3:
#             #return "Text is mostly non-ascii"
#             quality_score -= 0.2
#         #return text

#         #check for gibberish
#         gibberish_pattern = r'[^a-zA-Z\s]{10,}'
#         gibberish_matches = len(re.findall(gibberish_pattern, text))
#         if gibberish_matches > 5:
#             #return "Text is mostly gibberish"
#             quality_score -= 0.2
#         #return text

#         if quality_score < 0.3:
#             pass
#         # pass_to_human_review = True
#         return text

def extract_pdfplumber(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        t = ""
        for page in pdf.pages:
            t += page.extract_text(use_text_flow=True)
        return t


if __name__ == "__main__":
    print("PyPDF2 extraction:")
    print(extract_pypdf2(pdf_path))
    print("--------------------------------")
    print("pdfplumber extraction:")
    print(extract_pdfplumber(pdf_path))
    print("--------------------------------")
