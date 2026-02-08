"""
Demo 04g: Specialized chunking (code AST, legal sections, tables).
Run alone: python 04g_specialized_chunking.py
Uses only stdlib (ast, re) + optional test_documents files.
"""

import re
from pathlib import Path

LEGAL_SAMPLE = """Section 7.3 - Liability. The parties agree that damages not exceeding $500,000.
Section 2.4 - Definitions. Gross negligence means a conscious disregard of reasonable care."""

CODE_SAMPLE = '''"""Code chunking demo."""
def calculate_revenue(quarter, base_revenue):
    return base_revenue * (1.0 + quarter * 0.05)

class FinancialReport:
    def __init__(self, company_name, year):
        self.company_name = company_name
        self.year = year
'''


def get_legal_text():
    p = Path(__file__).resolve().parent / "test_documents" / "02_legal_contract.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return LEGAL_SAMPLE


def get_code_text():
    p = Path(__file__).resolve().parent / "test_documents" / "10_code_example.py"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return CODE_SAMPLE


def main():
    print("=" * 60)
    print("DEMO: Specialized Chunking")
    print("=" * 60)

    # 1) Code: function/class boundaries (AST)
    code = get_code_text()
    try:
        import ast
        tree = ast.parse(code)
        names = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.append(node.name)
        if names:
            print("\n1. Code (AST): chunk by function/class")
            for i, name in enumerate(names[:5], 1):
                print(f"   Chunk {i}: {name}")
            print("   ✅ Never split mid-function; use AST")
    except Exception:
        print("\n1. Code: use AST to chunk by function/class")

    # 2) Legal: section boundaries
    legal_text = get_legal_text()
    legal_sections = [p.strip() for p in re.split(r"\n(?=Section \d+\.\d+)", legal_text) if p.strip()]
    if not legal_sections and "Section" in legal_text:
        legal_sections = [legal_text]
    print("\n2. Legal: chunk by clause/section (Section 7.3, Section 2.4, ...)")
    print(f"   Result: {len(legal_sections)} section chunks")
    print("   ✅ Each clause/section = one chunk")

    # 3) Tables
    print("\n3. Tables: entire table = one chunk (or convert to markdown)")
    print("   Example: Keep '| Q1 | $2.1M | ...' as single chunk")
    print("   ✅ Don't split tables across chunks")

    print("\n✅ Match strategy to document type (code, legal, tables)")
    print("=" * 60)


if __name__ == "__main__":
    main()
