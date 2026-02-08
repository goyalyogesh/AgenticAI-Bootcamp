
from pathlib import Path

# Inline sample (or read from test_documents/02_legal_contract.txt if present)
SAMPLE = """Section 7.3 - Liability and Damages

The parties agree that in the event of a breach of Section 7.3, the liable party shall pay damages not exceeding $500,000. This limitation does not apply to breaches involving gross negligence or willful misconduct as defined in Section 2.4.

Notwithstanding the foregoing, neither party shall be liable for indirect, incidental, special, or consequential damages arising from the performance or non-performance of this agreement."""


def get_text():
    p = Path(__file__).resolve().parent / "test_documents" / "02_legal_contract.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return SAMPLE


def main():
    print("=" * 60)
    print("DEMO: Naive Chunking (Character-Based)")
    print("=" * 60)

    text = get_text()
    print("\nOriginal Text:")
    print("-" * 60)
    print(text)
    print("-" * 60)

    chunk_size = 100
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    print(f"\nNaive Chunking (chunk_size={chunk_size}, no overlap):")
    print("-" * 60)
    for idx, chunk in enumerate(chunks, 1):
        print(f"\nChunk {idx}:")
        print(f"'{chunk}'")

    print("\n❌ Problems:")
    print("  - Chunks can end mid-word ('pay da' → 'mages')")
    print("  - No context at boundaries; meaning lost")
    print("  - Bad for retrieval: LLM sees fragments")
    print("=" * 60)


if __name__ == "__main__":
    main()
