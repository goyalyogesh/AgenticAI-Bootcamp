from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

SAMPLE = """Section 7.3 - Liability and Damages

The parties agree that in the event of a breach of Section 7.3, the liable party shall pay damages not exceeding $500,000. This limitation does not apply to breaches involving gross negligence or willful misconduct as defined in Section 2.4."""
def get_text():
    p = Path(__file__).resolve().parent / "test_documents" / "03_aws_ec2_pricing.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return SAMPLE

def main():
    text = get_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, # max number of characters in a chunk
        chunk_overlap=50, # number of characters to overlap between chunks
        separators=["\n\n", "\n", ". ", " ", ""], # order of the separators is important
    )

    chunks = splitter.split_text(text)

    for i, ch in enumerate(chunks, 1):
        print(f"\nChunk {i} (len={len(ch)}):\n{ch!r}")

    # quick overlap sanity-check (only if we have 2+ chunks)
    if len(chunks) >= 2:
        print("\n--- overlap check ---")
        print("chunk1:", chunks[0])
        print("chunk2:", chunks[1])

if __name__ == "__main__":
    main()
