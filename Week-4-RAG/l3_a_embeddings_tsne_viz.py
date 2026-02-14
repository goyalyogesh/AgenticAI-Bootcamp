import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from openai import OpenAI

# Optional: load .env for OPENAI_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SENTENCES = [
    # Animals
    "The cat sat on the mat",
    "The dog barked at the mailman",
    "The lion roared in the jungle",
    # Technology
    "The computer crashed again",
    "The software update failed",
    "The server went offline",
    # Finance
    "The stock market crashed",
    "Revenue increased 20%",
    "Quarterly earnings exceeded expectations",
    # Food
    "The pizza was delicious",
    "I love eating sushi",
    "The restaurant served amazing pasta",
]


def main():
    print("=" * 60)
    print("DEMO: Embeddings → t-SNE 2D Visualization")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Set OPENAI_API_KEY (e.g. in .env or export).")
        return

    client = OpenAI()
    print(f"\nEmbedding {len(SENTENCES)} sentences with text-embedding-3-small...")

    embeddings = []
    for sentence in SENTENCES:
        response = client.embeddings.create(
            model="text-embedding-3-small", # 1536 dimensions
            input=sentence,
        )
        embeddings.append(response.data[0].embedding)

    embeddings = np.array(embeddings)  # shape: (12, 1536)
    print(f"  Shape: {embeddings.shape}")

    print("Reducing 1536D → 2D with t-SNE...")
    n_samples = embeddings.shape[0]
    perplexity = min(30, max(2, n_samples - 1))  # must be < n_samples
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    print("Plotting...")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=80, alpha=0.7)

    for i, sentence in enumerate(SENTENCES):
        label = sentence[:25] + ("..." if len(sentence) > 25 else "")
        ax.annotate(
            label,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )

    ax.set_title("Sentence Embeddings in 2D (t-SNE)\nSimilar topics cluster together")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    out_path = Path(__file__).resolve().parent / "embeddings_tsne_plot.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"  Saved: {out_path}")

    plt.show()
    print("\n✅ Animals, tech, finance, food should form loose clusters in the plot.")
    print("=" * 60)


if __name__ == "__main__":
    main()
