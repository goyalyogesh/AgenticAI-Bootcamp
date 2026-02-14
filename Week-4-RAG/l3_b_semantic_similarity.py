
from pathlib import Path
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv  

load_dotenv()
SAMPLE_SENTENCES = [
    "The cat sat on the mat.",
    "The feline rested on the rug.",
    "I love eating pizza.",
    "The dog barked at the mailman.",
]

query = "Where is the cat?"

def main():
    print("=" * 60)
    print("DEMO: Semantic Similarity")
    print("=" * 60)

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = embeddings_model.embed_documents(SAMPLE_SENTENCES)
    query_embedding = embeddings_model.embed_query(query)

    embeddings = np.array(embeddings)
    query_embedding = np.array(query_embedding)
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("\nCosine Similarity Matrix:")
    print("-" * 60)
    n = len(SAMPLE_SENTENCES)
    header = "".join(f" | Sent {i+1}" for i in range(n))
    print(f"{'':<30}{header}")
    print("-" * 60)
    for i, sent1 in enumerate(SAMPLE_SENTENCES):
        row = f"{sent1[:28]:<28} |"
        for j in range(n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            row += f"  {sim:.3f}  |"
        print(row)
    print("-" * 60)
    print("Similar sentences should have high similarity scores.")
    print("Query: ", query)
    print(f"Similarity between query and first sentence: {cosine_similarity(query_embedding, embeddings[0])}")
    print(f"Similarity between query and second sentence: {cosine_similarity(query_embedding, embeddings[1])}")
    print(f"Similarity between query and third sentence: {cosine_similarity(query_embedding, embeddings[2])}")
    print(f"Similarity between query and fourth sentence: {cosine_similarity(query_embedding, embeddings[3])}") 

if __name__ == "__main__":
    main()