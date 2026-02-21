# We have defined our schema already in the previous notes.

# Now we will use it to store the documents.
# We use the Hybrid approach to load the documents

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import psycopg
from pgvector.psycopg import register_vector
import hashlib
from psycopg.types.json import Json

load_dotenv()

CONNECTION_STRING = "postgresql://langchain:langchain@localhost:6024/langchain"
EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the documents
loader = PyPDFLoader("test_documents/financial_report.pdf")
docs = loader.load()

# Split the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
splitted_docs = splitter.split_documents(docs)

# Store the documents in the database
for doc in splitted_docs[:5]:
    print(doc)

texts = [doc.page_content for doc in splitted_docs]

# Embed the documents
embeddings = EMBEDDINGS_MODEL.embed_documents(texts)

def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

with psycopg.connect(CONNECTION_STRING) as conn:
    register_vector(conn)
    with conn.cursor() as cur:

        for i, (doc, emb) in enumerate(zip(splitted_docs, embeddings)):

            meta = doc.metadata or {}

            cur.execute(
                """
                INSERT INTO documents (
                    content, embedding,
                    source_file, page_number, chunk_index, total_chunks, doc_hash,
                    department, access_level, created_by,
                    doc_type, chunk_type, extraction_method, extraction_confidence, chunk_length,
                    metadata
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    doc.page_content,
                    emb,
                    meta.get("source"),
                    meta.get("page"),
                    i,
                    len(splitted_docs),
                    sha256_hash(doc.page_content),
                    "finance",
                    "internal",
                    "nachiketh",
                    "pdf",
                    "text",
                    "pypdfloader",
                    0.95,
                    len(doc.page_content),
                    Json(meta),
                ),
            )

        conn.commit()