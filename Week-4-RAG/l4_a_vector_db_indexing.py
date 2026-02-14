from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "rag_production"
loader = PyPDFLoader("test_documents/financial_report.pdf")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

splitted_docs = splitter.split_documents(docs)

for s,n in zip(splitted_docs, range(len(splitted_docs))):
    print(f"Chunk {n+1}: {s.page_content[:50]}...")
    print(f"Metadata: {s.metadata}")
    print("-" * 50)

# Indexing the documents into the vector database

vector_store = PGVector.from_documents(
    documents = splitted_docs,
    embedding =embeddings_model, # not embeddings
    collection_name=collection_name,
    connection=connection,
)

# Querying the vector database
query = "What is Q4 revenue?"
# results = vector_store.similarity_search(query, k=10)
results = vector_store.similarity_search_with_score(query, k=4)

# for r in results:
#     #print(f"Score: {r.score}")
#     print(f"Page content: {r.page_content}") # chunk of the document
#     print(f"Metadata: {r.metadata}") # metadata of the chunk
#     print("-" * 50)

for r, s in results:
    print(f"Score: {s}") # distance between the query and the chunk
    print(f"Page content: {r.page_content}") # chunk of the document
    print("-" * 50)