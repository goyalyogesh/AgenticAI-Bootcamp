from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
from langchain_core.indexing.base import InMemoryRecordManager # in Production we use PostgreSQLRecordManager, SQLiteRecordManager
from langchain_core.indexing import index
load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "rag_production"
namespace = "langchain_rag"
record_manager = InMemoryRecordManager(namespace=namespace)
record_manager.create_schema()

cats_loader = TextLoader("test_documents/cat.txt")
dogs_loader = TextLoader("test_documents/dog.txt")
dogs_loader_v2 = TextLoader("test_documents/dog_v2.txt")

cats_docs = cats_loader.load()
dogs_docs = dogs_loader.load()
dogs_docs_v2 = dogs_loader_v2.load()


# Indexing the documents into the vector database

vector_store = PGVector(
    embeddings =embeddings_model, 
    collection_name=collection_name,
    connection=connection,
)

# index
index_1 = index(cats_docs, record_manager, vector_store, source_id_key="source")
print("Indexing the cats documents", index_1)

index_2 = index(dogs_docs, record_manager, vector_store, source_id_key="source")
print("Indexing the dogs documents", index_2)

index_3 = index(dogs_docs_v2, record_manager, vector_store, source_id_key="source")
print("Indexing the dogs documents v2", index_3)

# Repeated indexing
index_4 = index(cats_docs, record_manager, vector_store, source_id_key="source")
print("Indexing the cats documents again", index_4)

# Modify the content of cats.txt
with open("test_documents/cat.txt", "w") as f:
    f.write("I love tom and jerry")

# re-index
cats_loader_v2 = TextLoader("test_documents/cat.txt")
cats_docs_v2 = cats_loader_v2.load()
index_5 = index(cats_docs_v2, record_manager, vector_store, source_id_key="source")
print("Indexing the cats documents v2", index_5)
