from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader

loader = PyPDFLoader("test_documents/financial_report.pdf")

docs = loader.load()

for doc in docs:
    print(f"data type is {type(doc)}")
    print(f"page content: {doc.page_content}")
    print(f"metadata: {doc.metadata}")
    print("Full document:")
    print(doc)
    print("-" * 50)

# output:
# data type is <class 'langchain_community.document.Document'>
# page content: This is a test document.
# metadata: {'source': 'test_documents/financial_report.pdf', 'page': 1}
# --------------------------------------------------

print("Using PDFPlumberLoader:")
loader = PDFPlumberLoader("1810.04805v2.pdf")
docs = loader.load()
for doc in docs:
    print(f"data type is {type(doc)}")
    print(f"page content: {doc.page_content}")
    print(f"metadata: {doc.metadata}")
    print("Full document:")
    print(doc)
    print("-" * 50)