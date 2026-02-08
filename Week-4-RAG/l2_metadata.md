# Minimum metadata

```python
chunk_metadata ={
    # source tracking
    "source_file":"Q4_report.pdf",
    "page_number": 12,
    "chunk_index": 5,
    "total_chunks": 40, # document has 40 chunks in total

    #content ype
    "chunk_type": "text",
    "clause_number": "7.2",

    #Quality
    "ocr_confidence": 0.85,
    "chunk_length": 250,

    # Timestamps
    "created_at": "2025-01-26T20:32:10",
    "doc_hash": "a3safsf.." # for change detection
}

access_metadata = {
    "department": "finance",
    "access_level" : "confidential", # public
}

```