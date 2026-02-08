"""
Demo 1: RAG Fundamentals - Why We Need RAG
Simple script to demonstrate the hallucination problem and why RAG solves it.
Uses LangChain for LLM interactions and RAG chains.
All required data is in this file or read from test_documents/ (no demo_docs dependency).
"""

from pathlib import Path
import sys

CLASS_DEMO_DIR = Path(__file__).resolve().parent
TEST_DOCS_DIR = CLASS_DEMO_DIR / "test_documents"
if str(CLASS_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(CLASS_DEMO_DIR))

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# In-file: RAG context (read from file or use minimal fallback)
RAG_CONTEXT_FALLBACK = """Fiscal Year 2024 Financial Report

Revenue Growth
Quarter  |  Revenue
Q1       |  $2.1M
Q2       |  $2.8M
Q3       |  $3.4M
Q4       |  $4.2M

Key Metrics: Customer Acquisition 78%, Annual Growth 23%, Customer Retention 92%.

Executive Summary: This report covers the financial performance of QuantumFlux Industries for fiscal year 2024. Q4 revenue reached $4.2 million, representing a 23% increase from Q3."""


def _get_rag_context() -> str:
    """Load RAG context from test_documents/01_financial_report.txt or use inline fallback."""
    path = TEST_DOCS_DIR / "01_financial_report.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return RAG_CONTEXT_FALLBACK


def demonstrate_hallucination():
    """Shows how LLMs hallucinate when they don't have access to specific data."""
    
    print("=" * 60)
    print("DEMO: The Hallucination Problem")
    print("=" * 60)
    
    query = "What was QuantumFlux Industries' revenue in Q4 2024?"
    print(f"\nQuery: {query}")
    print("\nLLM Response (without RAG):")
    print("-" * 60)
    
    try:
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{query}")
        ])
        chain = prompt | llm
        response = chain.invoke({"query": query})
        print(response.content)
    except Exception as e:
        if "api_key" in str(e).lower() or "OPENAI" in str(e).upper():
            print("[Set OPENAI_API_KEY for live LLM response. Without RAG, LLM would generate plausible but false facts about QuantumFlux Industries.]")
        else:
            raise
    
    print("\n Problem: LLM can confidently either generate information about a company that doesn't exist or  generate saying that it doesn't know the answer.")
    return None


def demonstrate_rag_solution():
    """Shows how RAG grounds the LLM with actual documents using LangChain."""
    
    print("\n" + "=" * 60)
    print("DEMO: RAG Solution - Grounding with Documents")
    print("=" * 60)
    
    document_context = _get_rag_context()
    query = "What was QuantumFlux Industries' revenue in Q4 2024?"
    
    print(f"\nQuery: {query}")
    print("\nRetrieved Document Context:")
    print("-" * 60)
    print(document_context)
    print("-" * 60)
    
    print("\nLLM Response (with RAG):")
    print("-" * 60)
    try:
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers based on provided documents. Answer based ONLY on the document provided. If the information is not in the document, say 'I don't have that information in the provided document.'"),
            ("human", """Based on the following document, answer the question.

Document:
{document_context}

Question: {query}""")
        ])
        chain = rag_prompt | llm
        response = chain.invoke({"document_context": document_context, "query": query})
        print(response.content)
    except Exception as e:
        if "api_key" in str(e).lower() or "OPENAI" in str(e).upper():
            print("[Set OPENAI_API_KEY for live response. With RAG, LLM would answer from the document: Q4 revenue $4.2M, 23% increase.]")
        else:
            raise
    print("\n✅ Solution: LLM now answers based on actual document content!")
    print("   The answer is grounded in real data, not hallucinated.")
    print("   This is how LangChain RAG chains work - retrieve, augment, generate!")
    return None


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG FUNDAMENTALS DEMO")
    print("=" * 60)
    print("\nThis demo shows:")
    print("1. Why LLMs hallucinate without access to specific data")
    print("2. How RAG solves this by providing document context")
    print("\n" + "-" * 60 + "\n")
    
    # Demo 1: Hallucination problem
    demonstrate_hallucination()
    
    # Demo 2: RAG solution
    demonstrate_rag_solution()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY:")
    print("RAG = Retrieve relevant documents + Augment prompt + Generate answer")
    print("LangChain provides RAG chains that automate this process!")
    print("=" * 60 + "\n")
