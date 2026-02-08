from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=0.6)

query_message = "What was the QuantumFlux Industries revenue in Q4 2025?"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions about the following text:"),
    ("user", "{query}")
])

prompt = prompt_template.invoke({
    "query": query_message
})

llm_response = llm_openai.invoke(prompt)
print(llm_response.content)