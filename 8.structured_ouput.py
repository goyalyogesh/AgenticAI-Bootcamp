#example for structured output
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=0.6)

class SmartResponse(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    email: str = Field(description="The email of the person")
    phone: str = Field(description="The phone number of the person")

#example for structured output
response = llm_openai.invoke("Can you generate some random data for a person?")
print("Unstructured response:")
print(response.content)
print("--------------------------------")
response = llm_openai.with_structured_output(SmartResponse).invoke("Can you generate some random da ta for a person?")
print("Structured response:")
print(response)
print("--------------------------------")