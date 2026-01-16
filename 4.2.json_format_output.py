from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

class LearningOutline(BaseModel):
    topic_title: str = Field(description="Title of the learning topic")
    content_outline: list[str] = Field(...,description = "A list of Bullet points representing the outline of the topic")

parser = PydanticOutputParser(pydantic_object=LearningOutline)
format_instructions = parser.get_format_instructions()

llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

args = argparse.ArgumentParser(description="Generate learning outline using Gemini LLM")
args.add_argument("--topic", type=str, help="Topic of the learning outline", required=True)
args.add_argument("--audience", type=str, help="Target audience for the learning outline", required=True)

parsed_args = args.parse_args()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert educational content creator"),
        ("human", """
        Create a detailed learning outline on the topic of 
        {topic} for {audience}.
        Respond in JSON format following these instructions: 
        {format_instructions}
        Do not include any commentary or md fences, 
        adhere to JSON schema and field description strictly.
        """)
    ]
)
# first step - build the prompt with variables
# second step - invoke the LLM with build prompt

#chaining is combining multiple steps into a single chain
chain = prompt | llm_google # create a chain by piping prompt to llm (LCEL syntax)

variables = {
    "topic" : parsed_args.topic,
    "audience": parsed_args.audience,
    "format_instructions": format_instructions
}

response = chain.invoke(variables)
print("Response from Gemini Model:")
print("Response Content:")
print(response.content)
print("********************Gemini Response end******************************")

#python 4.2.json_format_output.py --topic "Agentic AI Bootcamp" --audience "Test Engineer"