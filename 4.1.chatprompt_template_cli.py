from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse

load_dotenv()

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
        Respond with format of: 
        Topic Title: <title>
        Content Outline: <List of bullet points>
        """)
    ]
)
# first step - build the prompt with variables
# second step - invoke the LLM with build prompt

#chaining is combining multiple steps into a single chain
chain = prompt | llm_google # create a chain by piping prompt to llm (LCEL syntax)

variables = {
    "topic" : parsed_args.topic,
    "audience": parsed_args.audience
}

response = chain.invoke(variables)
print("Response from Gemini Model:")
print("Response Content:")
print(response.content)
print("********************Gemini Response end******************************")

#python 4.1.chatprompt_template_cli.py --topic "Agentic AI Bootcamp" --audience "Aspiring AI professionals"