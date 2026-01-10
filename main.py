from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

m1 = "Who is the President of India?"

response_openai  = llm_openai.invoke(m1)
response_google = llm_google.invoke(m1)

print("Question1: ", m1)

print("OpenAI Response: ", response_openai.content)
print("Google Response: ", response_google.content)

m2 = "what is his age ?"

response_openai_m2  = llm_openai.invoke(m2)
response_google_m2 = llm_google.invoke(m2)

print("Question2: ", m2)

print("OpenAI Response: ", response_openai_m2.content)
print("Google Response: ", response_google_m2.content)