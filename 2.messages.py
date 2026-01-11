from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=1) #initializing the openai model
# Temperature - controls the randomness of the model's output
# max temperature - 1.0 (most random)
# min temperature - 0.0 (least random)
# max tokens - controls the maximum number of tokens in the response
# min tokens - controls the minimum number of tokens in the response
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1) #initializing the google model

m1 = "Who is the Prime Minister of India?" #first question

messages = [
    SystemMessage(content ="You are a funny assistant that can answer questions and make jokes."),
    HumanMessage(content = m1)]

messages_2 = [
    SystemMessage(content ="You are a funny assistant that can answer questions and make jokes."),
    HumanMessage(content = "Tell me about the Volcano in Indonesia")]

response = llm_openai.invoke(messages)
#display the type
print(type(response))
#display the content
print(response.content)


response_2 = llm_openai.invoke(messages_2)
#display the type
print(type(response_2))
#display the content
print(response_2.content)

# Temperature as high when the outcome you want is more random, and dynamic and creative.
# Temperature as low when the outcome you want is more predictable, and consistent.