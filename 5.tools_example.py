from langchain.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()
llm_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=1) #initializing the openai model

@tool
def get_weather(city: str) -> str:
    """Get weather for a city"""
    return f"Sunny, 25°C in {city}"

@tool
def book_flight(city: str) -> dict:
    """Book a flight to a city"""
    return {"booking_id": "FL123", "destination": city}

# Create tools list
tools = [get_weather, book_flight]

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Be friendly and concise."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")  # Tool results go here
])

# Create the agent
agent = create_tool_calling_agent(llm_openai, tools, prompt)

# Create the executor (runs the loop)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,        # Show what's happening
    max_iterations=5,    # Safety limit
    handle_parsing_errors=True
)

result = agent_executor.invoke({
    "input": "What's the weather in Bangalore? If it's good, book me a flight."
})

print(result["output"])