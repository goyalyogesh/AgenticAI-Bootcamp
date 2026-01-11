from langchain.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city
    """
    # Mock implementation
    weather_data = {
        "bangalore": "Sunny, 28°C",
        "mumbai": "Rainy, 26°C",
        "delhi": "Cloudy, 22°C"
    }
    return weather_data.get(city.lower(), "Weather data not available")


print(get_weather.run("bangalore"))


@tool
def book_flight(origin: str, destination: str, date: str) -> dict:
    """Book a flight from one city to another.

    Args:
        origin: The origin city
        destination: The destination city
        date: The date of the flight
    """
    # Mock implementation
    return {
        "booking_id": "1234567890",
        "route": f"{origin} to {destination}",
        "date": date,
        "status": "confirmed"
    }


# list of tools
tools = [get_weather, book_flight]

llm_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=1).bind_tools(tools)
print("*************************************************")
response = llm_openai.invoke("What is the weather in Bangalore?")
print(response)
print("*************************************************")
response = llm_openai.invoke("Book a flight from Bangalore to Mumbai on 12/01/2026")
print(response)