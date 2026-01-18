from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# define state

class SimpleState(TypedDict):
    city: str
    weather: str
    temperature: float
    booking: str

# Node
def check_weather_node(state):
    return {"weather" : "rainy"}

def check_temperature_node(state):
    return {"temperature": 20.0}

def book_flight_node(state):
    print("before booking: ", state)
    print("Entered book_flight_node")
    return {"booking": "confirmed"}

# Build Graph
workflow = StateGraph(SimpleState)
workflow.add_node("check_weather", check_weather_node)
workflow.add_node("check_temperature", check_temperature_node)
workflow.add_node("book_flight", book_flight_node)

workflow.add_edge(START,"check_weather")
workflow.add_edge("check_weather", "check_temperature")
workflow.add_edge("check_temperature", "book_flight")
workflow.add_edge("book_flight", END)

# compile
app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="simplegraph_v2.png")
# run
# result = app.invoke({"city": "Mumbai"})
result = app.invoke({"city": "Mumbai", "booking": "un_confirmed", "temperature": "50.0"})
print("*************************************************")
print("Result: ", result)
print("*************************************************")

# if the field is not present in the state, it will not be updated

# StateGraph is an immutable object
# Always return a new state object