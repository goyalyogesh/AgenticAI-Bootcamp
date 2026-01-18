from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import operator
from dotenv import load_dotenv
load_dotenv()

# Tools
@tool
def check_order_status(order_id: str) -> dict:
    """Check the status of an order."""
    return {"order_id": order_id, "status": "shipped", "eta": "2024-01-20"}

@tool
def create_ticket(issue: str, priority: str) -> dict:
    """Create a support ticket."""
    return {"ticket_id": "TKT12345", "issue": issue, "priority": priority}

# State
class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add] # Annotated to indicate messages can be appended and modified by the agent
    should_escalate: bool
    issue_type: str

# Setup
tools = [check_order_status, create_ticket]
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Nodes
def agent_node(state: SupportState):
    """Agent node that processes messages and decides on tool usage."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: SupportState) -> Literal["continue", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"

# Build graph
workflow = StateGraph(SupportState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")
# should_continue is not a node, it is a function that is used to determine the next node to execute

# Compile
app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="support_agent_with_tools.png")
# Run
result = app.invoke({
    "messages": [HumanMessage(content="Check order ORD123 status")],
    "should_escalate": False,
    "issue_type": ""
})

print("\n" + "="*50)
for msg in result["messages"]:
    if hasattr(msg, 'content'):
        print(f"{msg.type}: {msg.content}")
# create a ticket for the issue
print("*************************************************")
print("Result: ", result)
print("*************************************************")

result = app.invoke({
    "messages": [HumanMessage(content="Create a ticket for the issue: The order is not delivered on time")],
    "should_escalate": False,
    "issue_type": "delivery_issue"
})

print("\n" + "="*50)
for msg in result["messages"]:
    if hasattr(msg, 'content'):
        print(f"{msg.type}: {msg.content}")
print("*************************************************")
print("Result: ", result)
print("*************************************************")