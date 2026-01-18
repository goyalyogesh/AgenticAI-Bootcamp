from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import operator
from dotenv import load_dotenv
load_dotenv()

# 1. Update State
class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    should_escalate: bool
    issue_type: str
    user_tier: str  # ADD THIS

# 2. Create check_tier node
def check_user_tier_node(state: SupportState):
    # YOUR CODE HERE
    # Mock: return {"user_tier": "vip"} or {"user_tier": "standard"}
    pass

# 3. Create routing function
def route_by_tier(state: SupportState) -> str:
    # YOUR CODE HERE
    # Return "vip_path" or "standard_path" based on state["user_tier"]
    pass

# Build graph
workflow = StateGraph(SupportState)

# 4. Add to graph
workflow.add_node("check_tier", check_user_tier_node)
workflow.set_entry_point("check_tier")  # Start here now

workflow.add_conditional_edges(
    "check_tier",
    route_by_tier,
    {
        # YOUR ROUTING DICT HERE
        # "vip_path": "???",
        # "standard_path": "???"
    }
)

"""
Extend the support agent with user tier-based routing:

Current:
   Start → Handle Request → [Escalate OR Respond]

Your Task:
   Start → Check User Tier → [VIP Path OR Standard Path]
           
VIP Path: Auto-resolve (no escalation)
Standard Path: May escalate if complex

What You Need to Add:
A new node: check_user_tier_node
Update state to track user_tier
A routing function: route_by_tier
Conditional edge from check_tier node

✅ Success Criteria
Graph has new check_tier node
Conditional routing works (VIP vs Standard)
Can trace different paths in output
VIP users skip escalation, Standard users may escalate

"""