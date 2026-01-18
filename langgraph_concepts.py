from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

# 1. Define State
class MyMemory(TypedDict):
    user_tier: str

def get_user_tier(state) -> MyMemory: # node function
    return {"user_tier": "premium"}