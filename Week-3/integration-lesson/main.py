from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Optional
from prompt_manager import PromptManager
from ab_test_manager import ABTestManager
from pathlib import Path
load_dotenv()

prompt_manager = PromptManager(prompts_dir=str(Path(__file__).parent / "prompts"))
ab_test_manager = ABTestManager()


#Load the Prompt 
def load_prompt_with_fallback(agent_name:str, user_id: str):
    """ Load the prompt with fallback to v1.0.0 if current version is not available"""
    prompt_version = ab_test_manager.get_prompt_version(agent_name, user_id)

    if prompt_version == "current":
        try:
            prompt_data = prompt_manager.load_prompt(agent_name, prompt_version)
            return prompt_data, prompt_version
        except ValueError:
            prompt_version = "v1.0.0"
            prompt_data = prompt_manager.load_prompt(agent_name, prompt_version)
            return prompt_data, prompt_version
    else:
        prompt_data = prompt_manager.load_prompt(agent_name, prompt_version)
        return prompt_data, prompt_version

# Define the State

class AgentState(TypedDict):
    ticket: str
    user_id: str
    user_email: str
    specialist: Literal['billing', 'technical', 'general','escalate']
    routing_confidence: float
    routing_reasoning: str
    response: str
    specialist_used: str
    iteration_count: int
    log_trace: list
    sanitized_ticket: str
    prompt_version: str
    tokens_used: Optional[int]
    cost: Optional[float]
    error: Optional[str]


class TicketRouting(BaseModel):
    """ Simple Routing Decision """
    specialist: Literal['billing', 'technical', 'general','escalate']
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)

# Agents

# Define each class for new agents - 
# The reason, 
# The response is different for each agent - so we need to define a new class for each agent
# The specialist is different for each agent - so we need to define a new class for each agent
# The iteration count is different for each agent - so we need to define a new class for each agent
# The log trace is different for each agent - so we need to define a new class for each agent

class SupervisorAgent:
    """ Supervisor Agent """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.6).with_structured_output(TicketRouting)
    # Use the cheapest model for this agent

    def route(self, ticket: str, user_id: str) :
        """ Route the ticket to the appropriate specialist """
        prompt_data, prompt_version = load_prompt_with_fallback("supervisor", user_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, ticket)
        messages = [
            SystemMessage(content=compiled_prompt),
            HumanMessage(content=ticket)
        ]
        return self.llm.invoke(messages)


class BillingAgent:
    """ Billing Agent """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.6)
    # Use the cheapest model for this agent

    def handle(self, ticket: str, user_id: str) -> str : # Enhance the readability of the code by using the return type
        """ Handle the Billing related ticket"""
        prompt_data, prompt_version = load_prompt_with_fallback("billing", user_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, ticket)
        messages = [
            SystemMessage(content=compiled_prompt),
            HumanMessage(content=ticket)
        ]
        return self.llm.invoke(messages).content


class TechnicalAgent:
    """ Technical Agent """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.6)

    def handle(self, ticket: str, user_id: str) -> str :
        """ Handle the Technical related ticket"""
        prompt_data, prompt_version = load_prompt_with_fallback("technical", user_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, ticket)
        messages = [
            SystemMessage(content=compiled_prompt),
            HumanMessage(content=ticket)
        ]
        return self.llm.invoke(messages).content

class GeneralAgent:
    """ General Agent """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.6)

    def handle(self, ticket: str,user_id: str) -> str :
        """ Handle the General related ticket"""
        prompt_data, prompt_version = load_prompt_with_fallback("general", user_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, ticket)
        messages = [
            SystemMessage(content=compiled_prompt),
            HumanMessage(content=ticket)
        ]
        return self.llm.invoke(messages).content

class EscalateAgent:
    """ Escalate Agent """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.6)

    def handle(self, ticket: str,user_id: str) -> str :
        """ Handle the Escalate related ticket"""
        prompt_data, prompt_version = load_prompt_with_fallback("escalate", user_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, ticket)
        messages = [
            SystemMessage(content=compiled_prompt),
            HumanMessage(content=ticket)
        ]
        return self.llm.invoke(messages).content

# Steps to create the graph:
# 1. Initialize the StateGraph
# 2. Add the nodes for each agent
# 3. Add the edges for each agent - based on the routing decision
# 4. Compile the graph
# 5. Run the graph

def create_simple_graph(supervisor, billing, technical, general, escalate):
    """
    Create a simple graph for the ticket routing system
    Args:
        supervisor: SupervisorAgent
        billing: BillingAgent
        technical: TechnicalAgent
        general: GeneralAgent
        escalate: EscalateAgent
    Returns:
        workflow: StateGraph

    """
    # Initialize the StateGraph
    workflow = StateGraph(AgentState)

    # Supervisor Node
    def supervisor_node(state: AgentState) -> AgentState:
        """ Supervisor Node """
        print("Supervisor Node: Analyzing the ticket")
        routing = supervisor.route(state["ticket"],state["user_id"])
        # Update the state
        state["specialist"] = routing.specialist # This is not a recommended practice - we should use the return type to update the state
        state["routing_confidence"] = routing.confidence
        state["routing_reasoning"] = routing.reasoning
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["log_trace"] = state.get("log_trace", []) 
        state["log_trace"].append({
            "agent": "supervisor",
            "action": "routing",
            "specialist": routing.specialist,
            "confidence": routing.confidence,
            "reasoning": routing.reasoning
        })
        print(f"Routed to {routing.specialist} with confidence {routing.confidence} and reasoning {routing.reasoning}")
        return state

    workflow.add_node("supervisor", supervisor_node)

    # Billing Node
    def billing_node(state: AgentState) -> AgentState:
        """ Billing Node """
        print("Billing Node: Handling the ticket")
        state["response"] = billing.handle(state["ticket"],state["user_id"])
        state["specialist_used"] = "billing"
        # Update the state
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["log_trace"] = state.get("log_trace", []) 
        state["log_trace"].append({
            "agent": "billing",
            "action": "handling",
            "response": state["response"]
        })
        print(f"Handled the ticket with response {state['response']}")
        return state

    workflow.add_node("billing", billing_node)

    # Technical Node
    def technical_node(state: AgentState) -> AgentState:
        """ Technical Node """
        print("Technical Node: Handling the ticket")
        state["response"] = technical.handle(state["ticket"],state["user_id"])
        state["specialist_used"] = "technical"
        # Update the state
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["log_trace"] = state.get("log_trace", []) 
        state["log_trace"].append({
            "agent": "technical",
            "action": "handling",
            "response": state["response"]
        })
        print(f"Handled the ticket with response {state['response']}")
        return state

    workflow.add_node("technical", technical_node)

    # General Node
    def general_node(state: AgentState) -> AgentState:
        """ General Node """
        print("General Node: Handling the ticket")
        state["response"] = general.handle(state["ticket"],state["user_id"])
        state["specialist_used"] = "general"
        # Update the state
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["log_trace"] = state.get("log_trace", []) 
        state["log_trace"].append({
            "agent": "general",
            "action": "handling",
            "response": state["response"]
        })
        print(f"Handled the ticket with response {state['response']}")
        return state

    workflow.add_node("general", general_node)

    # Escalate Node
    def escalate_node(state: AgentState) -> AgentState:
        """ Escalate Node """
        print("Escalate Node: Escalating the ticket")
        state["response"] = escalate.handle(state["ticket"],state["user_id"])
        state["specialist_used"] = "escalate"
        # Update the state
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["log_trace"] = state.get("log_trace", []) 
        state["log_trace"].append({
            "agent": "escalate",
            "action": "handling",
            "response": state["response"]
        })
        print(f"Escalated the ticket with response {state['response']}")
        return state

    workflow.add_node("escalate", escalate_node)

    # Add the edges for each agent - based on the routing decision
    # Build Graph Structure
    workflow.add_edge(START, "supervisor")

    def routing_decision(state: AgentState) -> str:
        """ Routing Decision """
        return state["specialist"]

    workflow.add_conditional_edges("supervisor", routing_decision, {
        "billing": "billing",
        "technical": "technical",
        "general": "general",
        "escalate": "escalate"
    })
    # Connect every specialist to the END node
    workflow.add_edge("billing", END)
    workflow.add_edge("technical", END)
    workflow.add_edge("general", END)
    workflow.add_edge("escalate", END)
    return workflow.compile()

def main():
    """ Main function """
    supervisor = SupervisorAgent()
    billing = BillingAgent()
    technical = TechnicalAgent()
    general = GeneralAgent()
    escalate = EscalateAgent()
    graph = create_simple_graph(supervisor, billing, technical, general, escalate)
    
    test_cases = [
        "I was charged twice for my order# 12345",
        "How do I integrate my website with your payment gateway?",
        "I forgot my password, can you help me reset it?",
        "My Subscription is not working, can you help me?",
        "API is returning 429 error",
        "I am going to sue you for $1000000"
    ]
    results = []
    for test_case in test_cases:
        print(f"Testing: {test_case}")
        initial_state = {
            "ticket": test_case,
            "user_id": "1234567890",
            "specialist": "general", # default specialist
            "routing_confidence": 0.0,
            "routing_reasoning": "",
            "response": "",
            "specialist_used": "",
            "iteration_count": 0,
            "log_trace": []
        }
        result = graph.invoke(initial_state)
        results.append(result)
        print(f"Result: {result}")
        print(f"Specialist used: {result['specialist_used']}")
        print(f"Response: {result['response']}")
        print(f"Routing confidence: {result['routing_confidence']}")
        print(f"Routing reasoning: {result['routing_reasoning']}")
        print(f"Iteration count: {result['iteration_count']}")
        print(f"Log trace: {result['log_trace']}")
        print("-"*100)
    print(result)

if __name__ == "__main__":
    main()