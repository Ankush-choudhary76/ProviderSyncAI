# from typing import List
# from smolagents import CodeAgent, Tool
# from ..infrastructure.settings import settings
# from ..infrastructure.models.grok_model import GrokModel
# from .tools.nppes_tool import NppesTool
# from .tools.web_search_tool import WebSearchTool

# class NppesTool(Tool):
#     name = "NPPES Provider Lookup"
#     description = "Searches the NPI registry for healthcare providers."
#     output_type = "json"  # ✅ Add this line!

#     def run(self, query: str):
#         # Your implementation here
#         return {"result": f"Searched NPPES for {query}"}
    

# class WebSearchTool(Tool):
#     name = "Web Search Tool"
#     description = "Performs a web search and returns top results."
#     output_type = "string"  # ✅ Add this line!

#     def run(self, query: str):
#         # Your implementation here
#         return f"Results for: {query}"
    
    
# def build_agent() -> CodeAgent:
#     tools: List[Tool] = [NppesTool(), WebSearchTool()]
    
#     # Initialize Grok model if API key is available
#     model = None
#     if settings.grok_api_key:
#         try:
#             model = GrokModel(api_key=settings.grok_api_key, model_name="grok-beta", temperature=0.7)
#         except Exception as e:
#             # Log error but continue with default model if Grok fails
#             import logging
#             logging.warning(f"Failed to initialize Grok model: {e}. Using default model.")
    
#     # CodeAgent will use the Grok model if available, otherwise fallback to default behavior
#     agent = CodeAgent(tools=tools, model=model, name="coordinator")
#     return agent


from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent

from ..infrastructure.settings import settings
from ..infrastructure.models.grok_model import GrokModel
from .state import AgentState
from .specialized.data_validation_agent import data_validation_node
from .specialized.information_enrichment_agent import information_enrichment_node
from .specialized.quality_assurance_agent import quality_assurance_node

# Define members/workers
members = ["DataValidation", "InformationEnrichment", "QualityAssurance"]

# Supervisor System Prompt
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

options = members + ["FINISH"]

# Using a simpler router or function calling if possible. 
# Since Grok might not support structured output perfectly via LangChain yet, 
# we can use a simple prompt-based router or JSON mode if available.
# We will assume Grok is smart enough to output the name.

async def supervisor_node(state: AgentState):
    messages = state["messages"]
    # If no messages, do nothing or start
    
    model = GrokModel()
    
    # We can use binding or just a prompting strategy.
    # Simple prompting strategy:
    prompt = SystemMessage(content=system_prompt.format(members=", ".join(members)))
    
    # Add history
    request = messages[-1] if messages else HumanMessage(content="Start processing.")
    
    # We might need to summarize state if history is long, but for now pass all.
    response = await model.ainvoke([prompt] + messages)
    
    # Simple text parsing for the route
    content = response.content.strip()
    
    # Fallback to defaults if unclear, or check for keywords
    next_step = "FINISH"
    for m in members:
        if m in content:
            next_step = m
            break
            
    if "FINISH" in content:
        next_step = "FINISH"
        
    return {"next": next_step}


def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("DataValidation", data_validation_node)
    workflow.add_node("InformationEnrichment", information_enrichment_node)
    workflow.add_node("QualityAssurance", quality_assurance_node)

    # Add edges
    for member in members:
        # After a worker runs, go back to supervisor
        workflow.add_edge(member, "supervisor")

    # The supervisor determines the next step
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

    # Start point
    workflow.add_edge(START, "supervisor")

    return workflow.compile()

# Compatibility for existing code calling build_agent()
# If main.py expects an agent with .run(), we need a wrapper.
# The graph has .ainvoke().
class GraphAgentWrapper:
    def __init__(self, graph):
        self.graph = graph
        
    async def run(self, input_data: str):
        # We need to constructing initial state
        # Assuming input_data is the prompt string
        initial_state = {
            "messages": [HumanMessage(content=input_data)],
            "next": "supervisor",
            "provider_data": {}, # Should be populated if context is known
            "agents_run": []
        }
        result = await self.graph.ainvoke(initial_state)
        # Extract final message
        return result["messages"][-1].content

def build_agent():
    """
    Builds and returns the graph wrapped as an agent.
    """
    graph = build_graph()
    return GraphAgentWrapper(graph)
