from typing import Optional
from langgraph.graph import StateGraph, END, START

from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.frontend_agent import FrontendNode
from core.langgraph.utils.language_node import LanguageDetectorNode
from core.langgraph.utils.urgency_node import UrgencyDetectorNode
from core.langgraph.utils.symptom_agent import SymptomAgentNode
from core.langgraph.utils.urgency_node import UrgencyDetectorNode 
from core.langgraph.utils.tool_manager import MCPToolManager
from core.logging import get_logger

logger = get_logger("AGENTIC GRAPH")


def start_router(state: MedicalAgentState):
    curr = state.get("current_agent", None)
    if not curr:
        return "frontend"
    if curr == "frontend_agent" or curr == "frontend":
        return "frontend"
    if curr == "symptom_agent" or curr == "symptom":
        return "symptom"
    # Default fallback
    return "frontend"


def frontend_to_other(state: MedicalAgentState):
    if state["symptom_trigger"] == True:
        return "symptom"
    else:
        return "continue"



def build_triage_agent(mcp_manager: Optional[MCPToolManager]):
    graph = StateGraph(MedicalAgentState)

    frontend = FrontendNode()
    language = LanguageDetectorNode()
    urgency = UrgencyDetectorNode()
    symptom = SymptomAgentNode(
        mcp_manager=mcp_manager,
        allowed_tools=["Symptom Knowledge Base Smart Query"]
    )

    
    graph.add_node("frontend", frontend)
    graph.add_node("language", language)
    graph.add_node("urgency", urgency)
    graph.add_node("symptom", symptom)

    
    graph.add_conditional_edges(
        START, 
        start_router,
        {"frontend": "frontend", "symptom": "symptom"}
    )
    graph.add_conditional_edges(
        "frontend",
        frontend_to_other,
        {"symptom": "symptom", "continue": "language"}
    )
    graph.add_edge("language", END)
    graph.add_edge("symptom", "urgency")
    graph.add_edge("urgency", END)


    return graph
