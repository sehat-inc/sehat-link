from langgraph.graph import StateGraph, END

from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.nodes import (
    FrontendNode,
    LanguageDetectorNode,
    UrgencyDetectorNode
)
from core.logging import get_logger

logger = get_logger("AGENTIC GRAPH")

# Returns uncompiled graph since checkpointer was tweaking
def build_triage_agent():
    graph = StateGraph(MedicalAgentState)

    frontend = FrontendNode()
    language = LanguageDetectorNode()
    urgency = UrgencyDetectorNode()

    graph.add_node("frontend", frontend)
    graph.add_node("language", language)
    graph.add_node("urgency", urgency)

    graph.set_entry_point("frontend")

    # Parallel Branch
    graph.add_edge("frontend", "language")

    graph.add_edge("language", "urgency")

    graph.add_edge("urgency", END)

    return graph
