from langgraph.graph import StateGraph, END

from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.frontend_agent import FrontendNode
from core.langgraph.utils.language_node import LanguageDetectorNode
from core.langgraph.utils.urgency_node import UrgencyDetectorNode
from core.langgraph.utils.prescription_agent import PrescriptionAgent
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
    graph.add_node("prescription_agent", PrescriptionAgent())


    graph.set_entry_point("frontend")

    # Parallel Branch
    graph.add_edge("frontend", "language")

    graph.add_edge("language", "urgency")

    graph.add_edge("urgency", END)


    return graph
