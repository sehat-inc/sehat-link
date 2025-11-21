from typing import Optional, Any, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.messages import AIMessage

from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.frontend_agent import TriageAgent
from core.langgraph.utils.language_node import LanguageDetectorNode
from core.langgraph.utils.urgency_node import UrgencyDetectorNode
from core.langgraph.utils.symptom_agent import SymptomAgentNode
from core.langgraph.utils.tool_manager import MCPToolManager
from core.langgraph.utils.tools import mcp_tool_node
from core.logging import get_logger


logger = get_logger("AGENTIC GRAPH")


def should_continue(state: MedicalAgentState) -> Literal["tools", "max_iterations", "__end__"]:
    """
    Enhanced conditional edge with max iteration check
    """
    messages = state["messages"]
    last_message = messages[-1]
    logger.info(f"LAST MESSAGE FROM SHOULD CONTINUE: {last_message}")

    # Check Max Iterations
    max_tool_calls = 10
    if state.get("tool_call_count", 0) >= max_tool_calls:
        logger.warning(f"Max Tool Calls Reached: ({max_tool_calls}) reached")
        return "max_iterations"
    
    # Check for errors
    if state.get("error_count", 0) >= 3:
        logger.error("Too many errors, ending execution")
        return "__end__"
    
    # Check if LLM wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("RETURNING TOOL")
        return "tools"
    
    return "__end__"

def should_continue_traige(state: MedicalAgentState):
    if state["symptom_trigger"] == True:
        return "symptom"
    else:
        return "continue"

async def max_iterations_node(state: MedicalAgentState):
    """Handle max iterations exceeded"""
    message = AIMessage(
        content="I've reached the maximum number of tool calls. Let me summarize what I've found so far..."
    )
    return {"messages": [message]}


def build_triage_agent():
    """
    Create LangGraph Workflow
    """

    graph = StateGraph(MedicalAgentState)
    
    triage = TriageAgent()
    symptom = SymptomAgentNode()
    language = LanguageDetectorNode()

    graph.add_node("triage", triage.run)
    graph.add_node("symptom", symptom.run)
    graph.add_node("language", language)
    graph.add_node("tools", mcp_tool_node)
    graph.add_node("max_iterations", max_iterations_node)

    # Set Entrypoint
    graph.set_entry_point("triage")

    # Add Conditional Edge from Triage
    graph.add_conditional_edges(
        "triage",
        should_continue_traige,
        {
            "symptom": "symptom",
            "continue": "language",
        }
    )

    graph.add_conditional_edges(
        "symptom",
        should_continue,
        {
            "tools": "tools",
            "max_iterations": "max_iterations",
            "__end__": END
        }
    )

    
    graph.add_edge("language", END)
    graph.add_edge("tools", "symptom")
    graph.add_edge("max_iterations", END)

    return graph
