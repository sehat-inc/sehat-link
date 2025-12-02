from typing import Optional, Any, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.messages import AIMessage

from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.frontend_agent import TriageAgent
from core.langgraph.utils.language_node import LanguageDetectorNode
from core.langgraph.utils.program_eligibility_agent import ProgrammeEligibilityNode
from core.langgraph.utils.urgency_node import UrgencyDetectorNode
from core.langgraph.utils.symptom_agent import SymptomAgentNode
from core.langgraph.utils.doctor_agent import DoctorAgentNode
from core.langgraph.utils.prescription_agent import PrescriptionAgent
from core.langgraph.utils.tool_manager import MCPToolManager
from core.langgraph.utils.tools import mcp_tool_node
from core.logging import get_logger


logger = get_logger("AGENTIC GRAPH")


def start_router(state: MedicalAgentState) -> Literal["symptom", "triage", "doctor", "program", "prescription"]:
    messages = state.get("messages", [])
    
    # Look at the last human message (not AI/tool messages)
    for msg in reversed(messages):
        if msg.__class__.__name__ == "HumanMessage":
            if hasattr(msg, "content") and isinstance(msg.content, list):
                if any(isinstance(i, dict) and i.get("type") == "image_url" for i in msg.content):
                    # Only process if we haven't already
                    if not state.get("prescription_processed"):
                        return "prescription"
            break

    curr = state.get("current_agent", "__default__")
    logger.info(f"CURRENT AGENT: {curr}")

    if curr == "__default__":
        return "triage"
    elif curr == "triage_agent":
        return "triage"
    elif curr == "symptom_agent":
        return "symptom"
    elif curr == "programme_eligibility_agent":
        return "program"
    elif curr == "doctor_agent":
        return "doctor"
    
    return "triage"
    

def should_continue_symptom(state: MedicalAgentState) -> Literal["tools", "max_iterations", "urgency"]:
    """
    Enhanced conditional edge with max iteration check.
    All handoffs from symptom now go through urgency node first.
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
        return "urgency"  # Still go through urgency to set state
    
    # Check if LLM wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("RETURNING TOOL")
        return "tools"

    # All exits from symptom agent go through urgency node
    # Urgency node will check if it should run or skip (urgency_checked flag)
    return "urgency"


def route_after_urgency(state: MedicalAgentState) -> Literal["program", "doctor", "__end__"]:
    """
    Route to appropriate agent after urgency check is complete.
    """
    current_agent = state.get("current_agent", "")
    
    if current_agent == "programme_eligibility_agent":
        logger.info("Post-urgency: Routing to Program Eligibility Agent")
        return "program"
    
    if current_agent == "doctor_agent":
        logger.info("Post-urgency: Routing to Doctor Agent")
        return "doctor"
    
    logger.info("Post-urgency: Ending conversation")
    return "__end__"

def should_continue_program(state: MedicalAgentState) -> Literal["tools", "symptom", "doctor",  "__end__"]:
    """
    Enhanced conditional edge with max iteration check
    """
    messages = state["messages"]
    last_message = messages[-1]
    logger.info(f"LAST MESSAGE FROM SHOULD CONTINUE: {last_message}")

    
    # Check for errors
    if state.get("error_count", 0) >= 3:
        logger.error("Too many errors, ending execution")
        return "__end__"
    
    # Check if LLM wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("RETURNING TOOL")
        return "tools"

    if state.get("current_agent") == "symptom_agent":
        logger.info("Handoff to Symptom Agent")
        return "symptom"
    
    if state.get("current_agent") == "doctor_agent":
        logger.info("Handoff to Doctor Agent")
        return "doctor"
    
    return "__end__"

def should_continue_doctor(state: MedicalAgentState) -> Literal["tools", "symptom", "program", "__end__"]:
    """
    Enhanced conditional edge with max iteration check
    """
    messages = state["messages"]
    last_message = messages[-1]
    logger.info(f"LAST MESSAGE FROM SHOULD CONTINUE: {last_message}")

    
    # Check for errors
    if state.get("error_count", 0) >= 3:
        logger.error("Too many errors, ending execution")
        return "__end__"
    
    # Check if LLM wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("RETURNING TOOL")
        return "tools"

    if state.get("current_agent") == "symptom_agent":
        logger.info("Handoff to Symptom Agent")
        return "symptom"
    
    if state.get("current_agent") == "programme_eligibility_agent":
        logger.info("Handoff to Program Agent")
        return "program"
    
    return "__end__"



def should_continue_traige(state: MedicalAgentState) -> Literal["tools" ,"symptom", "program", "doctor", "continue"]:
    
    messages = state["messages"]
    last_message = messages[-1]
    logger.info(f"LAST MESSAGE FROM SHOULD CONTINUE: {last_message}")
    
    # Check if LLM wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("RETURNING TOOL")
        return "tools"

    if state["current_agent"] == "symptom_agent":
        return "symptom"
    if state["current_agent"] == "programme_eligibility_agent":
        return "program"
    if state["current_agent"] == "doctor_agent":
        return "doctor"
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
    program = ProgrammeEligibilityNode()
    doctor = DoctorAgentNode()
    urgency = UrgencyDetectorNode()
    prescription = PrescriptionAgent()

    graph.add_node("triage", triage.run)
    graph.add_node("symptom", symptom.run)
    graph.add_node("language", language.run)
    graph.add_node("program", program.run)
    graph.add_node("urgency", urgency.run)
    graph.add_node("doctor", doctor.run)
    graph.add_node("prescription", prescription.run)
    graph.add_node("triage_tools", mcp_tool_node)
    graph.add_node("symptom_tools", mcp_tool_node)
    graph.add_node("program_tools", mcp_tool_node)
    graph.add_node("doctor_tools", mcp_tool_node)
    graph.add_node("max_iterations", max_iterations_node)


    graph.add_conditional_edges(
        START,
        start_router,
        {
            "triage": "triage",
            "doctor": "doctor",
            "program": "program",
            "symptom": "symptom",
            "prescription": "prescription" 
        }
    )

    # Add Conditional Edge from Triage
    graph.add_conditional_edges(
        "triage",
        should_continue_traige,
        {
            "tools" : "triage_tools",
            "symptom": "symptom",
            "program": "program",
            "doctor": "doctor",
            "continue": "language",
        }
    )

    graph.add_conditional_edges(
        "symptom",
        should_continue_symptom,
        {
            "tools": "symptom_tools",
            "max_iterations": "max_iterations",
            "urgency": "urgency"
        }
    )
    
    # Route after urgency check to appropriate agent or end
    graph.add_conditional_edges(
        "urgency",
        route_after_urgency,
        {
            "program": "program",
            "doctor": "doctor",
            "__end__": END
        }
    )
    
    graph.add_conditional_edges(
        "program",
        should_continue_program,
        {
            "tools": "program_tools",
            "symptom": "symptom",
            "doctor": "doctor",
            "__end__": END
        }
    )

    graph.add_conditional_edges(
        "doctor",
        should_continue_doctor,
        {
            "tools": "doctor_tools",
            "symptom": "symptom",
            "program": "program",
            "__end__": END
        }
    )
    
    graph.add_edge("prescription", END)
    graph.add_edge("language", END)
    graph.add_edge("triage_tools", "triage")
    graph.add_edge("symptom_tools", "symptom")
    graph.add_edge("program_tools", "program")
    graph.add_edge("doctor_tools", "doctor")
    graph.add_edge("max_iterations", END)

    return graph
