from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.nodes import (
    FrontendNode,
    LanguageDetectorNode,
    UrgencyDetectorNode
)


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

    compiled = graph.compile()

    return compiled


async def run_graph_with_message(graph, state: MedicalAgentState, user_message: str) -> Dict[str, Any]:
    """
    Execute graph with a new user message and return updated state + response.
    
    Args:
        graph: Compiled LangGraph instance
        state: Current conversation state
        user_message: New message from user
    
    Returns:
        Dict with 'state', 'response', and 'agent' keys
    """
    try:
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_message))
        
        # Execute graph (async invoke)
        result = await graph.ainvoke(state)
        
        # Extract AI response (last message)
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        last_response = ai_messages[-1].content if ai_messages else "I'm processing your request..."
        
        return {
            "state": result,
            "response": last_response,
            "agent": result.get("current_agent", "unknown"),
            "urgency": result.get("detected_urgency", "Low"),
            "language": result.get("detected_language", "english")
        }
    
    except Exception as e:
        import traceback
        print(f"Graph execution error: {e}")
        print("\n======= GRAPH ERROR =======")
        print(e)
        traceback.print_exc()
        print("======= END GRAPH ERROR =======\n")
        return {
            "state": state,
            "response": "I apologize, but I encountered an error. Please try again.",
            "agent": state.get("current_agent", "frontend"),
            "error": str(e)
        }



