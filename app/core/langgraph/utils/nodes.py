from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI 
from api.mcp.client import Client
from langchain_core.messages import SystemMessage, AIMessage

from utils.state import MedicalAgentState
from core.prompts.triage_agent_prompt import triage_agent_prompt
from core.longterm_memory import PineconeMemory


class Node:
    """
    Base Node Interface :) 
    
    Remarks:
        I did this solely because after re-writing this code times that all nodes must be 
        uniform and I cant just create functions like in docs otherwise it will get extremely
        hard to scale when there are more than 5 agents
    """
    name: str

    def __init__(self, name: str):
        self.name = name

    async def __call__(self, state: MedicalAgentState) -> Dict[str, Any]:
        raise NotImplementedError("Node Children must implement __call__")




# class TriageAgentNode:
#     """
#     LangGraph based Triage Agent Node
#     """
#
#     def __init__(
#         self, 
#         mcp_client: Client, 
#         api_key: str, 
#         state: MedicalAgentState,
#     ):
#         self.mcp_client = mcp_client
#         self.llm = ChatGoogleGenerativeAI(
#             api_key=api_key,
#             model="gemini-2.5-flash",
#             temperature=0.7
#         )
#         self.state = state
#         self.tools = [] 
#
#     async def init_agent(self, pc_api_key, pc_index_name):
#         self.tools = await self.mcp_client.list_tools()
#         self.memory = PineconeMemory(pc_api_key, pc_index_name)
#
#     async def run(self):
#         if self.state["symptoms_summary"] and not self.state.get("similar_cases"):
#             similar_cases = await self.memory.search_similar_cases(
#                 user_id=self.state["user_id"],
#                 query=self.state["symptoms_summary"],
#                 top_k=2,
#                 filter_dict={"type": "conversation"}
#             )
#             self.state["similar_cases"] = similar_cases
#
#             similar_context = "\n".join([
#                 f"- {case['date']}: {case['summary']} (Outcome: {case['outcome']})"
#                 for case in similar_cases
#             ])
#         else:
#             similar_context = "No similar cases found"
#
#         system_prompt = triage_agent_prompt(
#             state=self.state,
#             similar_context=similar_context
#         )
#
#         messages = [SystemMessage(content=system_prompt)] + self.state["messages"]
#         response = await self.llm.ainvoke(messages)
#
#         self.state["messages"].append(AIMessage(content=response.text))
#
#         # Trigger MCP deep research if sufficient data
#         if self.state["sufficient_symptom_data"] and not self.state["symptom_research_result"]:
#             self.state["requires_deep_research"] = True
#
#         return self.state
