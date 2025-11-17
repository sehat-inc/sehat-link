from langgraph_swarm import create_handoff_tool, create_swarm

transfer_to_symptom_agent = create_handoff_tool(
    agent_name="symptom",
    description="Transfer user to Symptom Agent named Ms Bukhari that is responsible for aiding in symptom research and medical analysis."
) 

transfer_to_frontend_agent = create_handoff_tool(
    agent_name="frontend",
    description="Transfer user to Frontend Agent named Ms Sehat that is responsible for all greetings and small talk. The Frontend agent is an AI healthcare receptionist"
) 


