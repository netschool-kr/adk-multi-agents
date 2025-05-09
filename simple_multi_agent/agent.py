from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

# Agent A: finds the capital of a country (using a tool if needed)
agent_A = LlmAgent(
    name="CapitalFinder",
    model="gemini-2.0",  # using Google Gemini model for demonstration
    description="Finds the capital city of a given country and stores it in state.",
    instruction="You are an agent that finds the capital city of any country the user asks about. "
                "When the user provides a country name, output the capital city and save it to state under 'capital_city'.",
    tools=[google_search],  # it can use Google Search if it needs information
    output_key="capital_city"  # this tells ADK to store the final answer in session.state['capital_city']
)

# Agent B: provides information about a city (expects 'capital_city' to be in state)
agent_B = LlmAgent(
    name="CityExpert",
    model="gemini-2.0",
    description="Provides a brief fact or information about a specified city.",
    instruction="You are an expert on cities. The city of interest is stored in the state under 'capital_city'. "
                "Share one interesting fact about that city."
    # We rely on the instruction to tell the agent to look at state['capital_city'].
    # (Alternatively, we could programmatically pass state content via include_contents or input variables.)
)

# Combine them in a sequential workflow
pipeline_agent = SequentialAgent(
    name="CountryCapitalInfoAgent",
    sub_agents=[agent_A, agent_B]
)
