from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.genai import types
import os

# Use a supported model
MODEL_NAME = os.getenv("ADK_MODEL", "gemini-1.5-flash")

# Agent A: Finds the capital of a country
agent_A = LlmAgent(
    name="CapitalFinder",
    model=MODEL_NAME,
    description="Finds the capital city of a given country and stores it in state.",
    instruction="You are an agent that finds the capital city of any country the user asks about. "
                "When the user provides a country name, output the capital city and save it to state under 'capital_city'.",
    tools=[google_search],
    output_key="capital_city"
)

# Agent B: Provides information about a city
agent_B = LlmAgent(
    name="CityExpert",
    model=MODEL_NAME,
    description="Provides a brief fact or information about a specified city.",
    instruction="You are an expert on cities. The city of interest is stored in the state under 'capital_city'. "
                "Share one interesting fact about that city."
)
root_agent = LlmAgent(name="greeter", model=MODEL_NAME)
# Combine them in a sequential workflow
pipeline_agent = SequentialAgent(
    name="CountryCapitalInfoAgent",
    sub_agents=[agent_A, agent_B]
)

if __name__ == "__main__":
    # Session and runner setup
    APP_NAME = "country_capital_info"
    USER_ID = "default_user"
    SESSION_ID = "session_1"

    # Initialize session service
    session_service = InMemorySessionService()
    session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    # Create runner
    runner = Runner(agent=pipeline_agent, app_name=APP_NAME, session_service=session_service)

    print("Running CountryCapitalInfoAgent... (Ctrl+C to exit)")
    while True:
        try:
            user_input = input("Enter a country name: ")
            content = types.Content(role="user", parts=[types.Part(text=user_input)])
            events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
            for event in events:
                if hasattr(event, 'is_final_response') and event.is_final_response():
                    final_text = event.content.parts[0].text
                    print(final_text)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
