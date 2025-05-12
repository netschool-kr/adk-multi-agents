# agent.py

from google.adk.agents import Agent      # ADK Agent class for LLM-based agents
import requests                          # To call an external API for definitions (if needed)

def get_definition(term: str) -> str:
    """
    Fetch the definition of an English word.

    Args:
        term (str): The word to define.

    Returns:
        str: A short definition of the word, or a message if not found.
    """
    # Use a free dictionary API to get the definition
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Extract the first definition from the response JSON
            definition = data[0]['meanings'][0]['definitions'][0]['definition']
            return definition
        else:
            # If API returns 404 or error, handle gracefully
            return "I’m sorry, I couldn’t find a definition for that word."
    except Exception as e:
        # Handle any unexpected errors (network issues, parsing errors)
        return "I’m sorry, I couldn’t retrieve the definition due to an error."
    
# Define the root agent for the vocabulary assistant
root_agent = Agent(
    name="vocab_assistant",
    description="An agent that explains the meanings of English words.",

    # List the tools the agent can use (our get_definition function)
    tools=[get_definition],

    # Specify the LLM model to use (Google Gemini via API key)
    model="gemini-2.0-flash", 

    # Instruction prompt to define the agent’s behavior
    instruction=(
        "You are a helpful vocabulary assistant. "
        "Your job is to explain the meanings of English words. "
        "When the user asks for a definition of a word, you may use the get_definition tool to fetch the formal definition, "
        "then explain it in simple terms. "
        "Provide clear, concise explanations. If a word is not found, apologize and suggest they check the spelling."
    )
)

