import os
from dotenv import load_dotenv
load_dotenv()

# Initialize Vertex AI SDK
import vertexai
# from vertexai.language_models import TextGenerationModel # Commented out or delete as we are using Gemini
from vertexai.generative_models import GenerativeModel # Using GenerativeModel class for Gemini models
vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["GOOGLE_CLOUD_LOCATION"])

# Change to the successfully tested Gemini model ID
MODEL_NAME = "gemini-2.0-flash-lite-001"
# classifier_model = TextGenerationModel.from_pretrained(MODEL_NAME) # Previous code
# Change classifier_model initialization to use GenerativeModel for Gemini
classifier_model = GenerativeModel(MODEL_NAME)

# Define and register ADK BaseLlm derived class
from google.adk.models import BaseLlm
from google.adk.models.registry import LLMRegistry
from google.genai import types as genai_types # Use alias to differentiate from ADK's potential types

# New ADK LLM wrapper class for Gemini models
class VertexAIGemini(BaseLlm):
    """LLM wrapper for using Vertex AI Gemini models with ADK."""
    model: str = MODEL_NAME  # Pydantic model field; actual model ID determined at instantiation

    @staticmethod
    def supported_models():
        # Regex pattern for Vertex AI Gemini model IDs (e.g., gemini-1.5-pro-001, gemini-1.0-pro, gemini-2.0-flash-lite-001, etc.)
        # Modified to a more general Gemini pattern
        return [
            r"^gemini-.*$"
        ]

    async def generate_content_async(self, llm_request, stream: bool = False):
        """Generates content from the Vertex AI Gemini model based on the llm_request."""
        if hasattr(llm_request, 'messages'):
            # LlmRequest.messages in ADK can be a list of Content objects.
            # Gemini SDK's generate_content expects a single Content object or a list of strings.
            # This needs to be adapted based on LlmRequest structure; here, we assume using text from the last user message.
            # For actual use, verify the LlmRequest structure and convert messages if needed.
            if llm_request.messages and llm_request.messages[-1].parts:
                content_input = llm_request.messages[-1].parts[0].text
            else:
                raise ValueError("LLMRequest messages are empty or not in expected format for Gemini prompt.")
        elif hasattr(llm_request, 'prompt'):
            content_input = llm_request.prompt    # Single prompt text
        else:
            content_input = str(llm_request)      # Fallback: try direct string conversion

        # Use GenerativeModel from Vertex AI SDK
        gemini_model_instance = GenerativeModel(self.model) # Use model ID from constructor

        # Note: stream=True (streaming) processing is not fully implemented here.
        # Further implementation might be needed depending on how ADK's BaseLlm expects streaming.
        if stream:
            # Streaming example (compatibility with ADK BaseLlm and detailed implementation needed)
            # responses = gemini_model_instance.generate_content(content_input, stream=True)
            # for response_chunk in responses:
            #     output_text_chunk = getattr(response_chunk, "text", "")
            #     if output_text_chunk:
            #         output_content_chunk = genai_types.Content(role="assistant", parts=[genai_types.Part(text=output_text_chunk)])
            #         yield output_content_chunk
            # return # End here if streaming
            print("Streaming not fully implemented in this example for VertexAIGemini.")
            # Temporarily act like non-streaming
            pass


        response = gemini_model_instance.generate_content(content_input)

        # Extract text from the response (may vary based on Gemini model's response structure)
        # Attempt to access .text directly, or parse from response.candidates[0].content.parts[0].text
        # Defaulting to .text for common cases
        output_text = ""
        try:
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 output_text = response.candidates[0].content.parts[0].text
            elif hasattr(response, 'text'): # For simple text responses
                output_text = response.text
            else: # Unexpected response structure
                output_text = str(response) # Safely convert to string
        except AttributeError: # If the response object doesn't have the expected attributes
             output_text = str(response)


        output_content = genai_types.Content(role="assistant", parts=[genai_types.Part(text=output_text)])
        yield output_content

# Register the new Gemini LLM class with the registry
LLMRegistry.register(VertexAIGemini)

# Check if registered model patterns are recognized correctly (example output)
# Changed candidates list from Bison models to Gemini model ID
print("ADK registered model list check (Gemini Test):")
gemini_candidates = [MODEL_NAME, "gemini-1.5-pro-latest", "gemini-1.0-pro-001"] # Example Gemini IDs
for model_id in gemini_candidates:
    try:
        # When an LlmAgent is created with a model argument, LLMRegistry calls resolve.
        # This doesn't fetch the LLM instance directly here but can check pattern matching.
        llm_instance = LLMRegistry.resolve(model_id) # This attempts to create an instance
        print(f"✔ {model_id} (Resolved to: {llm_instance.__class__.__name__})")
    except ValueError as e:
        print(f"✘ {model_id} (Error: {e})")


# Tool function for document classification
def classify_document(text: str) -> str:
    """Classifies the uploaded document content and returns the category name."""
    prompt = f"Classify the following document into a single broad category:\n\"\"\"\n{text}\n\"\"\"\nCategory:"
    # Use generate_content instead of classifier_model.predict and modify response handling
    # response = classifier_model.predict(prompt, max_output_tokens=5, temperature=0.0) # Previous code
    # Gemini models typically pass max_output_tokens, etc., via generation_config
    from vertexai.generative_models import GenerationConfig
    generation_config = GenerationConfig(
        max_output_tokens=10, # Set short as it's a category name
        temperature=0.0
    )
    response = classifier_model.generate_content(prompt, generation_config=generation_config)

    category = ""
    try:
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                category = response.candidates[0].content.parts[0].text.strip()
        elif hasattr(response, 'text'): # For simple text responses
            category = response.text.strip()
        else: # Unexpected response structure
            category = "Unknown Category"
    except AttributeError:
        category = "Unknown Category"

    return category

# Define LLM agent (using the changed MODEL_NAME)
from google.adk.agents import LlmAgent
document_agent = LlmAgent(
    name="DocumentClassifierAgent",
    model=MODEL_NAME,  # Gemini model ID to be handled by VertexAIGemini
    description="Classification Agent: Classifies uploaded documents into categories.",
    instruction=(
        "You are a document classification agent. "
        "When a user provides a document, identify its category. "
        "Use the 'classify_document' tool to get the category, then respond with the category name and a brief explanation."
    ),
    tools=[classify_document]
)
root_agent = document_agent

# Agent execution example (InMemory session and Runner setup)
if __name__ == "__main__":
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    # from google.genai import types  # Already aliased as genai_types

    session_service = InMemorySessionService()
    # Using hardcoded session_id as in the provided "agent copy.py"
    session_service.create_session(app_name="doc_classifier_app", user_id="user1", session_id="session1")
    runner = Runner(agent=root_agent, app_name="doc_classifier_app", session_service=session_service)
    print("Agent running... (Ctrl+C to exit)") # Changed to English
    while True:
        try:
            user_input = input("Enter document content: ") # Changed to English
            # Use genai_types.Content instead of types.Content
            user_content = genai_types.Content(role="user", parts=[genai_types.Part(text=user_input)])
            events = runner.run(user_id="user1", session_id="session1", new_message=user_content)
            # Print the final agent response
            for event in events:
                if hasattr(event, "is_final_response") and event.is_final_response():
                    answer = event.content.parts[0].text
                    print("Agent:", answer)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during agent execution: {e}") # Changed to English
            import traceback
            traceback.print_exc()
