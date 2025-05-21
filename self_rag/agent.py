import os
from dotenv import load_dotenv
load_dotenv()

from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel

# Initialize embedding model (using Vertex AI Embedding API)
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#:~:text=Supported%20Models%3A
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# In-memory index to store document chunks and their vectors
DOCUMENT_INDEX = []  # each entry: {"text": chunk_text, "vector": embedding_values}

def ingest_document(doc_text: str) -> str:
    """Ingest a legal document by splitting it into chunks, embedding them, and storing for retrieval."""
    # Split the document text into chunks (e.g., by paragraphs)
    chunks = [paragraph for paragraph in doc_text.split("\n\n") if paragraph.strip()]
    # If very large paragraphs, further split or truncate (for simplicity, not shown here)

    for chunk in chunks:
        # Get embedding for the chunk
        emb_response = embedding_model.get_embeddings([chunk])
        vector = emb_response[0].values  # embedding vector for this chunk
        # Store the chunk and its vector in the index
        DOCUMENT_INDEX.append({"text": chunk, "vector": vector})
    return f"Ingested document with {len(chunks)} sections."

import math

def cosine_similarity(vec1, vec2):
    # Helper to compute cosine similarity between two vectors
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    return dot / (norm1 * norm2)

def search_documents(query: str) -> str:
    """Search the ingested documents for relevant content and return the most relevant text chunk."""
    if not DOCUMENT_INDEX:
        return "No documents available to search."
    # Embed the query
    query_emb = embedding_model.get_embeddings([query])[0].values
    # Find the best match in DOCUMENT_INDEX
    best_text = ""
    best_score = -1.0
    for entry in DOCUMENT_INDEX:
        score = cosine_similarity(query_emb, entry["vector"])
        if score > best_score:
            best_score = score
            best_text = entry["text"]
    # You could also return multiple top chunks; here we return the top one.
    return best_text

from google.adk import Agent

# Define the system instruction prompt for the agent
AGENT_INSTRUCTION = """
You are a helpful legal document Q&A assistant. You can ingest legal documents and answer questions about their content.
When the user asks a question about the documents:
- Use the search_documents tool to find the most relevant section of the text. (For example, search_documents(query="...").)
- Once you have the relevant text, read it and formulate a clear, concise answer for the user.
- Always provide a short citation from the text in your answer, prefaced with 'Based on section:' and a brief quote.
If the question is not related to the documents or if no document has been provided, politely indicate you have no information.
"""

# Instantiate the LLM agent with Gemini model and our tools
legal_qa_agent = Agent(
    name="legal_doc_qa_agent",
    model="gemini-2.0-flash-001",  # Gemini 2.0 Flash model for generation
    description="Agent that answers questions about uploaded legal documents using retrieval-augmented generation.",
    instruction=AGENT_INSTRUCTION,
    tools=[ingest_document, search_documents]
)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Prepare the sample document (as a single string)
contract_text = """Section 1: Parties. This Agreement is between Alpha Corp and Beta LLC...
Section 9: Termination. Either party may terminate this Agreement by giving thirty (30) days' written notice if the other party breaches any material term and fails to cure such breach within that period. Additionally, either party may terminate without cause with ninety (90) days' notice to the other party.
Section 10: Governing Law. This Agreement shall be governed by the laws of the State of California..."""

# Ingest the document (agent can also do this via tool invocation, here we call directly for setup)
ingest_document(contract_text)

# Set up an in-memory session and runner for the agent
session_service = InMemorySessionService()
session_service.create_session(app_name="legal_doc_qa_app", user_id="user1", session_id="session1")
runner = Runner(agent=legal_qa_agent, app_name="legal_doc_qa_app", session_service=session_service)

# Function to run a user query through the agent and get the final answer
def ask_agent(question: str) -> str:
    user_message = types.Content(role="user", parts=[types.Part(text=question)])
    events = runner.run(user_id="user1", session_id="session1", new_message=user_message)
    # Find the final response event
    for event in events:
        if event.is_final_response():
            return event.content.parts[0].text
    return "(No answer)"

# Now ask a question to the agent
query = "Under what conditions can this contract be terminated?"
response = ask_agent(query)
print("Agent's answer:", response)

