import os
from dotenv import load_dotenv
load_dotenv()

# Initialize Vertex AI SDK
import vertexai
from vertexai.generative_models import GenerativeModel  # Using GenerativeModel for Gemini
vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"]
)

# Use a tested Gemini model
MODEL_NAME = "gemini-2.0-flash-001"

# Sample catalog
PRODUCT_CATALOG = [
    {"id": "P1001", "name": "Wireless Mouse", "category": "Electronics", "price": 25.0},
    {"id": "P1002", "name": "Bluetooth Keyboard", "category": "Electronics", "price": 45.0},
    {"id": "P2001", "name": "Running Shoes", "category": "Sportswear", "price": 60.0},
]


def search_product_catalog(query: str) -> dict:
    """Lookup a product by name or category."""
    q = query.lower()
    for p in PRODUCT_CATALOG:
        if q in p["name"].lower() or q in p["category"].lower():
            return {"id": p["id"], "name": p["name"], "price": p["price"]}
    return {}

INVENTORY_DB = {
    "P1001": 15,
    "P1002": 8,
    "P2001": 0,
}

def check_stock(product_id: str, quantity: int) -> str:
    """Return 'In Stock' or 'Out of Stock'."""
    return "In Stock" if INVENTORY_DB.get(product_id, 0) >= quantity else "Out of Stock"

import json
from google.adk.tools import ToolContext

def process_order(product_id: str, quantity: int, tool_context: ToolContext) -> str:
    """Simulate order and return JSON confirmation."""
    import datetime, random
    order_id = f"ORD-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1000,9999)}"
    confirmation = {
        "order_id": order_id,
        "product_id": product_id,
        "quantity": quantity,
        "status": "confirmed",
        "payment": "approved"
    }
    # allow LLM summarization
    tool_context.actions.skip_summarization = False
    return json.dumps(confirmation)

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

# Agents setup
product_agent = LlmAgent(
    name="ProductLookupAgent",
    model=MODEL_NAME,
    description="Lookup product details.",
    instruction=(
        "Use search_product_catalog tool to find product by name or category. "
        "Respond with JSON {id, name, price}."
    ),
    tools=[FunctionTool(func=search_product_catalog)],
    output_key="product_info"
)

inventory_agent = LlmAgent(
    name="InventoryAgent",
    model=MODEL_NAME,
    description="Check stock availability.",
    instruction=(
        "Given product_info JSON and quantity, use check_stock tool. "
        "Respond exactly 'In Stock' or 'Out of Stock'."
    ),
    tools=[FunctionTool(func=check_stock)],
    output_key="stock_status"
)

order_agent = LlmAgent(
    name="OrderAgent",
    model=MODEL_NAME,
    description="Process orders and output confirmation.",
    instruction=(
        "Given product_info and stock_status: "
        "If 'In Stock', call process_order tool and output JSON. "
        "If 'Out of Stock', output 'Unable to complete order: item is out of stock.'"
    ),
    tools=[FunctionTool(func=process_order)]
)

workflow_agent = SequentialAgent(
    name="EcommerceWorkflow",
    sub_agents=[product_agent, inventory_agent, order_agent]
)
root_agent = workflow_agent

import asyncio

async def main():
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.genai import types as genai_types

    session_service = InMemorySessionService()
    # Asynchronously create session
    await session_service.create_session(
        app_name="ECommerce_app", user_id="user1", session_id="session1"
    )

    runner = Runner(
        agent=root_agent,
        app_name="ECommerce_app",
        session_service=session_service
    )
    print("Agent running... (Ctrl+C to exit)")

    while True:
        try:
            user_input = input("Enter request: ") or "I want to buy 2 units of Wireless"
            user_content = genai_types.Content(
                role="user", parts=[genai_types.Part(text=user_input)]
            )

            # Iterate over async generator
            async for event in runner.run_async(
                user_id="user1", session_id="session1", new_message=user_content
            ):
                if hasattr(event, "is_final_response") and event.is_final_response():
                    text = ""
                    if event.content and event.content.parts:
                        text = "".join(
                            [p.text for p in event.content.parts if getattr(p, 'text', None)]
                        )
                    print("Agent:", text or "Final response did not contain any displayable text.")
                    break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during agent execution: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
