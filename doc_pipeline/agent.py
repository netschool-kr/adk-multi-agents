# agent.py
from google.adk.agents import LlmAgent, SequentialAgent  # and ParallelAgent if needed later
# Also import or configure your model. For example:
MODEL_NAME = "gemini-2.0-flash"  # or another model like "text-bison-001" depending on your setup

# --- Define Sub-Agent 1: Summarizer ---
summarizer_agent = LlmAgent(
    name="SummarizerAgent",
    model=MODEL_NAME,
    description="Summarizes an English document into a concise summary.",
    instruction=(
        "You are a document summarization AI. "
        "Your task is to read an English document provided by the user and produce a concise summary. "
        "Focus on the main points and keep the summary brief and clear. "
        "Output only the summary text, without extraneous commentary."
    ),
    output_key="summary"  # The summary will be stored in state['summary']
)

# --- Define Sub-Agent 2: Translator ---
translator_agent = LlmAgent(
    name="TranslatorAgent",
    model=MODEL_NAME,
    description="Translates English text to Spanish.",
    instruction=(
        "You are a translation AI. You will be given some text in English, and your task is to translate it accurately into Spanish. "
        "Preserve the meaning of the original text. Output only the translated Spanish text."
        "\n\nText to translate:\n{summary}"
    ),
    output_key="translation"  # The Spanish translation will be stored in state['translation']
)

# --- Define Sub-Agent 3: Reviewer ---
reviewer_agent = LlmAgent(
    name="ReviewerAgent",
    model=MODEL_NAME,
    description="Reviews the translated summary for accuracy and clarity.",
    instruction=(
        "You are an expert bilingual editor. You will be given an English summary and its Spanish translation. "
        "Compare the translation to the original summary for accuracy and completeness. Improve the Spanish text if necessary for clarity or correctness. "
        "If the translation is perfect, you can simply repeat it or confirm it.\n\n"
        "**English Summary:**\n{summary}\n\n"
        "**Spanish Translation (to review):**\n{translation}\n\n"
        "Provide a final corrected Spanish summary as needed, without additional commentary (output only the final Spanish text)."
    ),
    output_key="final_summary"  # The reviewed Spanish summary in state['final_summary']
)

# --- Compose the workflow with a SequentialAgent ---
pipeline_agent = SequentialAgent(
    name="DocSummaryTranslateReviewAgent",
    sub_agents=[summarizer_agent, translator_agent, reviewer_agent],
    description="Executes a sequence of summarization, translation, and review to produce a translated summary of a document."
    # No output_key here – the SequentialAgent will return the final sub-agent's result as the overall result.
)
# For ADK to recognize the agent, assign it to the special `root_agent` variable:
root_agent = pipeline_agent

