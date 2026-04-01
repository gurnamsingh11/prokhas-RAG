from typing_extensions import TypedDict
from langchain.agents import create_agent

from src.loaders.universal_loader import UniversalDocumentLoader
from src.embeddings.llm_model import get_chat_model


# ── Structured Output Schema (TypedDict) ──────────────────────────────────────
class ClaimInfo(TypedDict):
    account_number: str
    claim_amount: str


# ── Core Extraction Logic ─────────────────────────────────────────────────────
class ClaimExtractor:
    def __init__(self):
        self.loader = UniversalDocumentLoader()
        self.llm = get_chat_model()

        # Create agent with structured output
        self.agent = create_agent(
            model=self.llm,
            tools=[],
            response_format=ClaimInfo,  # <-- TypedDict used here
        )

    def extract_from_file(self, file_path: str) -> ClaimInfo:
        # Step 1: Load document
        docs = self.loader.load(file_path)

        # Step 2: Combine text
        full_text = "\n".join([doc.page_content for doc in docs])

        # Step 3: Prompt LLM
        result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""
Extract the following fields from the document:

1. account_number
2. claim_amount

Return ONLY structured data.

Document:
{full_text}
""",
                    }
                ]
            }
        )

        # Step 4: Return structured response
        return result["structured_response"]
