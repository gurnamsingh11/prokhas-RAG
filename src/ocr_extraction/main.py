import logging
from langchain_core.documents import Document
from src.config.config import settings
logger = logging.getLogger(__name__)
from openai import OpenAI
import base64

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=settings.HUGGINGFACEHUB_API_TOKEN,
)


PROMPT = """You are a strict OCR engine.

Rules:
- Extract ONLY text that is explicitly visible in the image.
- Do NOT infer, guess, complete, or correct text.
- Do NOT add punctuation, formatting, or missing words.
- Do NOT describe the image.
- Do NOT explain anything.
- If a word is unclear, output [UNCERTAIN].
- Preserve exact casing, spacing, and line breaks.
- Output ONLY the extracted text.
- If no text is visible, output: [NO_TEXT]

Violation of any rule is an error."""


def encode_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def extract_text_from_image(image_path: str, image_name: str, prompt=PROMPT) -> Document:
    try:
        base64_image = encode_image(image_path)

        completion = client.chat.completions.create(
            model="moonshotai/Kimi-K2.5:novita",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        )

        text = completion.choices[0].message.content.strip()

    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.error("OCR failed for %s: %s", image_name, exc)
        text = f"[OCR failed for {image_name}: {exc}]"

    return Document(
        page_content=text,
        metadata={
            "source": image_name,
            "page_label": "",
            "source_type": "image",
            "do_not_split": True,
        },
    )
