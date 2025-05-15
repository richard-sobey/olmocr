"""Prompt builder that mirrors the legacy pipeline.build_page_query() logic."""
import base64
from olmocr.prompts import build_finetuning_prompt


def build_prompt(png_bytes: bytes, anchor_text: str, max_tokens: int = 3000) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_finetuning_prompt(anchor_text)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
