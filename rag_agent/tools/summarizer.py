from typing import Optional
from smolagents import Tool, Model


class SummarizerTool(Tool):
    name = "summarizer_tool"
    description = (
        "Summarizes the content of a text or a set of document chunks. "
        "Optionally, you can provide a short question or query to guide the summarization. "
        "If you don't provide a question, the tool will summarize the content in a general way. "
    )
    inputs = {
        "text": {
            "type": "string",
            "description": ("The text to summarize."),
        },
        "query": {
            "type": "string",
            "description": (
                "Optional. If provided, the summarization will be guided by this question or query. "
                "If you don't have a specific question, skip this field. "
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, text: str, query: Optional[str] = None) -> str:
        if not isinstance(text, str):
            raise TypeError("The text to summarize must be a string")
        if not isinstance(query, (str, type(None))):
            raise TypeError("The query must be a string or None")

        prompt = f"Summarize the following text:\n\n{text}\n"
        if query:
            prompt += f"\n\nPlease focus on the following question: {query}\n"

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        response = self.model(messages=message).content
        return response.strip()

