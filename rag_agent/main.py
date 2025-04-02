import gradio as gr
from rag_agent.tools.indexer import DocumentIndexer
from rag_agent.tools.retriever import TextRetriever
from smolagents import HfApiModel, CodeAgent

from rag_agent.db import init_db

init_db()


def chat(message, history):
    indexing_tool = DocumentIndexer()
    retriever_tool = TextRetriever()

    agent = CodeAgent(
        tools=[indexing_tool, retriever_tool],
        model=HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
        max_steps=4,
        verbosity_level=2,
    )

    response = agent.run(
        task=message["text"],
        additional_args=(
            {
                "input_document_paths": message["files"] if message["files"] else [],
            }
        ),
    )

    return response


demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    multimodal=True,
    textbox=gr.MultimodalTextbox(
        file_count="multiple",
        file_types=["text", ".pdf", ".docx", ".md"],
        sources=["upload", "microphone"],
    ),
)

demo.launch()
