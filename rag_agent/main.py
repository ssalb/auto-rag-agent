import gradio as gr
from rag_agent.tools.indexer import DocumentIndexer
from rag_agent.tools.retriever import TextRetriever
from rag_agent.tools.summarizer import SummarizerTool
from smolagents import HfApiModel, CodeAgent, MLXModel

from rag_agent.db import init_db

init_db()


def chat(message, history):
    indexing_tool = DocumentIndexer()
    search_tool = TextRetriever()

    # model = MLXModel(model_id="mlx-community/Meta-Llama-3.1-8B-Instruct-bf16")
    model = MLXModel(model_id="mlx-community/Qwen2.5-Coder-7B-Instruct-bf16")
    # model = HfApiModel(model_id="Qwen/Qwen2.5-72B-Instruct")

    summarizer_tool = SummarizerTool(model=model)
    
    agent = CodeAgent(
        tools=[search_tool, indexing_tool, summarizer_tool],
        model=model,
        max_steps=4,
        verbosity_level=2,
    )

    agent.prompt_templates["system_prompt"] = (
        agent.prompt_templates["system_prompt"]
        + " Remember to use the tools when needed, but more importantly, don't use them when not needed. "
        + "Somtimes you can answer directly without using any tool. "
    )

    response = agent.run(
        task=message["text"],
        additional_args=(
            {
                "input_document_paths": (
                    message["files"] if message["files"] else ["No documents provided"]
                ),
                "conversation_history": history if history else [],
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
