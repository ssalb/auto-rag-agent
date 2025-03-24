import gradio as gr
from rag_agent.tools.indexer import DocumentIndexer

def dummy(message, history):
    indexer = DocumentIndexer()
    for doc in message["files"]:
        result = indexer.forward(doc)
        break
    return result
    

demo = gr.ChatInterface(
    fn=dummy, 
    type="messages", 
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["text", ".pdf", ".docx", ".md"], sources=["upload", "microphone"])
)

demo.launch()