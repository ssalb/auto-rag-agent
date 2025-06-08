from smolagents import Tool
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from rag_agent.tools.utils.embeddings import encode
from rag_agent.tools.utils.ner import extract_entities
from rag_agent.tools.utils.semantic_search import bulk_insert_chunks

import logging

logger = logging.getLogger(__name__)


class DocumentIndexer(Tool):
    name = "document_indexer"
    description = (
        "This is a tool that indexes documents for later retrieval. "
        "Only use this tool if a document or a URL has been provided by the user. "
        "If a URL is provided, it must start with 'https://'. "
        "Returns: A string containing informing whether the indexing process succeeded. "
        "Example usage: `print(document_indexer(document_path='https://example.com/my_document.pdf'))`"
    )
    inputs = {
        "document_path": {
            "type": "string",
            "description": "Document to index. It can be a local file path or a URL. ",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.converter = DocumentConverter()
        self.chunker = HybridChunker()

    def forward(self, document_path: str) -> None:

        doc_name = document_path.split("/")[-1]
        response_text = f"Processing {doc_name}...\n"

        try:
            doc = self.converter.convert(document_path).document
        except Exception as e:
            logger.warning(f"Failed to convert document: {e}")
            return response_text + "Failed to convert document"

        try:
            chunk_iter = self.chunker.chunk(dl_doc=doc)

            rows = []
            for chunk in chunk_iter:
                # Using only text for now. More features would depend on the nature of the document
                enriched_text = self.chunker.contextualize(chunk=chunk)
                entities = extract_entities(enriched_text)
                embedding = encode([enriched_text])

                row = {
                    "doc_name": doc_name,
                    "chunk_text": enriched_text,
                    "named_entities": entities,
                    "embedding": embedding[0].tolist(),
                }
                rows.append(row)
        except Exception as e:
            logger.warning(f"Failed to process chuncks: {e}")
            response_text += "Failed to process chuncks. Will try to index the rest of the document.\n"

        try:
            bulk_insert_chunks(rows)
        except Exception as e:
            logger.warning(f"Failed to index document: {e}")
            return response_text + "Failed to index document."

        return response_text + "Document indexed successfully."


if __name__ == "__main__":
    indexer = DocumentIndexer()
    result = indexer.forward("../../README.md")
    print(result)
