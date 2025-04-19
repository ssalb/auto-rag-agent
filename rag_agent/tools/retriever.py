from typing import Optional
from smolagents import Tool
from rag_agent.tools.utils.embeddings import encode
from rag_agent.tools.utils.ner import extract_entities
from rag_agent.tools.utils.semantic_search import search_similar_chunks


class TextRetriever(Tool):
    name = "search_tool"
    description = (
        "Uses semantic search to search parts of previously indexed documents that could be most relevant to answer your query. "
        "If you need a tool for your task, you should always try this tool first, as it might already have the answer you are looking for. "
        "If the results are not relevant, ask the user for more context in form of documents or URLs."
        "Returns: A string containing the retrieved text chunks. "
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The query to perform. This should be semantically meaningful avoiding single keywords. "
                "Use the affirmative form rather than a question. "
                "Search accuracy is improved if you include named entities in your query that are relevant to it."
            ),
        },
        "doc_name": {
            "type": "string",
            "description": (
                "Optional. If provided, the search will be limited to this document. "
                f"If you don't have the document name, skip this field. The {name} tool always returns the document name "
                "for each chunk, so you can use it to improve the search results in a subsequent step."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, max_results: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.max_results = max_results

    def forward(self, query: str, doc_name: Optional[str] = None) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        query_entities = extract_entities(query)
        query_embedding = encode([query])[0].tolist()

        # Get twice as many results as needed to allow for filtering and reranking
        # based on named entities
        results = search_similar_chunks(
            query_embedding, limit=2 * self.max_results, doc_scope=doc_name
        )

        if doc_name is not None:
            # Filter results by document name
            results = self.__filter_by_doc_name(results, doc_name)

        # Try to rerank and filter results by named entities
        results = self.__rerank_by_named_entities(results, query_entities)

        # only use the top N results
        results = results[: self.max_results]
        return "\nRetrieved texts:\n" + "".join(
            [
                f"\n\n<document_chunk> \nDocument of Origin: {result["doc_name"]}\n=====\n"
                + result["chunk_text"]
                + "\n</document_chunk>"
                for result in results
            ]
        )

    def __filter_by_doc_name(self, results, doc_scope):
        if doc_scope is None:
            return results
        return [result for result in results if result["doc_name"] == doc_scope]

    def __rerank_by_named_entities(self, results, query_entities):
        if len(query_entities) == 0:
            return results
        query_ne = set(query_entities.keys())
        scores = [
            len(query_ne.intersection(result["named_entities"].keys()))
            for result in results
        ]
        sorted_results = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [res[0] for res in sorted_results]
