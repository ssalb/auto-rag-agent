from smolagents import Tool
from rag_agent.tools.utils.embeddings import encode
from rag_agent.tools.utils.ner import extract_entities
from rag_agent.tools.utils.semantic_search import search_similar_chunks


class TextRetriever(Tool):
    name = "retriever"
    description = (
        "Uses semantic search to retrieve parts of previously indexed documents that could be most relevant to answer your query. "
        "There will always be result unless no document has been indexed yet, but they may not be relevant. "
        "You should always try this tool first, as it might already have the answer you are looking for. "
        "If the results are not relevant, ask the user for more context in form of documents or URLs. "
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question. "
                "Search accuracy is improved if you include named entities in your query that are relevant to it."
            ),
        }
    }
    output_type = "string"

    def __init__(self, max_results: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.max_results = max_results

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        query_entities = extract_entities(query)
        query_embedding = encode([query])[0].tolist()

        # Get twice as many results as needed to allow for filtering
        # and reranking based on named entities
        results = search_similar_chunks(query_embedding, limit=2 * self.max_results)

        filtered_results = results
        if len(query_entities) > 0:
            # Try to rerank and filter results by named entities
            query_ne = set(query_entities.keys())
            scores = [len(query_ne.intersection(result["named_entities"].keys())) for result in results]
            sorted_results = sorted(
                zip(results, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            filtered_results = [
                result
                for result, score in sorted_results
                if score > 0
            ]
            # If no results match the named entities, fall back to the original results
            if len(filtered_results) == 0:
                filtered_results = results

        # only use the top 5 results
        filtered_results = filtered_results[:self.max_results]
        return "\nRetrieved texts:\n" + "".join(
            [
                f"\n\n===== From {result["doc_name"]} =====\n" + result["chunk_text"]
                for result in filtered_results
            ]
        )
