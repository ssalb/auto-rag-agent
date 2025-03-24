from sentence_transformers import SentenceTransformer
from numpy import ndarray
import rag_agent.config as config


model = "BAAI/bge-small-en-v1.5"

emb_model = SentenceTransformer(model, device=config.device)

def encode(texts: list[str]) -> list[ndarray]:
    return emb_model.encode(texts)

if __name__ == "__main__":
    text = (
        "'I wish it need not have happened in my time,' said Frodo. 'So do I,' said Gandalf, "
        "'and so do all who live to see such times. But that is not for them to decide. "
        "All we have to decide is what to do with the time that is given us.'"
    )

    print(text)
    result = emb_model.encode([text])
    print(type(result[0]), len(result[0]))