from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import rag_agent.config as config

model = "elastic/distilbert-base-uncased-finetuned-conll03-english"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForTokenClassification.from_pretrained(model)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="first",
    device=config.device,
)


def extract_entities(text: str) -> dict[str]:
    results = ner_pipeline(text)
    return {entity["word"]: entity["entity_group"] for entity in results}


if __name__ == "__main__":
    text = (
        "'I wish it need not have happened in my time,' said Frodo. 'So do I,' said Gandalf, "
        "'and so do all who live to see such times. But that is not for them to decide. "
        "All we have to decide is what to do with the time that is given us.'"
    )

    print(text)
    entities = extract_entities(text)
    print(entities)
