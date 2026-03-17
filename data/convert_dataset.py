from datasets import load_dataset
import json

OUTPUT_PATH = "../data/sample_questions.json"

# Load BoolQ dataset (guaranteed to exist)
dataset = load_dataset("boolq", split="train")

converted = []

for i, item in enumerate(dataset):
    question = item["question"]
    answer = "Yes" if item["answer"] else "No"

    converted.append({
        "id": i + 1,
        "question": question,
        "answer": answer
    })

# take first 100
converted = converted[:100]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2)

print(f"Saved {len(converted)} questions to {OUTPUT_PATH}")