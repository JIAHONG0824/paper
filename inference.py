from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import ir_datasets
import json

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

if __name__ == "__main__":
    batch_size = 32
    num_return_sequences = 50

    model_name = "doc2query/all-t5-base-v1"

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    model = torch.compile(model)

    dataset = ir_datasets.load("beir/fiqa")
    corpus = []
    for doc in dataset.docs_iter():
        if not doc.text:
            continue
        corpus.append((doc.doc_id, doc.text))
    print(f"Corpus Size:", len(corpus))

    # batch inference
    h = {}
    for i in tqdm(range(0, len(corpus), batch_size), desc="Generating Queries"):
        batch = corpus[i : i + batch_size]
        batch_ids = [item[0] for item in batch]
        batch_documents = [item[1] for item in batch]

        inputs = tokenizer(
            batch_documents,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_return_sequences,
            )

        all_queries = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, doc_id in enumerate(batch_ids):
            start_index = j * num_return_sequences
            end_index = start_index + num_return_sequences
            h[doc_id] = all_queries[start_index:end_index]

    # Save results
    with open("all-t5-base-v1-DE.json", "w") as f:
        json.dump(h, f, indent=4, ensure_ascii=False)
    print("Finished generating and saved to all-t5-base-v1-DE.json")
