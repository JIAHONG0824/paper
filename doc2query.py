from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import defaultdict
from transformers import set_seed
from tqdm import tqdm
import argparse
import torch
import json
import os

num = 20


def process(batch):
    documents = [x[1] for x in batch]
    input_ids = tokenizer(
        documents, max_length=384, truncation=True, return_tensors="pt", padding=True
    ).input_ids.to("cuda:1")
    outputs = model.generate(
        input_ids, max_length=64, do_sample=True, top_k=10, num_return_sequences=num
    )
    temp = defaultdict(list)
    for i, output in enumerate(outputs):
        query = tokenizer.decode(output, skip_special_tokens=True)
        temp[i // num].append(query)
    return temp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="doc2query/all-t5-base-v1")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:1"
    )
    print(model.dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    batch_size = 32
    with open(args.input_jsonl, "r") as fin:
        for round in range(1, 6):
            set_seed(round)
            batch = []
            with open(f"{args.output_dir}/{round}.jsonl", "w") as fout:
                for line in tqdm(fin):
                    doc_id, document = json.loads(line).values()
                    batch.append((doc_id, document))
                    if len(batch) == batch_size:
                        temp = process(batch)
                        for i in range(len(batch)):
                            fout.write(
                                json.dumps(
                                    {
                                        "id": batch[i][0],
                                        "document": batch[i][1],
                                        "generated_queries": temp[i],
                                    }
                                )
                                + "\n"
                            )
                        batch = []
                if len(batch) > 0:
                    temp = process(batch)
                    for i in range(len(batch)):
                        fout.write(
                            json.dumps(
                                {
                                    "id": batch[i][0],
                                    "document": batch[i][1],
                                    "generated_queries": temp[i],
                                }
                            )
                            + "\n"
                        )
            fin.seek(0)
