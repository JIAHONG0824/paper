from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import set_seed
from tqdm import tqdm
import argparse
import torch
import json
import os

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
    model.eval()
    print(model.dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    datas = []
    with open(args.input_jsonl, "r") as fin:
        for line in fin:
            item = json.loads(line)
            id, document = item["id"], item["document"]
            datas.append((id, document))

    batch_size = 128
    nums = 5
    for round in range(1, 6):
        set_seed(round)
        with open(f"{args.output_dir}/{round}.jsonl", "w") as fout:
            for i in tqdm(range(0, len(datas), batch_size)):
                ids = [x[0] for x in datas[i : i + batch_size]]
                texts = [x[1] for x in datas[i : i + batch_size]]
                input_ids = tokenizer(
                    texts,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                    padding=True,
                ).input_ids.to(model.device)
                outputs = model.generate(
                    input_ids,
                    max_length=64,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=nums,
                )
                queries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for j in range(0, len(queries), nums):
                    fout.write(
                        json.dumps(
                            {
                                "id": ids[j // nums],
                                "document": texts[j // nums],
                                "generated_queries": queries[j : j + nums],
                            }
                        )
                        + "\n"
                    )
