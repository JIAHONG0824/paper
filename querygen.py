from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import set_seed
from tqdm import tqdm
import argparse
import torch
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="doc2query/all-t5-base-v1")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:1"
    )
    model.eval()
    print(model.dtype)

    datas = []
    with open(args.input_jsonl, "r") as fin:
        for line in fin:
            item = json.loads(line)
            document, query = item["document"], item["query"]
            datas.append((document, query))

    n = 20
    batch_size = 32
    set_seed(42)
    with open(args.output_jsonl, "w") as fout:
        for i in tqdm(range(0, len(datas), batch_size)):
            documents = [x[0] for x in datas[i : i + batch_size]]
            original_queries = [x[1] for x in datas[i : i + batch_size]]
            input_ids = tokenizer(
                documents,
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
                num_return_sequences=n,
            )
            generated_queries = tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            for j in range(0, len(generated_queries), n):
                fout.write(
                    json.dumps(
                        {
                            "document": documents[j // n],
                            "query": original_queries[j // n],
                            "generated_queries": generated_queries[j : j + n],
                        }
                    )
                    + "\n"
                )
