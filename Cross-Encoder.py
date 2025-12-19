from sentence_transformers import CrossEncoder
from tqdm import tqdm
import argparse
import torch
import json

if __name__ == "__main__":
    model = CrossEncoder(
        "BAAI/bge-reranker-base", device="cuda:1", activation_fn=torch.nn.Identity()
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    with open(args.input_jsonl, "r") as f, open(args.output_jsonl, "w") as fout:
        for line in tqdm(f):
            data = json.loads(line)
            document, query, generated_queries = (
                data["document"],
                data["query"],
                data["generated_queries"],
            )
            pairs = [(qg, document) for qg in generated_queries]
            rewards = model.predict(pairs, convert_to_tensor=True)
            maximum = torch.argmax(rewards)
            minimum = torch.argmin(rewards)
            fout.write(
                json.dumps(
                    {
                        "prompt": document,
                        "chosen": generated_queries[maximum],
                        "rejected": generated_queries[minimum],
                    }
                )
                + "\n"
            )
