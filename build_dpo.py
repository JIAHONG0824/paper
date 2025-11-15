from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import torch
import json

prefix = {
    "facebook/contriever": None,
    "facebook/contriever-msmarco": None,
    "BAAI/bge-base-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages:",
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
    )
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    model = SentenceTransformer(args.model, device="cuda:1", prompts=prefix[args.model])

    datas = []
    with open(args.input_jsonl, "r") as f:
        for line in f:
            item = json.loads(line)
            document, query, generated_queries = (
                item["document"],
                item["query"],
                item["generated_queries"],
            )
            datas.append((document, query, generated_queries))
    with open(args.output_jsonl, "w") as f:
        for document, query, generated_queries in tqdm(datas):
            query_embeddings = model.encode_query(query, convert_to_tensor=True)
            document_embeddings = model.encode_document(
                document, convert_to_tensor=True
            )
            baseline = document_embeddings @ query_embeddings

            temp = [f"{document}\n{qg}" for qg in generated_queries]
            temp_embeddings = model.encode_document(temp, convert_to_tensor=True)

            scores = temp_embeddings @ query_embeddings
            gains = scores - baseline

            best_idx = torch.argmax(gains)
            worst_idx = torch.argmin(gains)
            if gains[best_idx] <= 0 or gains[worst_idx] >= 0:
                continue

            f.write(
                json.dumps(
                    {
                        "prompt": document,
                        "chosen": generated_queries[best_idx],
                        "rejected": generated_queries[worst_idx],
                    }
                )
                + "\n"
            )
