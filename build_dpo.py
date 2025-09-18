from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["train", "dev"],
        required=True,
    )
    args = parser.parse_args()

    model = SentenceTransformer(args.model, device="cuda:1")

    datas = []
    with open(f"{args.type}_DE.jsonl", "r") as f:
        for line in f:
            document, query, querygen = json.loads(line).values()
            datas.append((document, query, querygen))
    with open(f"{args.type}-dpo.jsonl", "w") as f:
        for document, query, querygen in tqdm(datas):
            document_embeddings = model.encode_document(document)
            query_embeddings = model.encode_query(query)
            baseline = model.similarity(query_embeddings, document_embeddings).item()
            new_document = [
                f"{document}\n{generated_query}" for generated_query in querygen
            ]
            new_document_embeddings = model.encode_document(new_document)
            max_gain = float("-inf")
            gains = []
            for i, (generated_query, new_document_embedding) in enumerate(
                zip(querygen, new_document_embeddings)
            ):
                score = model.similarity(
                    query_embeddings, new_document_embedding
                ).item()
                gain = score - baseline
                gains.append((gain, generated_query))
            max_gain = max(gains, key=lambda x: x[0])[0]
            if max_gain <= 0:
                continue
            gains.sort(key=lambda x: x[0], reverse=True)
            chosen = gains[0][1]
            rejected = gains[-1][1]
            f.write(
                json.dumps(
                    {
                        "prompt": document,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )
                + "\n"
            )
