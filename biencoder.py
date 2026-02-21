from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import json

prefix = {
    "intfloat/e5-base-v2": {
        "query": "query: ",
        "document": "passage: ",
    },
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
        choices=["intfloat/e5-base-v2", "BAAI/bge-base-en-v1.5"],
    )
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    model = SentenceTransformer(
        args.model, device=args.device, prompts=prefix[args.model]
    )

    with open(args.input_jsonl, "r") as f, open(
        f"{args.model.split('/')[0]}.jsonl", "w"
    ) as out:
        for line in tqdm(f):
            item = json.loads(line)
            document, query, generated_queries = (
                item["document"],
                item["query"],
                item["generated_queries"],
            )
            d_emb = model.encode_document(document)
            q_emb = model.encode_query(query)
            # s0 = s(d,q)
            s0 = model.similarity(d_emb, q_emb)
            expanded_document = [f"{document}\n{gq}" for gq in generated_queries]
            expanded_d_embs = model.encode_document(expanded_document)
            # s1 = s(d+gq,q)
            s1 = model.similarity(expanded_d_embs, q_emb)
            # reward = s(d+gq,q) - s(d,q)
            reward = s1 - s0
            mx_idx = reward.argmax()
            mn_idx = reward.argmin()
            if reward[mx_idx] <= 0 or reward[mn_idx] >= 0:
                continue
            out.write(
                json.dumps(
                    {
                        "prompt": document,
                        "chosen": generated_queries[mx_idx],
                        "rejected": generated_queries[mn_idx],
                    }
                )
                + "\n"
            )
