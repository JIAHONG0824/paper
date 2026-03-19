from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import json

prefix = {
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
        choices=["facebook/contriever-msmarco", "BAAI/bge-base-en-v1.5"],
    )
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1)
    args = parser.parse_args()
    alpha = args.alpha
    model = SentenceTransformer(args.model, device="cuda", prompts=prefix[args.model])
    if args.model == "facebook/contriever-msmarco":
        model.add_module(str(len(model)), models.Normalize())
    datas = []
    with open(args.input_jsonl, "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            id, document, generated_queries = (
                item["id"],
                item["document"],
                item["generated_queries"],
            )
            d_emb = model.encode_document(document, convert_to_tensor=True)
            mean = model.encode_query(generated_queries, convert_to_tensor=True).mean(
                dim=0
            )
            d_emb = alpha * d_emb + mean
            d_emb = F.normalize(d_emb, p=2, dim=0)
            datas.append(
                {
                    "id": id,
                    "vector": d_emb.cpu().tolist(),
                }
            )
    with open("corpus/corpus.jsonl", "w") as f:
        for data in datas:
            f.write(json.dumps(data) + "\n")
