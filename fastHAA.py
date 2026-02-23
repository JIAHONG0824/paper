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
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()

    model = SentenceTransformer(
        args.model, device=args.device, prompts=prefix[args.model]
    )
    if args.model == "facebook/contriever-msmarco":
        model.add_module(str(len(model)), models.Normalize())
    with open(args.input_jsonl, "r") as f, open(args.output_jsonl, "w") as out:
        for line in tqdm(f):
            item = json.loads(line)
            id, document, generated_queries = (
                item["id"],
                item["document"],
                item["generated_queries"],
            )
            d_emb = model.encode_document(document, convert_to_tensor=True)
            anchor = model.encode_query(generated_queries, convert_to_tensor=True).mean(
                dim=0
            )
            d_emb = F.normalize(d_emb, p=2, dim=0)
            anchor = F.normalize(anchor, p=2, dim=0)
            s = (d_emb * anchor).sum()
            mixed = s * d_emb + (1 - s) * anchor
            mixed = F.normalize(mixed, p=2, dim=0)

            out.write(
                json.dumps(
                    {
                        "id": id,
                        "vector": mixed.cpu().tolist(),
                    }
                )
                + "\n"
            )
