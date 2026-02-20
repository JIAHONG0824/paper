from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import subprocess
import argparse
import json
import os

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
        choices=[
            "facebook/contriever",
            "facebook/contriever-msmarco",
            "BAAI/bge-base-en-v1.5",
        ],
    )
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--input_jsonl", type=str, required=True)
    args = parser.parse_args()

    model = SentenceTransformer(args.model, device="cuda:1", prompts=prefix[args.model])

    os.makedirs("corpus", exist_ok=True)

    datas = []

    with open(args.input_jsonl, "r") as f:
        for line in f:
            item = json.loads(line)
            id, document, generated_queries = (
                item["id"],
                item["document"],
                item.get("generated_queries", []),
            )
            # Here we combine the document with its generated queries
            if args.model == "facebook/contriever":
                text = " ".join(generated_queries[: args.k] + [document])
            else:
                text = " ".join([document] + generated_queries[: args.k])
            datas.append((id, text))

    with open("corpus/corpus.jsonl", "w") as f:
        for i in tqdm(range(0, len(datas), 64)):
            ids = [x[0] for x in datas[i : i + 64]]
            texts = [x[1] for x in datas[i : i + 64]]
            vectors = model.encode_document(texts, convert_to_tensor=False)
            for id, vector in zip(ids, vectors):
                f.write(
                    json.dumps(
                        {
                            "id": id,
                            "vector": vector.tolist(),
                        }
                    )
                    + "\n"
                )

    command = [
        "python",
        "-m",
        "pyserini.index.faiss",
        "--input",
        "corpus",
        "--output",
        args.index_dir,
    ]
    subprocess.run(command)
