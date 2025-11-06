from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import subprocess
import argparse
import json
import os

prefix = {
    "facebook/contriever-msmarco": None,
    "BAAI/bge-base-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages:",
    },
    "intfloat/multilingual-e5-base": {"query": "query: ", "document": "passage: "},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "facebook/contriever-msmarco",
            "BAAI/bge-base-en-v1.5",
            "intfloat/multilingual-e5-base",
        ],
    )
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--input_jsonl", type=str, required=True)
    args = parser.parse_args()

    model = SentenceTransformer(args.model, device="cuda:1", prompts=prefix[args.model])

    os.makedirs("corpus", exist_ok=True)

    batch = []
    batch_size = 64

    with open(args.input_jsonl, "r") as fin, open("corpus/corpus.jsonl", "w") as fout:
        for line in tqdm(fin):
            item = json.loads(line)
            doc_id, document, querygen = (
                item["doc_id"],
                item["document"],
                item.get("querygen", []),
            )
            batch.append((doc_id, document))
            if len(batch) == batch_size:
                doc_ids = [x[0] for x in batch]
                texts = [x[1] for x in batch]
                vectors = model.encode_document(texts, convert_to_tensor=False)
                for doc_id, vector in zip(doc_ids, vectors):
                    fout.write(
                        json.dumps(
                            {
                                "id": doc_id,
                                "vector": vector.tolist(),
                            }
                        )
                        + "\n"
                    )
                batch = []
        if len(batch) > 0:
            doc_ids = [x[0] for x in batch]
            texts = [x[1] for x in batch]
            vectors = model.encode_document(texts, convert_to_tensor=False)
            for doc_id, vector in zip(doc_ids, vectors):
                fout.write(
                    json.dumps(
                        {
                            "id": doc_id,
                            "vector": vector.tolist(),
                        }
                    )
                    + "\n"
                )
        fin.seek(0)
        if args.k != 0:
            for line in tqdm(fin):
                item = json.loads(line)
                doc_id, querygen = item["doc_id"], item.get("querygen", [])
                queries = querygen[: args.k]
                vectors = model.encode_document(queries, convert_to_tensor=True)
                for vec in vectors:
                    fout.write(
                        json.dumps(
                            {
                                "id": doc_id,
                                "vector": vec.tolist(),
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
