from tqdm import tqdm
import ir_datasets
import subprocess
import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="0: disable document expansion; 1: enable document expansion for seen docs; 2: enable document expansion for all docs",
    )
    args = parser.parse_args()

    dataset = ir_datasets.load(f"beir/{args.dataset}")
    corpus = []
    for doc in dataset.docs_iter():
        content = getattr(doc, "title", "")
        content = f"{content}\n{doc.text}".strip()
        if not content:
            continue
        corpus.append((doc.doc_id, content))

    train = ir_datasets.load(f"beir/{args.dataset}/train")
    dev = ir_datasets.load(f"beir/{args.dataset}/dev")
    test = ir_datasets.load(f"beir/{args.dataset}/test")

    seen = set()
    for qrel in train.qrels_iter():
        seen.add(qrel.doc_id)
    for qrel in dev.qrels_iter():
        seen.add(qrel.doc_id)
    for qrel in test.qrels_iter():
        seen.add(qrel.doc_id)

    with open(args.json, "r") as f:
        DE = json.load(f)

    os.makedirs("corpus", exist_ok=True)

    with open("corpus/corpus.jsonl", "w") as f:
        for id, contents in tqdm(corpus):
            if args.mode == 0:
                generated_queries = []
            elif args.mode == 1:
                if id in seen:
                    generated_queries = DE.get(id, [])
                else:
                    generated_queries = []
            else:
                generated_queries = DE.get(id, [])
            f.write(
                json.dumps(
                    {
                        "id": id,
                        "contents": "\n".join(generated_queries + [contents]),
                    }
                )
            )
    command = [
        "python",
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        "corpus",
        "--index",
        "BM25",
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "1",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    subprocess.run(command)
