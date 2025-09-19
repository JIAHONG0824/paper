from tqdm import tqdm
import subprocess
import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--index_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs("corpus", exist_ok=True)

    with open(args.input_jsonl, "r") as fin, open("corpus/corpus.jsonl", "w") as fout:
        for line in tqdm(fin):
            data = json.loads(line)
            doc_id = data["doc_id"]
            document = data["document"]
            expanded_queries = data.get("querygen", [])
            fout.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "contents": "\n".join([document] + expanded_queries[: args.k]),
                    }
                )
                + "\n"
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
        f"{args.index_dir}",
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "8",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    subprocess.run(command)
