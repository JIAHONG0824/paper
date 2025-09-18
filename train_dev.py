from sklearn.model_selection import train_test_split
import pyterrier_doc2query
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="doc2query/all-t5-base-v1")
    parser.add_argument(
        "--corpus_jsonl",
        type=str,
        required=True,
        help="input jsonl file, each line is {document:...,query:...}",
    )
    args = parser.parse_args()

    datas = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            document, query = json.loads(line).values()
            datas.append({"document": document, "query": query})
    doc2query = pyterrier_doc2query.Doc2Query(
        checkpoint=args.model,
        num_samples=20,
        batch_size=20,
        doc_attr="document",
        verbose=True,
        fast_tokenizer=True,
        device="cuda:1",
    )
    results = doc2query(datas)
    train, dev = train_test_split(results, test_size=0.1, random_state=42)

    with open("train_DE.jsonl", "w") as f:
        for result in train:
            document, query, querygen = (
                result["document"],
                result["query"],
                result["querygen"],
            )
            f.write(
                json.dumps(
                    {
                        "document": document,
                        "query": query,
                        "querygen": querygen.split("\n"),
                    }
                )
                + "\n"
            )
    with open("dev_DE.jsonl", "w") as f:
        for result in dev:
            document, query, querygen = (
                result["document"],
                result["query"],
                result["querygen"],
            )
            f.write(
                json.dumps(
                    {
                        "document": document,
                        "query": query,
                        "querygen": querygen.split("\n"),
                    }
                )
                + "\n"
            )
    print("Done")
