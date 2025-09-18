import pyterrier_doc2query
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="doc2query/all-t5-base-v1")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="input jsonl file, each line is {id:...,document:...}",
    )
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    corpus = []
    with open(args.input_jsonl, "r") as f:
        for line in f:
            corpus.append(json.loads(line))

    doc2query = pyterrier_doc2query.Doc2Query(
        checkpoint=args.model,
        num_samples=20,
        batch_size=20,
        doc_attr="document",
        verbose=True,
        fast_tokenizer=True,
        device="cuda:1",
    )
    results = doc2query(corpus)
    with open(args.output_jsonl, "w") as f:
        for result in results:
            f.write(
                json.dumps(
                    {
                        "id": result["id"],
                        "document": result["document"],
                        "querygen": result["querygen"].split("\n"),
                    }
                )
                + "\n"
            )
