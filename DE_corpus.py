import pyterrier_doc2query
import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="doc2query/all-t5-base-v1")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="input jsonl file, each line is {doc_id:...,document:...}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    datas = []
    with open(args.input_jsonl, "r") as f:
        for line in f:
            datas.append(json.loads(line))
    print("Total documents:", len(datas))
    doc2query = pyterrier_doc2query.Doc2Query(
        checkpoint=args.model,
        num_samples=20,
        batch_size=20,
        doc_attr="document",
        verbose=True,
        fast_tokenizer=True,
        device="cuda:1",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(1, 4):
        results = doc2query(datas)
        with open(f"{args.output_dir}/{i}.jsonl", "w") as f:
            for result in results:
                f.write(
                    json.dumps(
                        {
                            "doc_id": result["doc_id"],
                            "document": result["document"],
                            "querygen": result["querygen"].split("\n"),
                        }
                    )
                    + "\n"
                )
