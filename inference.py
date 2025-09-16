import pyterrier_doc2query
import ir_datasets
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="doc2query/all-t5-base-v1")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    dataset = ir_datasets.load(f"beir/{args.dataset}")
    corpus = []
    for doc in dataset.docs_iter():
        content = getattr(doc, "title", "")
        content = f"{content}\n{doc.text}".strip()
        if not content:
            continue
        corpus.append({"id": doc.doc_id, "text": content})
    doc2query = pyterrier_doc2query.Doc2Query(
        checkpoint=args.model,
        num_samples=args.num_samples,
        batch_size=20,
        verbose=True,
        fast_tokenizer=True,
        device="cuda:1",
    )
    results = doc2query(corpus)
    DE = {result["id"]: result["querygen"].split("\n") for result in results}
    with open(f"{args.output}.json", "w") as f:
        json.dump(DE, f, indent=4)
    print(f"Saved to {args.output}.json")
