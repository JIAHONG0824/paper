import pyterrier_doc2query
import ir_datasets
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="doc2query/all-t5-base-v1",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
    )
    args = parser.parse_args()
    dataset = ir_datasets.load(f"beir/{args.dataset}")
    print("Corpus size:", dataset.docs_count())
    corpus = []
    for doc in dataset.docs_iter():
        content = getattr(doc, "title", "")
        content = f"{content}\n{doc.text}".strip()
        if not content:
            continue
        corpus.append({"id": doc.doc_id, "text": content})
    print("Number of valid documents:", len(corpus))
    doc2query = pyterrier_doc2query.Doc2Query(
        checkpoint=args.model,
        num_samples=args.num_samples,
        batch_size=40,
        verbose=True,
        fast_tokenizer=True,
        device="cuda:1",
    )
    results = doc2query(corpus)
    DE = {result["id"]: result["querygen"].split("\n") for result in results}
    with open(f"all-t5-base-v1_{args.dataset}_gen{args.num_samples}.json", "w") as f:
        json.dump(DE, f)
    print("Saved to:", f"all-t5-base-v1_{args.dataset}_gen{args.num_samples}.json")
