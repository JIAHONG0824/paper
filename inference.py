import pyterrier_doc2query
import ir_datasets
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="doc2query/all-t5-base-v1")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()
    dataset = ir_datasets.load(f"beir/{args.dataset}")
    print("Corpus size:", dataset.docs_count())
    corpus = []
    for doc in dataset.docs_iter():
        try:
            corpus.append({"id": doc.doc_id, "text": f"{doc.title}\n{doc.text}"})
        except AttributeError:
            if not doc.text:
                continue
            corpus.append({"id": doc.doc_id, "text": doc.text})
    print("non-empty docs:", len(corpus))
    doc2query = pyterrier_doc2query.Doc2Query(
        checkpoint=args.model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        verbose=True,
        fast_tokenizer=True,
        device="cuda:1",
    )
    DE = doc2query(corpus)
    with open(
        f"{args.model.split('/')[-1]}_{args.dataset}_querygen:{args.num_samples}.jsonl",
        "w",
    ) as f:
        for item in DE:
            f.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "text": item["text"],
                        "querygen": item["querygen"].split("\n"),
                    }
                )
                + "\n"
            )
    print("Done!")
