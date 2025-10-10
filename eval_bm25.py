from pyserini.search.lucene import LuceneSearcher
from collections import defaultdict
import ir_datasets
import numpy as np
import argparse


def evaluate_metrics(results, qrels, k):
    rr = []
    recall = []
    for qid, hits in results.items():
        hits = [hit.docid for hit in hits if hit.docid != qid]
        gt = qrels[qid]
        recall.append(len(set(hits) & set(gt)) / len(gt))
        s = 0
        for rank, hit in enumerate(hits[:10]):
            if hit in gt:
                s = 1 / (rank + 1)
                break
        rr.append(s)
    print("MRR@10:", round(np.mean(rr), 4))
    print(f"Recall@{k}:", round(np.mean(recall), 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, required=True, help="index directory")
    parser.add_argument("--dataset", type=str, required=True, help="ir_datasets name")
    parser.add_argument(
        "--k", type=int, default=1000, help="Number of top documents to retrieve"
    )
    args = parser.parse_args()

    searcher = LuceneSearcher(index_dir=args.index_dir)
    dataset = ir_datasets.load(args.dataset)

    qrels = defaultdict(list)
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id].append(qrel.doc_id)

    queries = []
    qids = []
    for query in dataset.queries_iter():
        queries.append(query.text)
        qids.append(query.query_id)

    results = searcher.batch_search(queries, qids, k=args.k, threads=8)
    evaluate_metrics(results, qrels, args.k)
    # with open("results.text", "w") as f:
    #     for qid, hits in results.items():
    #         hits = [hit for hit in hits if hit.docid != qid]
    #         for rank, hit in enumerate(hits):
    #             f.write(f"{qid} Q0 {hit.docid} {rank+1} {hit.score} Anserini\n")
