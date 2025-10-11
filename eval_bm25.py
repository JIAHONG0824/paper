from pyserini.search.lucene import LuceneSearcher
from collections import defaultdict
import ir_datasets
import numpy as np
import argparse


def evaluate_metrics(results, qrels, k):
    rr = []
    ndcg = []
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

        # --- 以下是新增的 NDCG@10 計算 ---

        # 1. 計算 DCG@10 (Discounted Cumulative Gain)
        dcg_score = 0
        top_10_hits = hits[:10]
        for rank, hit in enumerate(top_10_hits):
            if hit in gt:
                # 如果命中，加上它的分數。分數會因排名(rank)而折損
                # rank 從 0 開始，所以要加 2
                dcg_score += 1 / np.log2(rank + 2)

        # 2. 計算 IDCG@10 (Ideal Discounted Cumulative Gain)
        # 理想情況是所有相關文件排在最前面
        idcg_score = 0
        # 只考慮最多 10 個相關文件，因為我們計算的是 @10
        num_relevant_in_gt = min(len(gt), 10)
        for i in range(num_relevant_in_gt):
            idcg_score += 1 / np.log2(i + 2)

        # 3. 計算 NDCG@10
        # 如果 IDCG 為 0 (代表這個 query 沒有相關文件)，則 NDCG 為 0
        ndcg_score = dcg_score / idcg_score if idcg_score > 0 else 0
        ndcg.append(ndcg_score)

    print("MRR@10:", round(np.mean(rr), 4))
    print("nDCG@10:", round(np.mean(ndcg), 4))
    print(f"R@{k}:", round(np.mean(recall), 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--k", type=int, default=1000)
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
