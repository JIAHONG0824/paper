from sentence_transformers import SentenceTransformer
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from datasets import load_dataset
import pandas as pd
import argparse
import time

prefix = {
    "facebook/contriever": None,
    "facebook/contriever-msmarco": None,
    "BAAI/bge-base-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages:",
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BM25")
    args = parser.parse_args()

    if args.model != "BM25":
        model = SentenceTransformer(
            args.model,
            device="cuda:1",
            prompts=prefix[args.model],
            default_prompt_name="query",
        )
        searcher = FaissSearcher(index_dir="hc3", query_encoder=model)
    else:
        searcher = LuceneSearcher(index_dir="hc3")

    ds = load_dataset("mteb/HC3FinanceRetrieval", "qrels")
    qrels = pd.DataFrame(ds["test"])
    qrels = qrels.rename(
        columns={"query-id": "query_id", "corpus-id": "doc_id", "score": "relevance"}
    )

    queries = []
    qids = []
    ds = load_dataset("mteb/HC3FinanceRetrieval", "queries")

    for item in ds["test"]:
        queries.append(item["text"])
        qids.append(item["id"])

    start = time.time()
    results = searcher.batch_search(queries, qids, k=100, threads=8)
    end = time.time()
    print(f"搜尋 {len(queries)} 筆查詢花費時間: {end - start:.2f} 秒")

    run = pd.DataFrame(
        [
            {"query_id": qid, "doc_id": hit.docid, "score": hit.score}
            for qid, hits in results.items()
            for hit in hits
        ]
    )
    import ir_measures
    from ir_measures import *

    temp = ir_measures.calc_aggregate([nDCG @ 10, R @ 100], qrels, run)
    for k, v in temp.items():
        print(f"{k}: {v:.4f}")
