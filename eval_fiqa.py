from sentence_transformers import SentenceTransformer
from pyserini.search.faiss import FaissSearcher
import pandas as pd
import ir_datasets
import argparse
import ir_measures
from ir_measures import *

prefix = {
    "facebook/contriever": None,
    "facebook/contriever-msmarco": None,
    "BAAI/bge-base-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages:",
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "facebook/contriever",
            "facebook/contriever-msmarco",
            "BAAI/bge-base-en-v1.5",
        ],
    )
    args = parser.parse_args()

    model = SentenceTransformer(
        args.model,
        device="cuda:1",
        prompts=prefix[args.model],
        default_prompt_name="query",
    )
    searcher = FaissSearcher(index_dir="fiqa", query_encoder=model)

    # dev set
    dev = ir_datasets.load("beir/fiqa/dev")
    qrels = dev.qrels_iter()

    queries = []
    qids = []
    for query in dev.queries_iter():
        queries.append(query.text)
        qids.append(query.query_id)

    results = searcher.batch_search(queries, qids, k=500, threads=8)
    for qid, hits in results.items():
        seen = set()
        unique_hits = []
        for hit in hits:
            if hit.docid not in seen:
                unique_hits.append(hit)
                seen.add(hit.docid)
            if len(unique_hits) == 100:
                results[qid] = unique_hits
                break
    run = pd.DataFrame(
        [
            {"query_id": qid, "doc_id": hit.docid, "score": hit.score}
            for qid, hits in results.items()
            for hit in hits
        ]
    )
    temp = ir_measures.calc_aggregate([nDCG @ 10, R @ 100], qrels, run)
    print("Dev Set Results:")
    for k, v in temp.items():
        print(f"{k}: {v:.4f}")

    # test set
    test = ir_datasets.load("beir/fiqa/test")
    qrels = test.qrels_iter()

    queries = []
    qids = []
    for query in test.queries_iter():
        queries.append(query.text)
        qids.append(query.query_id)

    results = searcher.batch_search(queries, qids, k=500, threads=8)
    for qid, hits in results.items():
        seen = set()
        unique_hits = []
        for hit in hits:
            if hit.docid not in seen:
                unique_hits.append(hit)
                seen.add(hit.docid)
            if len(unique_hits) == 100:
                results[qid] = unique_hits
                break
    run = pd.DataFrame(
        [
            {"query_id": qid, "doc_id": hit.docid, "score": hit.score}
            for qid, hits in results.items()
            for hit in hits
        ]
    )
    temp = ir_measures.calc_aggregate([nDCG @ 10, R @ 100], qrels, run)
    print("Test Set Results:")
    for k, v in temp.items():
        print(f"{k}: {v:.4f}")
