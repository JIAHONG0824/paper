from sentence_transformers import SentenceTransformer, CrossEncoder
from pyserini.search.faiss import FaissSearcher
from ir_measures import nDCG, R
import pandas as pd
import ir_datasets
import ir_measures
import argparse
import time


prefix = {
    "intfloat/e5-base-v2": {
        "query": "query: ",
        "document": "passage: ",
    },
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
            "intfloat/e5-base-v2",
            "BAAI/bge-base-en-v1.5",
        ],
    )
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument(
        "--rerank", action="store_true", help="是否進行第二階段 reranking"
    )
    args = parser.parse_args()

    model = SentenceTransformer(
        args.model,
        device=args.device,
        prompts=prefix[args.model],
        default_prompt_name="query",
    )
    searcher = FaissSearcher(index_dir="fiqa", query_encoder=model)
    # test set
    test = ir_datasets.load("beir/fiqa/test")
    qrels = test.qrels_iter()

    queries = []
    qids = []
    for query in test.queries_iter():
        queries.append(query.text)
        qids.append(query.query_id)
    start = time.time()
    results = searcher.batch_search(queries, qids, k=100, threads=8)
    end = time.time()
    print(f"First-Stage Retrieval Time: {end - start:.2f} Seconds")

    run = pd.DataFrame(
        [
            {"query_id": qid, "doc_id": hit.docid, "score": hit.score}
            for qid, hits in results.items()
            for hit in hits
        ]
    )
    temp = ir_measures.calc_aggregate([nDCG @ 10, R @ 100], qrels, run)
    print("First-Stage Retrieval Results:")
    for k, v in temp.items():
        print(f"{k}: {v:.4f}")
    # 第二階段
    if args.rerank:
        reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device=args.device,
            model_kwargs={"torch_dtype": "bfloat16"},
        )
        store = test.docs_store()
        h = {}
        for query_id, text in zip(qids, queries):
            h[query_id] = text
        pairs = []
        for query_id, doc_id, score in run.itertuples(index=False):
            pairs.append((h[query_id], store.get(doc_id).text))
        scores = reranker.predict(pairs, show_progress_bar=True, batch_size=64)
        run["score"] = scores
        print(run.dtypes)
        run.sort_values(by=["query_id", "score"], ascending=[True, False], inplace=True)
        qrels = test.qrels_iter()
        temp = ir_measures.calc_aggregate([nDCG @ 10, R @ 100], qrels, run)
        for k, v in temp.items():
            print(f"{k}: {v:.4f}")
