from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AutoQueryEncoder
from pyserini.search import get_topics
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["facebook/contriever", "facebook/contriever-msmarco"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["trec-covid", "trec-news", "scifact"],
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of top documents to retrieve for each query",
    )
    parser.add_argument(
        "--ndcg_k",
        type=int,
        default=10,
        help="Depth for nDCG calculation (default: 10)",
    )
    parser.add_argument(
        "--recall_k",
        type=int,
        default=100,
        help="Depth for recall calculation (default: 100)",
    )
    args = parser.parse_args()

    if args.model == "facebook/contriever":
        _model = "contriever"
    else:
        _model = "contriever-msmarco"

    topics = get_topics(f"beir-v1.0.0-{args.dataset}-test")
    encoder = AutoQueryEncoder(encoder_dir=args.model, pooling="mean", device="cuda:1")
    searcher = FaissSearcher.from_prebuilt_index(
        f"beir-v1.0.0-{args.dataset}.{_model}", query_encoder=encoder
    )

    queries = []
    qids = []
    for qid in topics:
        queries.append(topics[qid]["title"])
        qids.append(str(qid))

    results = searcher.batch_search(queries=queries, q_ids=qids, k=args.k, threads=16)

    with open(f"run.beir.{_model}.{args.dataset}.txt", "w") as f:
        for qid, hits in results.items():
            hits = [hit for hit in hits if hit.docid != qid]
            for i, hit in enumerate(hits):
                f.write(f"{qid} Q0 {hit.docid} {i + 1} {hit.score} Faiss\n")

    result = subprocess.run(
        f"python -m pyserini.eval.trec_eval   -c -m ndcg_cut.{args.ndcg_k} beir-v1.0.0-{args.dataset}-test   run.beir.{_model}.{args.dataset}.txt",
        shell=True,
        capture_output=True,
        text=True,
    )
    ndcg_cut_10 = float(result.stdout.split()[2]) * 100

    result = subprocess.run(
        f"python -m pyserini.eval.trec_eval   -c -m recall.{args.recall_k} beir-v1.0.0-{args.dataset}-test   run.beir.{_model}.{args.dataset}.txt",
        shell=True,
        capture_output=True,
        text=True,
    )
    recall_100 = float(result.stdout.split()[2]) * 100

    print(f"Dataset: {args.dataset}")
    print(f"nDCG@{args.ndcg_k}: {ndcg_cut_10:.1f}\nR@{args.recall_k}: {recall_100:.1f}")
