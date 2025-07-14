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
        choices=[
            "scifact",
            "arguana",
            "fiqa",
            "dbpedia-entity",
            "trec-news",
            "climate-fever",
        ],
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

    results = searcher.batch_search(queries=queries, q_ids=qids, k=100, threads=16)

    with open(f"run.beir.{_model}.{args.dataset}.txt", "w") as f:
        for qid, hits in results.items():
            hits = [hit for hit in hits if hit.docid != qid]
            for i, hit in enumerate(hits):
                f.write(f"{qid} Q0 {hit.docid} {i + 1} {hit.score} Faiss\n")

    result = subprocess.run(
        f"python -m pyserini.eval.trec_eval   -c -m ndcg_cut.10 beir-v1.0.0-{args.dataset}-test   run.beir.{_model}.{args.dataset}.txt",
        shell=True,
        capture_output=True,
        text=True,
    )
    ndcg_cut_10 = float(result.stdout.split()[2])

    result = subprocess.run(
        f"python -m pyserini.eval.trec_eval   -c -m recall.100 beir-v1.0.0-{args.dataset}-test   run.beir.{_model}.{args.dataset}.txt",
        shell=True,
        capture_output=True,
        text=True,
    )
    recall_100 = float(result.stdout.split()[2])

    print(f"Dataset: {args.dataset}")
    print(f"nDCG@10: {ndcg_cut_10:.3f}\nRecall@100: {recall_100:.3f}")
