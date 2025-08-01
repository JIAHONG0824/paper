from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AutoQueryEncoder
from pyserini.search import get_topics
from tqdm import tqdm
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
        choices=["trec-covid", "nfcorpus", "fiqa", "scifact"],
    )
    parser.add_argument(
        "--top_k", type=int, default=100, help="Number of hits to return."
    )
    parser.add_argument("--ndcg_cutoff", type=int, default=10, help="nDCG@k")
    parser.add_argument(
        "--recall_cutoffs",
        nargs=2,
        type=int,
        default=[100, 1000],
        help="Recall@k1 and Recall@k2 cutoffs.",
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
    q_ids = []
    for qid in topics:
        q_ids.append(str(qid))
        queries.append(topics[qid]["title"])

    results = searcher.batch_search(
        queries=queries, q_ids=q_ids, k=args.top_k, threads=16
    )

    with open(f"run.beir.{_model}.{args.dataset}.txt", "w") as f:
        for qid, hits in tqdm(results.items()):
            hits = [hit for hit in hits if hit.docid != qid]
            for rank, hit in enumerate(hits):
                f.write(f"{qid} Q0 {hit.docid} {rank+1} {hit.score} Faiss\n")

    commands = [
        f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.{args.ndcg_cutoff} beir-v1.0.0-{args.dataset}-test run.beir.{_model}.{args.dataset}.txt",
        f"python -m pyserini.eval.trec_eval -c -m recall.{args.recall_cutoffs[0]} beir-v1.0.0-{args.dataset}-test run.beir.{_model}.{args.dataset}.txt",
        f"python -m pyserini.eval.trec_eval -c -m recall.{args.recall_cutoffs[1]} beir-v1.0.0-{args.dataset}-test run.beir.{_model}.{args.dataset}.txt",
    ]
    for command in commands:
        print(
            subprocess.run(command, shell=True, capture_output=True, text=True).stdout
        )
