from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AutoQueryEncoder
from tqdm import tqdm
import ir_datasets
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        choices=[
            "facebook/contriever",
            "facebook/contriever-msmarco",
            "BAAI/bge-base-en-v1.5",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fiqa", "scifact"],
    )
    parser.add_argument(
        "--top_k", type=int, default=1000, help="Number of hits to return."
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

    model_map = {
        "facebook/contriever": "contriever",
        "facebook/contriever-msmarco": "contriever-msmarco",
        "BAAI/bge-base-en-v1.5": "bge-base-en-v1.5",
    }
    if "contriever" in args.encoder:
        encoder = AutoQueryEncoder(
            encoder_dir=args.encoder, device="cuda:1", pooling="mean", l2_norm=False
        )
    else:
        encoder = AutoQueryEncoder(
            encoder_dir=args.encoder,
            device="cuda:1",
            pooling="cls",
            l2_norm=True,
            prefix="Represent this sentence for searching relevant passages:",
        )
    searcher = FaissSearcher.from_prebuilt_index(
        f"beir-v1.0.0-{args.dataset}.{model_map[args.encoder]}", query_encoder=encoder
    )

    dataset = ir_datasets.load(f"beir/{args.dataset}/test")

    queries = []
    q_ids = []
    for query in dataset.queries_iter():
        queries.append(query.text)
        q_ids.append(query.query_id)

    results = searcher.batch_search(
        queries=queries, q_ids=q_ids, k=args.top_k, threads=8
    )

    with open(f"run.beir.{model_map[args.encoder]}.{args.dataset}.txt", "w") as f:
        for qid, hits in tqdm(results.items()):
            hits = [hit for hit in hits if hit.docid != qid]
            for rank, hit in enumerate(hits):
                f.write(f"{qid} Q0 {hit.docid} {rank+1} {hit.score} Faiss\n")

    commands = [
        f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.{args.ndcg_cutoff} beir-v1.0.0-{args.dataset}-test run.beir.{model_map[args.encoder]}.{args.dataset}.txt",
        f"python -m pyserini.eval.trec_eval -c -m recall.{args.recall_cutoffs[0]} beir-v1.0.0-{args.dataset}-test run.beir.{model_map[args.encoder]}.{args.dataset}.txt",
        f"python -m pyserini.eval.trec_eval -c -m recall.{args.recall_cutoffs[1]} beir-v1.0.0-{args.dataset}-test run.beir.{model_map[args.encoder]}.{args.dataset}.txt",
    ]
    for command in commands:
        print(
            subprocess.run(command, shell=True, capture_output=True, text=True).stdout
        )
