from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import ir_datasets
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    searcher = LuceneSearcher.from_prebuilt_index(
        prebuilt_index_name=f"beir-v1.0.0-{args.dataset}.flat", verbose=True
    )
    dataset = ir_datasets.load(f"beir/{args.dataset}/test")

    queries = []
    qids = []
    for query in dataset.queries_iter():
        queries.append(query.text)
        qids.append(query.query_id)

    results = searcher.batch_search(queries=queries, qids=qids, k=args.top_k, threads=8)

    with open(f"run.beir.bm25-flat.{args.dataset}.txt", "w") as f:
        for qid, hits in tqdm(results.items()):
            hits = [hit for hit in hits if hit.docid != qid]
            for rank, hit in enumerate(hits):
                f.write(f"{qid} Q0 {hit.docid} {rank+1} {hit.score} Anserini\n")

    commands = [
        f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.{args.ndcg_cutoff} beir-v1.0.0-{args.dataset}-test run.beir.bm25-flat.{args.dataset}.txt",
        f"python -m pyserini.eval.trec_eval -c -m recall.{args.recall_cutoffs[0]} beir-v1.0.0-{args.dataset}-test run.beir.bm25-flat.{args.dataset}.txt",
        f"python -m pyserini.eval.trec_eval -c -m recall.{args.recall_cutoffs[1]} beir-v1.0.0-{args.dataset}-test run.beir.bm25-flat.{args.dataset}.txt",
    ]
    for command in commands:
        print(
            subprocess.run(command, shell=True, capture_output=True, text=True).stdout
        )
