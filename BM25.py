from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics
from tqdm import tqdm
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    searcher = LuceneSearcher.from_prebuilt_index(
        prebuilt_index_name=f"beir-v1.0.0-{args.dataset}.flat", verbose=True
    )
    topics = get_topics(f"beir-v1.0.0-{args.dataset}-test")

    queries = []
    qids = []
    for qid in topics:
        qids.append(str(qid))
        queries.append(topics[qid]["title"])

    results = searcher.batch_search(
        queries=queries, qids=qids, k=args.top_k, threads=16
    )
    
    with open(f"run.beir.bm25-flat.{args.dataset}.txt", "w") as f:
        for qid, hits in tqdm(results.items()):
            for rank, doc in enumerate(hits):
                if qid == doc.docid:
                    continue
                f.write(f"{qid} Q0 {doc.docid} {rank+1} {doc.score} Anserini\n")

    commands = [
        f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.{args.ndcg_cutoff} beir-v1.0.0-{args.dataset}-test run.beir.bm25-flat.{args.dataset}.txt",
        f"python -m pyserini.eval.trec_eval -c -m recall.{args.recall_cutoffs[0]} beir-v1.0.0-{args.dataset}-test run.beir.bm25-flat.{args.dataset}.txt",
        f"python -m pyserini.eval.trec_eval -c -m recall.{args.recall_cutoffs[1]} beir-v1.0.0-{args.dataset}-test run.beir.bm25-flat.{args.dataset}.txt",
    ]
    for command in commands:
        print(
            subprocess.run(command, shell=True, capture_output=True, text=True).stdout
        )
