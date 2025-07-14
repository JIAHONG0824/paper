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

    topics = get_topics(f"beir-v1.0.0-{args.dataset}-test")
    searcher = LuceneSearcher.from_prebuilt_index(f"beir-v1.0.0-{args.dataset}.flat")
    queries = []
    qids = []
    for qid in topics:
        queries.append(topics[qid]["title"])
        qids.append(str(qid))
    results = searcher.batch_search(queries=queries, qids=qids, k=100, threads=32)
    with open(f"run.beir.bm25-flat.{args.dataset}.txt", "w") as f:
        for qid, hits in tqdm(results.items()):
            hits = [hit for hit in hits if hit.docid != qid]
            for i, hit in enumerate(hits):
                f.write(f"{qid} Q0 {hit.docid} {i + 1} {hit.score} Anserini\n")

    result = subprocess.run(
        f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 beir-v1.0.0-{args.dataset}-test run.beir.bm25-flat.{args.dataset}.txt",
        shell=True,
        capture_output=True,
        text=True,
    )
    ndcg_cut_10 = float(result.stdout.split()[2])
    result = subprocess.run(
        f"python -m pyserini.eval.trec_eval -c -m recall.100 beir-v1.0.0-{args.dataset}-test run.beir.bm25-flat.{args.dataset}.txt",
        shell=True,
        capture_output=True,
        text=True,
    )
    recall_100 = float(result.stdout.split()[2])
    print(f"Dataset: {args.dataset}")
    print(f"nDCG@10: {ndcg_cut_10:.3f}\nRecall@100: {recall_100:.3f}")
