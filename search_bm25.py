from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import ir_datasets
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    searcher = LuceneSearcher(index_dir="BM25")

    dataset = ir_datasets.load(f"beir/{args.dataset}/test")

    queries = []
    qids = []
    for query in dataset.queries_iter():
        queries.append(query.text)
        qids.append(query.query_id)

    results = searcher.batch_search(queries, qids, k=1000, threads=8)

    with open(f"test.txt", "w") as f:
        for qid, hits in tqdm(results.items()):
            hits = [hit for hit in hits if hit.docid != qid]
            for rank, hit in enumerate(hits):
                f.write(f"{qid} Q0 {hit.docid} {rank+1} {hit.score} Anserini\n")

    commands = [
        f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 beir-v1.0.0-{args.dataset}-test test.txt",
        f"python -m pyserini.eval.trec_eval -c -m recall.100 beir-v1.0.0-{args.dataset}-test test.txt",
        f"python -m pyserini.eval.trec_eval -c -m recall.1000 beir-v1.0.0-{args.dataset}-test test.txt",
    ]
    for command in commands:
        subprocess.run(command, shell=True)
