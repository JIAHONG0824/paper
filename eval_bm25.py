from pyserini.search.lucene import LuceneSearcher
import ir_datasets
import subprocess
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    searcher = LuceneSearcher(index_dir=args.index_dir)
    test = ir_datasets.load(f"beir/{args.dataset}/test")

    queries = []
    qids = []
    for query in test.queries_iter():
        queries.append(query.text)
        qids.append(query.query_id)

    results = searcher.batch_search(queries, qids, k=1000, threads=8)
    with open("results.text", "w") as f:
        for qid, hits in results.items():
            hits = [hit for hit in hits if hit.docid != qid]
            for rank, hit in enumerate(hits):
                f.write(f"{qid} Q0 {hit.docid} {rank+1} {hit.score} Anserini\n")
    commands = [
        f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 beir-v1.0.0-{args.dataset}-test results.text",
        f"python -m pyserini.eval.trec_eval -c -m recall.100 beir-v1.0.0-{args.dataset}-test results.text",
        f"python -m pyserini.eval.trec_eval -c -m recall.1000 beir-v1.0.0-{args.dataset}-test results.text",
    ]
    for command in commands:
        subprocess.run(command, shell=True)
