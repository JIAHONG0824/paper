import ir_datasets
import subprocess
import random
import json
import os

dataset = ir_datasets.load("beir/fiqa")
print(dataset.docs_count())
corpus = {}
for doc in dataset.docs_iter():
    corpus[doc.doc_id] = doc.text

with open("DE.json", "r") as f:
    DE = json.load(f)

os.makedirs("corpus", exist_ok=True)

encoder_name = "facebook/contriever-msmarco"

nums = 20
for seed in range(42, 46):
    random.seed(seed)
    with open("corpus/corpus.jsonl", "w") as f:
        for id, d in corpus.items():
            if id not in DE:
                f.write(json.dumps({"id": id, "text": d}) + "\n")
                continue
            queries = DE[id]
            random.shuffle(queries)
            text = queries[:nums] + [d]
            f.write(json.dumps({"id": id, "text": "\n".join(text)}) + "\n")
        command = [
            "python",
            "-m",
            "pyserini.encode",
            "input",
            "--corpus",
            "corpus",
            "--fields",
            "text",
            "output",
            "--embeddings",
            f"contriever-msmarco/fiqa/head/{nums}-{seed}",
            "--to-faiss",
            "encoder",
            "--encoder",
            encoder_name,
            "--fields",
            "text",
            "--batch",
            "64",
            "--fp16",
            "--max-length",
            "512",
            "--device",
            "cuda:1",
        ]
        subprocess.run(command)
