from tqdm import tqdm
import json
import os

datas = []
with open("scifact/corpus.jsonl", "rb") as f:
    for line in f:
        datas.append(json.loads(line))

os.makedirs("bm25_corpus", exist_ok=True)

with open("bm25_corpus/corpus.jsonl", "w") as f:
    for data in tqdm(datas):
        f.write(
            json.dumps(
                {
                    "id": data["_id"],
                    "query": data["title"],
                    #"contents": f'{data["title"]} {data["text"]}',
                    "contents": f'{data["text"]}',
                }
            )
            + "\n"
        )
