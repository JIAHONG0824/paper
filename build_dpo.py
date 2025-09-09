import numpy as np
import ir_datasets
import random
import json
from collections import defaultdict
from pyserini.encode import AutoQueryEncoder, AutoDocumentEncoder
from tqdm import tqdm
from transformers import T5Tokenizer

device = "cuda:1"

q_encoder = AutoQueryEncoder(
    encoder_dir="facebook/contriever", device=device, pooling="mean", l2_norm=False
)
d_encoder = AutoDocumentEncoder(
    model_name="facebook/contriever", device=device, pooling="mean", l2_norm=False
)
dataset = ir_datasets.load("beir/fiqa/dev")

tokenizer = T5Tokenizer.from_pretrained("doc2query/all-t5-base-v1")

doc_map = {}
for doc in dataset.docs_iter():
    if len(tokenizer.encode(doc.text, truncation=True, max_length=512)) <= 79:
        continue
    doc_map[doc.doc_id] = doc.text

query_map = {}
for query in dataset.queries_iter():
    query_map[query.query_id] = query.text

qrel_map = defaultdict(list)
for qrel in dataset.qrels_iter():
    qrel_map[qrel.query_id].append(qrel.doc_id)

# DE = {}
with open("DE-deduplicated.json", "r") as f:
    DE = json.load(f)
    # for line in f:
    #     item = json.loads(line)
    #     DE[item["id"]] = item["querygen"]


with open("fiqa-dev-DPO.jsonl", "w") as f:
    for qid, doc_ids in tqdm(qrel_map.items()):
        q = query_map[qid]
        q_emb = q_encoder.encode(q)
        for doc_id in doc_ids:
            if doc_id not in doc_map:
                continue
            doc = doc_map[doc_id]
            doc_emb = d_encoder.encode(doc, max_length=512)[0]
            baseline = np.dot(q_emb, doc_emb)
            queries = DE[doc_id]
            new_docs = [f"{query}\n{doc}" for query in queries]
            new_docs_embs = d_encoder.encode(new_docs, max_length=512)

            cur_max = float("-inf")
            chosen = None
            rejected = []
            for i, (query, new_doc_emb) in enumerate(zip(queries, new_docs_embs)):
                score = np.dot(q_emb, new_doc_emb)
                if score > baseline:
                    if score > cur_max:
                        cur_max = score
                        chosen = query
                else:
                    rejected.append(query)
            if not chosen:
                chosen = q
            if not rejected:
                rejected.append(q)
            random.shuffle(rejected)
            for i in range(len(rejected[:10])):
                f.write(
                    json.dumps(
                        {"prompt": doc, "chosen": chosen, "rejected": rejected[i]}
                    )
                    + "\n"
                )
