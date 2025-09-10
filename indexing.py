from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ir_datasets
import subprocess
import argparse
import json
import os

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
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--head",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    encoder = SentenceTransformer(
        model_name_or_path=args.encoder, device="cuda:1", similarity_fn_name="dot"
    )

    dataset = ir_datasets.load(f"beir/{args.dataset}")
    print("docs_count:", dataset.docs_count())
    corpus = []
    for doc in dataset.docs_iter():
        try:
            content = f"{doc.title}\n{doc.text}"
        except AttributeError:
            content = doc.text
        if not content:
            continue
        corpus.append((doc.doc_id, content))
    print("non-empty docs:", len(corpus))

    with open("all-t5-base-v1-scifact-querygen20-1.json", "r") as f:
        DE = json.load(f)
    os.makedirs("corpus", exist_ok=True)

    batch_size = 64
    with open("corpus/corpus.jsonl", "w") as f:
        for i in tqdm(range(0, len(corpus), batch_size)):
            batch_texts = [item[1] for item in corpus[i : i + batch_size]]
            batch_ids = [item[0] for item in corpus[i : i + batch_size]]

            for i, (id, text) in enumerate(zip(batch_ids, batch_texts)):
                queries = []
                if id in DE:
                    queries = DE[id]
                if args.head:
                    batch_texts[i] = "\n".join(queries[: args.k] + [text])
                elif args.tail:
                    batch_texts[i] = "\n".join([text] + queries[: args.k])
                else:
                    batch_texts[i] = "\n".join([text])

            embeddings = encoder.encode_document(
                sentences=batch_texts, batch_size=batch_size, device="cuda:1"
            )
            # 2. 在記憶體中準備好整個批次的輸出內容
            lines_to_write = [
                json.dumps({"id": doc_id, "vector": emb.tolist()})
                for doc_id, emb in zip(batch_ids, embeddings)
            ]

            # 3. 將準備好的多行內容用換行符連接，再一次性寫入檔案
            f.write("\n".join(lines_to_write) + "\n")
    command = [
        "python",
        "-m",
        "pyserini.index.faiss",
        "--input",
        "corpus",
        "--output",
        f"{args.encoder.split('/')[-1]}-{args.dataset}-{args.head}-{args.tail}-{args.k}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
