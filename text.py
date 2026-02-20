from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import json

prefix = {
    "facebook/contriever-msmarco": None,
    "BAAI/bge-base-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages:",
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["facebook/contriever-msmarco", "BAAI/bge-base-en-v1.5"],
    )
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    model = SentenceTransformer(args.model, device="cuda:1", prompts=prefix[args.model])

    with open(args.input_jsonl, "r") as f, open(args.output_jsonl, "w") as out:
        for line in tqdm(f):
            item = json.loads(line)
            doc_id = item["id"]
            original_document = item["document"]
            generated_queries = item["generated_queries"]

            # 若沒有 queries，直接輸出空集合
            if not generated_queries:
                out.write(
                    json.dumps(
                        {
                            "id": doc_id,
                            "document": original_document,
                            "generated_queries": [],
                        }
                    )
                    + "\n"
                )
                continue

            S = []
            doc_cur = original_document

            # anchor: 所有 query embedding 的平均
            anchor = model.encode_query(generated_queries, convert_to_tensor=True).mean(
                dim=0
            )

            # baseline: 原始 document 對 anchor 的相似度
            d_emb = model.encode_document(doc_cur, convert_to_tensor=True)
            cur_best = model.similarity(d_emb, anchor)
            # 保險：轉成 Python float（避免 tensor 布林比較問題）
            cur_best = cur_best

            # 做法 B：用 remaining 維護還沒選的 index（不需要 visited）
            remaining = list(range(len(generated_queries)))

            while remaining:
                # 只對 remaining 候選做評分
                temp = [f"{doc_cur}\n{generated_queries[i]}" for i in remaining]
                temp_embs = model.encode_document(temp, convert_to_tensor=True)
                score = model.similarity(temp_embs, anchor)

                best_pos = score.argmax()
                mx = remaining[best_pos]  # 原始 index
                best_score = score[best_pos]

                if best_score <= cur_best:
                    break

                cur_best = best_score
                doc_cur = f"{doc_cur}\n{generated_queries[mx]}"
                S.append(generated_queries[mx])

                # swap-pop：O(1) 移除 remaining[best_pos]
                remaining[best_pos] = remaining[-1]
                remaining.pop()

            out.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "document": original_document,
                        "generated_queries": S,
                    }
                )
                + "\n"
            )
