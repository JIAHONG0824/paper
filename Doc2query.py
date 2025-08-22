from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["Qwen/Qwen3-4B"])
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fiqa"],
    )
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    llm = LLM(model=args.model)
    sampling_params = SamplingParams(
        n=20,
        temperature=1.0,
        max_tokens=128,
    )

    prompts = []
    with open(f"{args.dataset}/corpus.jsonl", "rb") as f:
        for line in f:
            info = json.loads(line)
            id, title, text = info["_id"], info["title"], info["text"]
            if text == "":
                print(f"Skipping empty doc: {id}")
                continue
            prompts.append(
                (
                    id,
                    [
                        {
                            "role": "user",
                            "content": f"Please write a question to answer the passage\nPassage:{title}\n{text}",
                        },
                        {"role": "assistant", "content": "Question:"},
                    ],
                )
            )
    h = {}
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]
        ids = [prompt[0] for prompt in batch_prompts]
        outputs = llm.chat(
            [prompt[1] for prompt in batch_prompts],
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        for id, output in zip(ids, outputs):
            generated_texts = [item.text for item in output.outputs]
            seen = set()
            DE = []
            for text in generated_texts:
                if text not in seen:
                    DE.append(text)
                    seen.add(text)
            h[id] = DE
    with open(f"{args.dataset}-DE.json", "w") as f:
        json.dump(h, f, indent=4, ensure_ascii=False)
    print(f"Saved to {args.dataset}-DE.jsonl")
