import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer

# --- 模型與 Tokenizer 設定 ---
train_dataset_dir = "fiqa-train-DPO.jsonl"
dev_dataset_dir = "fiqa-dev-DPO.jsonl"
model_name_or_path = "doc2query/all-t5-base-v1"

peft_config = LoraConfig(
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q", "k", "v", "o", "wi", "wo"],
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("正在載入訓練資料集...")
train_dataset = load_dataset("json", data_files=train_dataset_dir)["train"]
dev_dataset = load_dataset("json", data_files=dev_dataset_dir)["train"]


# 【新增】定義一個函數來添加任務前綴
def preprocess_function(examples):
    # 在這裡定義您想用的前綴
    # task_prefix = "generate Question: "  # 您可以自訂這個前綴
    examples["prompt"] = [p for p in examples["prompt"]]
    return examples


# 【新增】使用 .map() 來應用這個前綴
print("正在為資料集添加任務前綴...")
train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)
print(f"訓練集大小: {len(train_dataset)}")
print(f"驗證集大小: {len(dev_dataset)}")
print("--- 資料集範例 ---")
print(train_dataset[0])
print("--------------------")
# --- 訓練參數 ---
training_args = DPOConfig(
    output_dir="all-t5-base-v1-DPO",  # 建議換個名字以區分
    learning_rate=1e-5,
    per_device_train_batch_size=8,  # 從 8 或 4 開始可能更安全
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    do_predict=True,
    bf16=torch.cuda.is_available(),
    save_total_limit=3,
    logging_steps=10,  # 建議增加日誌步數，方便觀察 loss
    max_prompt_length=448,
    max_completion_length=64,
    logging_dir="./logs",
    report_to=["tensorboard"],
    warmup_steps=100,
)
# --- 初始化 DPOTrainer ---
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

# --- 開始訓練 ---
print("\n--- 開始 DPO 訓練 ---")
trainer.train()
print("\n--- DPO 訓練結束 ---")
