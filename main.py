from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_name = "deepseek-ai/deepseek-coder-1.3b-base"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()
print(model)
for name, module in model.named_modules():
    if "proj" in name:
        print(name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token  # 確保 tokenizer 有 pad_token，對於 causal LM 很重要

# 載入資料集
data_files = {"train": "data.jsonl"}
dataset = load_dataset('json', data_files=data_files)

# 過擬合小資料的練習
# dataset["train"] = dataset["train"].select([0, 7] * 4)  # 複製 100 次

# Tokenize
def tokenize_function(examples):
    prompts = examples["prompt"]
    completions = examples["completion"]
    inputs = [f"<|user|>: {q}\n<|assistant|>: {a}" for q, a in zip(prompts, completions)]

    tokenized_inputs = tokenizer(
        inputs,  # 假設你的資料集中文本欄位叫做 "text"
        max_length=128,       # 設定最大長度 (根據你的模型和硬體調整)
        padding="max_length",  # 不足最大長度時 padding 到最大長度
        truncation=True,      # 超過最大長度時截斷
        # return_tensors="np" # 返回 numpy array (如果是 PyTorch 則使用 "pt")
    )

    # 只在 <|assistant|> 之後部分建立 labels
    labels = []
    for input_text in inputs:
        user_split = input_text.split("<|assistant|>: ")
        if len(user_split) != 2:
            raise ValueError("格式錯誤，缺少 <|assistant|>: ")
        assistant_text = user_split[1]
        label_ids = tokenizer(
            assistant_text,
            max_length=128,
            padding="max_length",
            truncation=True
        )["input_ids"]
        # 將 padding token id 設成 -100
        # tokenize_function 會將 label padding 到最大長度，這會導致 loss 計算時把 padding 區域也算進去。對於 causal LM，labels 中 padding token id 應設為 -100，這樣 loss 計算時會忽略這些位置。
        label_ids = [(id if id != tokenizer.pad_token_id else -100) for id in label_ids]
        labels.append(label_ids)

    # 檢查是否有「全 -100」的 label, 會造成 loss 無法計算（空的有效 token）
    for label in labels:
        if all(l == -100 for l in label):
            print("Warning: label all -100!")
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function, remove_columns=["prompt", "completion"], batched=True)

# 設定 LoRA
lora_config = LoraConfig(
    r=4, # 4, 8
    lora_alpha=16, # 8, 32
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 依模型結構調整，可以加上 print(model) 來確認。
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# 訓練參數
training_args = TrainingArguments(
    output_dir="./deepseek-finetuned",
    per_device_train_batch_size=1, # 若記憶體不足可使用 1 or 2, <= 3 時容易 loss: 0
    num_train_epochs=10, # 訓練資料(資料夠多可設定 1-3，資料少需要更大比如 100)
    gradient_accumulation_steps=1, # 資料少
    logging_steps=1, # 設 1，可以每步都看到 loss。
    # save_steps=100, # 不需要每 100步存，否則會存很多空模型。
    save_total_limit=1,
    learning_rate=5e-6, # learning_rate=2e-4 對於 LoRA+小模型（如 1.5B）有時會過大，導致 loss 爆炸，生成時就會出 nan/inf。嘗試降低學習率 5e-5 或 1e-4, 1e-5, 5e-6
    max_grad_norm=1.0, # gradient clipping: 觀察 loss，有沒有變成 nan 或 inf？有的話就要直接 early stop
    fp16=False,  # 若無 GPU 可改成 False
    bf16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()

model.save_pretrained("./deepseek-finetuned")
tokenizer.save_pretrained("./deepseek-finetuned")

# 如果 loss 可以降到 0.1 以下，理論上模型能「背」住這兩筆資料。