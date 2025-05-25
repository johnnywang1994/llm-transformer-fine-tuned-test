# 用 transformer 訓練 deepseek-r1 fine tune 的 jsonl dataset 範例，並且透過 load_dataset 載入的相關實作方式

當你想用 Transformer 來 fine-tune DeepSeek-R1 模型，且資料格式是 jsonl（每行一個 JSON 物件）時，通常資料會包含輸入（input text）和目標（target text）欄位。下面我會示範一個簡單的 jsonl 範例，並且教你如何用 `datasets.load_dataset` 來載入這個資料集。

---

## 1. jsonl 範例檔案 (data.jsonl)

```jsonl
{"input": "What is the capital of France?", "target": "The capital of France is Paris."}
{"input": "Explain the process of photosynthesis.", "target": "Photosynthesis is the process by which green plants use sunlight to synthesize foods from carbon dioxide and water."}
{"input": "Who wrote 'Pride and Prejudice'?", "target": "Jane Austen wrote 'Pride and Prejudice'."}
```

---

## 2. 透過 `datasets.load_dataset` 載入 jsonl

```python
from datasets import load_dataset

# 假設你的 data.jsonl 在當前目錄
dataset = load_dataset("json", data_files="data.jsonl")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['input', 'target'],
#         num_rows: 3
#     })
# })
```

---

## 3. 進一步示範如何用此資料做 fine-tune

這裡以 Hugging Face Transformers 的 Trainer API 為例示範：

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

# 載入 tokenizer 和模型 (以 DeepSeek-R1 為例，請替換成實際模型名稱)
model_name = "DeepSeek-R1"  # 假設模型在 HF hub 有此名稱
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 資料預處理函式
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset["train"].map(preprocess_function, batched=True)

# 設定訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

---

## 小結

1. jsonl 格式每行是一個 JSON 物件，包含 `input` 與 `target` 欄位。
2. 用 `datasets.load_dataset("json", data_files="data.jsonl")` 載入資料。
3. 使用 tokenizer 對 `input` 和 `target` 分別編碼，並將 target 編碼放入 `labels`。
4. 使用 Trainer 或其他訓練方式進行 fine-tune。

---

如果你需要我幫你寫更完整的程式碼或針對 DeepSeek-R1 特殊需求調整，請隨時告訴我！