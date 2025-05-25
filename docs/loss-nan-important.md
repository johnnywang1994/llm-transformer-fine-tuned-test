你的程式架構已經很完整，針對微調少量資料、LoRA 設定與 `loss: nan` 問題的處理都很細緻。不過你遇到 `loss: nan` 的情況，這裡有幾點可進一步優化與排錯：

---

## ✅ 核心問題排查建議

### 🔍 1. **`tokenizer.as_target_tokenizer()` 在新版 Transformers 已棄用**

你目前這段可能導致意料外的 token 處理：

```python
with tokenizer.as_target_tokenizer():
    labels = tokenizer(...)
```

**建議：直接去掉，使用 tokenizer 處理輸出即可。**

```python
labels = tokenizer(
    outputs,
    max_length=128,
    padding="max_length",
    truncation=True
)["input_ids"]
```

---

### 🔍 2. **檢查是否有「全 -100」的 label**

這會造成 `loss` 無法計算（空的有效 token）。你可以加上以下 debug：

```python
for label in labels:
    if all(l == -100 for l in label):
        print("Warning: label all -100!")
```

如果你發現有 `prompt` 跟 `completion` 太短，導致 label 全部是 padding，請改短 `max_length` 或做 input 長度檢查。

---

### 🔍 3. **模型預設為 float32，但 LoRA 初始化容易不穩定（尤其在 CPU）**

建議你 **開啟 gradient checkpointing**：

```python
model.gradient_checkpointing_enable()
```

---

### 🔍 4. **Token重複部分沒處理好，導致學不到東西**

你目前是這樣 tokenize：

```python
inputs = examples["prompt"]
outputs = examples["completion"]
```

但在 LLM 中，最好是把 input/output 合併為單一序列格式，例如：

```
<|user|>: 你的 prompt
<|assistant|>: 你的 completion
```

然後只在 `completion` 的部分設為 label。

---

## ✅ 建議最佳格式（訓練格式與標記）

```python
def tokenize_function(examples):
    prompts = examples["prompt"]
    completions = examples["completion"]
    inputs = [f"<|user|>: {q}\n<|assistant|>: {a}" for q, a in zip(prompts, completions)]

    model_inputs = tokenizer(
        inputs,
        max_length=256,
        padding="max_length",
        truncation=True
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
            max_length=256,
            padding="max_length",
            truncation=True
        )["input_ids"]
        label_ids = [(id if id != tokenizer.pad_token_id else -100) for id in label_ids]
        labels.append(label_ids)

    model_inputs["labels"] = labels
    return model_inputs
```

---

## ✅ 微調檢查點整理

| 檢查項目                    | 建議                                  |
| ----------------------- | ----------------------------------- |
| Learning rate           | `5e-6` ✅                            |
| Batch size              | `1` ✅                               |
| Epochs                  | `10+` ✅                             |
| Gradient clipping       | `max_grad_norm=1.0` ✅               |
| Loss mask               | 確保 `-100` 有設定                       |
| Label 非空                | `label` 中不要全是 `-100`                |
| Tokenizer padding token | 有設 `pad_token` ✅                    |
| LoRA target\_modules    | 有設 `q_proj`, `k_proj` 等 ✅           |
| Model mode              | `gradient_checkpointing_enable()` ✅ |

---

## 🔧 Bonus: 訓練進度監控小技巧

可在 `Trainer` 加上 `callbacks` 來監控 loss 演變（甚至早停）：

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

---

