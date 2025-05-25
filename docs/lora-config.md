# 如果使用 Lora 搭配 transformer train deepseek-r1 1.5B model 會 killed，我可以如何調整 LoraConfig 或其他參數?

當你使用 LoRA 搭配 Transformer 訓練 DeepSeek-R1 1.5B 模型時遇到 OOM (Out of Memory) 被 killed 的問題，這通常表示你的 GPU 記憶體不足。以下是一些你可以調整 LoRAConfig 和其他參數的策略來解決這個問題：

**1. LoRAConfig 調整：**

*   **`r` (LoRA rank):**  這是最直接影響記憶體佔用的參數。  **減小 `r` 的值可以顯著減少 LoRA 權重的數量，從而降低記憶體使用量。**  可以從一個較小的值開始嘗試，例如 8 或 4，然後逐步增加，直到找到一個平衡點，既能提供足夠的性能，又不會超出記憶體限制。
*   **`lora_alpha`:**  這個參數影響 LoRA 的縮放比例。  **通常不需要調整 `lora_alpha` 來解決 OOM 問題。**  它主要影響訓練的穩定性和收斂速度。  可以保持預設值 (通常是 `r`)。
*   **`lora_dropout`:**  這個參數用於 LoRA 的 dropout。  **通常不需要調整 `lora_dropout` 來解決 OOM 問題。**  它主要用於正則化，防止過擬合。  可以保持預設值 (通常是 0.0)。
*   **`target_modules`:**  仔細選擇需要應用 LoRA 的模塊。  **只針對關鍵的 Transformer 層 (例如 attention 和 MLP 層) 應用 LoRA，可以有效減少參數數量。**  可以嘗試只針對 `q_proj`, `k_proj`, `v_proj`, `o_proj` 和 `gate_proj`, `down_proj`, `up_proj` 應用 LoRA。  可以使用 `print(model)` 來查看模型的結構，以確定哪些模塊最適合應用 LoRA。
*   **`bias`:**  這個參數控制是否訓練 bias。  **如果記憶體非常緊張，可以考慮將 `bias` 設置為 `"none"`，不訓練 bias。**  但這可能會略微降低性能。

**LoRAConfig 範例調整：**

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,  # 降低 rank
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "down_proj", "up_proj"], # 只針對關鍵層
    bias="none", # 如果記憶體非常緊張
    task_type="CAUSAL_LM",
)
```

**2. 其他參數調整：**

*   **Batch Size:**  **這是最直接影響記憶體使用的參數。 減小 batch size 可以顯著減少記憶體佔用。**  可以從一個較小的 batch size 開始 (例如 1 或 2)，然後逐步增加，直到達到記憶體限制。
*   **Gradient Accumulation Steps:**  **增加 gradient accumulation steps 可以模擬更大的 batch size，而無需增加單次迭代的記憶體使用量。**  例如，如果你的 batch size 是 2，gradient accumulation steps 是 4，則相當於 batch size 為 8。
*   **Mixed Precision Training (fp16/bf16):**  **使用混合精度訓練可以減少記憶體使用量。**  `fp16` (float16) 和 `bf16` (bfloat16) 都需要 Ampere 架構或更新的 GPU。  使用 `torch.cuda.amp.autocast` 或 Transformers 的 `Trainer` 中的 `fp16` 或 `bf16` 參數來啟用混合精度訓練。
*   **Gradient Checkpointing:**  **啟用 gradient checkpointing 可以減少記憶體使用量，但會增加計算時間。**  Gradient checkpointing 會在反向傳播過程中重新計算某些激活值，而不是將它們儲存在記憶體中。  可以使用 `model.gradient_checkpointing_enable()` 或 Transformers 的 `Trainer` 中的 `gradient_checkpointing` 參數來啟用 gradient checkpointing。
*   **Optimizer:**  **嘗試使用更節省記憶體的優化器，例如 AdamW 或 Lion。**  AdamW 是一種常用的優化器，通常比 Adam 更穩定。  Lion 是一種更新的優化器，據稱比 AdamW 更節省記憶體。
*   **Data Loading:**  **確保你的數據加載是高效的，避免在訓練過程中不必要地將大量數據加載到記憶體中。**  使用數據迭代器和分批處理來加載數據。
*   **Offload Model to CPU:**  **如果你的 CPU 記憶體足夠大，可以考慮將部分模型或中間激活值 offload 到 CPU。**  這會顯著降低訓練速度，但可以讓你訓練更大的模型。  可以使用 Accelerate 庫來實現 offload。
*   **Quantization:**  **訓練前或訓練後對模型進行量化可以減少模型大小和記憶體使用量。**  可以使用 bitsandbytes 庫進行量化。  例如，可以使用 8-bit 或 4-bit 量化。

**程式碼範例 (Transformers Trainer):**

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="path/to/your/output",
    per_device_train_batch_size=2,  # 減小 batch size
    gradient_accumulation_steps=4,  # 增加 gradient accumulation steps
    fp16=True,  # 使用 fp16
    # bf16=True, # 可以嘗試使用 bf16，如果你的 GPU 支持
    gradient_checkpointing=True, # 啟用 gradient checkpointing
    optim="adamw_torch", # 或者 "lion"
    # 其他訓練參數
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # 其他訓練參數
)

trainer.train()
```

**Debug 技巧:**

*   **使用 `torch.cuda.memory_summary()` 監控 GPU 記憶體使用情況。**  在訓練過程中定期打印記憶體使用情況，可以幫助你找出記憶體瓶頸。
*   **使用 `nvidia-smi` 命令監控 GPU 使用情況。**  `nvidia-smi` 可以顯示 GPU 的記憶體使用量、利用率等信息。
*   **逐步嘗試不同的參數組合。**  一次只調整一個參數，然後觀察其對記憶體使用和性能的影響。

**總結：**

解決 LoRA 訓練 OOM 問題的關鍵是找到一個平衡點，既能滿足記憶體限制，又能提供足夠的性能。  從減小 `r` 和 batch size 開始，然後逐步嘗試其他技巧，例如 gradient accumulation steps、混合精度訓練和 gradient checkpointing。  使用 debug 工具監控記憶體使用情況，並逐步調整參數，直到找到最佳配置。  记住，不同的模型和数据集可能需要不同的配置。