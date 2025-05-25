# 如何用 transformer 載入透過 lora fine tuned 的 deepseek-r1 model?

載入透過 LoRA fine-tuned 的 DeepSeek-R1 模型，你需要以下步驟和程式碼範例 (使用 `transformers` 庫):

**1. 安裝必要的庫:**

```bash
pip install transformers accelerate peft
```

*   `transformers`: Hugging Face 的 transformers 庫，用於載入和使用預訓練模型。
*   `accelerate`: 用於在不同硬體上加速訓練和推理，例如 GPU。
*   `peft`:  Parameter-Efficient Fine-Tuning (PEFT) 庫，其中包含 LoRA 相關的工具。

**2. 程式碼範例 (Python):**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 定義模型和 LoRA 適配器的路徑
model_name = "deepseek-ai/deepseek-coder-1.3b-base"  # 或者其他 DeepSeek-R1 的變體
lora_model_path = "your_lora_model_path"  # 替換成你的 LoRA 模型路徑

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 如果沒有 pad token，使用 eos token

# 載入 base 模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # 讓 accelerate 自動管理裝置
    trust_remote_code=True, # DeepSeek 模型可能需要
    torch_dtype="auto"  # 使用自動混合精度，可以提高效率
)

# 載入 LoRA 適配器
model = PeftModel.from_pretrained(model, lora_model_path)

# 設定模型為評估模式 (inference)
model.eval()

# 如果模型很大，可以嘗試合併 LoRA 權重到基礎模型 (僅用於推理，會修改模型權重)
# model = model.merge_and_unload()

# 推理示例
prompt = "def hello_world():\n  "  # 你的 prompt
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    generation_output = model.generate(
        input_ids=input_ids.input_ids,
        max_new_tokens=256,  # 設定生成 token 的最大數量
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.15,
        do_sample=True  # 啟用採樣，讓生成更具創造性
    )

output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
print(output)
```

**程式碼解釋:**

1.  **Import Libraries:** 導入必要的庫。
2.  **Define Paths:** 設定基礎模型名稱和 LoRA 模型路徑。  `model_name` 應該是 DeepSeek-R1 的正確模型 ID (例如 `deepseek-ai/deepseek-coder-1.3b-base`)。 `lora_model_path`  應該指向包含 `adapter_config.json` 和 `adapter_model.bin` 的目錄。
3.  **Load Tokenizer:** 載入與基礎模型匹配的 tokenizer。 重要的是確保 tokenizer 與用於訓練 LoRA 模型的 tokenizer 相同。 如果 tokenizer 沒有 pad token，則將其設置為 eos token。
4.  **Load Base Model:** 載入基礎的 DeepSeek-R1 模型。
    *   `device_map="auto"`: 讓 `accelerate` 庫自動管理模型在 CPU 和 GPU 之間的分配。
    *   `trust_remote_code=True`:  DeepSeek 模型可能需要這個參數，因為它們可能包含來自 Hub 的自定義程式碼。  **注意：** 在設置為 True 之前，請仔細檢查程式碼的安全性。
    *   `torch_dtype="auto"`:  讓 PyTorch 自動選擇最佳的資料類型（例如，float16 或 bfloat16），以獲得更好的效能。
5.  **Load LoRA Adapter:** 使用 `PeftModel.from_pretrained` 載入 LoRA 適配器。  這會將 LoRA 權重加載到基礎模型中。
6.  **Set to Evaluation Mode:**  `model.eval()` 將模型設置為評估模式。 這會禁用 dropout 和其他訓練時的行為。
7.  **Merge LoRA Weights (Optional):** 如果你只想進行推理，並且想要簡化模型，你可以使用 `model.merge_and_unload()` 將 LoRA 權重合併到基礎模型中。  **注意：** 這會修改模型的權重，並且你將無法再卸載 LoRA 適配器。  如果你的模型非常大，並且你想要節省記憶體，這可能是一個好主意。
8.  **Inference Example:**  展示如何使用模型生成文本。
    *   **Tokenize the prompt:** 使用 tokenizer 將你的 prompt 轉換為輸入 ID。
    *   **Generate text:** 使用 `model.generate()` 方法生成文本。  調整 `max_new_tokens`、`temperature`、`top_p`、`top_k` 和 `repetition_penalty` 等參數以控制生成文本的品質和創造性。  `do_sample=True` 啟用採樣。
    *   **Decode the output:** 使用 tokenizer 將生成的 token ID 轉換回文本。  `skip_special_tokens=True` 移除特殊 token，例如 pad token。

**重要注意事項:**

*   **模型相容性:** 確保 LoRA 模型與你使用的 DeepSeek-R1 模型變體相容。  如果 LoRA 模型是在 DeepSeek Coder 1.3B 上訓練的，則不應在 DeepSeek 7B 上使用它。
*   **LoRA 模型路徑:**  `lora_model_path` 必須指向包含 LoRA 適配器檔案的目錄。  通常，這個目錄包含 `adapter_config.json` 和 `adapter_model.bin` 檔案。
*   **記憶體:** DeepSeek-R1 模型可能很大，因此你可能需要使用 GPU 或啟用模型並行性才能成功載入和使用它。  `device_map="auto"` 有助於管理記憶體使用。
*   **`trust_remote_code=True`:**  當使用 `trust_remote_code=True` 時要小心，因為它允許從 Hugging Face Hub 執行任意程式碼。 在執行程式碼之前，請務必檢查程式碼的安全性。
*   **資料類型:** 使用 `torch_dtype="auto"` 允許 PyTorch 自動選擇最佳的資料類型，這可以提高效能。  你也可以嘗試使用 `torch_dtype=torch.float16` 或 `torch_dtype=torch.bfloat16` 以進一步減少記憶體使用。
*   **Tokenizer:** 確保你使用的 tokenizer 與用於訓練 LoRA 模型的 tokenizer 相同。
*   **參數調整:** 實驗 `model.generate()` 中的參數以獲得最佳結果。  `temperature` 控制隨機性，`top_p` 和 `top_k` 控制生成的多樣性，`repetition_penalty` 防止重複。

**例子：**

假設你已經在 DeepSeek Coder 1.3B 上微調了一個 LoRA 模型，並且你的 LoRA 模型儲存在 `./my_lora_model` 目錄中。  那麼程式碼會變成：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_name = "deepseek-ai/deepseek-coder-1.3b-base"
lora_model_path = "./my_lora_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(model, lora_model_path)
model.eval()

prompt = "def factorial(n):\n  \"\"\"Calculates the factorial of a number.\"\"\"\n  "
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    generation_output = model.generate(
        input_ids=input_ids.input_ids,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.15,
        do_sample=True
    )

output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
print(output)
```

請記住替換  `"./my_lora_model"`  為你的實際 LoRA 模型路徑。