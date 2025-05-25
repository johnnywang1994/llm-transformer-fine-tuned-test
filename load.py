from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 定義模型和 LoRA 適配器的路徑
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 或者其他 DeepSeek-R1 的變體
lora_model_path = "./deepseek-finetuned"  # 替換成你的 LoRA 模型路徑

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token  # 如果沒有 pad token，使用 eos token

# 載入 base 模型
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  trust_remote_code=True, # DeepSeek 模型可能需要
  torch_dtype="auto"  # 使用自動混合精度，可以提高效率
)

# 載入 LoRA 適配器
model = PeftModel.from_pretrained(model, lora_model_path)

# 設定模型為評估模式 (inference)
model.eval().to(model.device)

# 如果模型很大，可以嘗試合併 LoRA 權重到基礎模型 (僅用於推理，會修改模型權重)
# model = model.merge_and_unload()

def generate_text(prompt, max_new_tokens=128):
  # If NVIDIA driver is on your system
  # inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
  inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
  print(inputs)

  with torch.no_grad():
    output = model.generate(
      **inputs,
      max_new_tokens=max_new_tokens, # 設定生成 token 的最大數量
      temperature=0.7,
      do_sample=True, # 啟用採樣，讓生成更具創造性
      # do_sample=False # 先用貪婪解碼測試
      # top_k=50,
      # top_p=0.9,
      # use_cache=True
    )
  print(output)

  return tokenizer.decode(output[0], skip_special_tokens=True)

# Test
prompt = "你喜歡交朋友嗎?是否對交朋友感到樂此不疲?"
output = generate_text(prompt)
print(output)