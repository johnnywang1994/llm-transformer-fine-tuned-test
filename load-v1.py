from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./deepseek-finetuned"

model = AutoModelForCausalLM.from_pretrained(
  model_path,
  # torch_dtype=torch.float16,
  # device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_text(prompt, max_new_tokens=100):
  # If NVIDIA driver is on your system
  # inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
  inputs = tokenizer(prompt, return_tensors="pt")
  print(inputs)

  with torch.no_grad():
    output = model.generate(
      **inputs,
      max_new_tokens=max_new_tokens,
      do_sample=False # 先用貪婪解碼測試
      # do_sample=True,
      # temperature=0.5,
      # top_k=50,
      # top_p=0.9,
      # use_cache=True
    )
  print(output)

  return tokenizer.decode(output[0], skip_special_tokens=True)

# Test
prompt = "請告訴我你是誰?自我介紹你自己"
output = generate_text(prompt)
print(output)