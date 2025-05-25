from transformers import pipeline

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

pipe = pipeline("text-generation", model=model_name)
messages = [
    {"role": "user", "content": "如何用 transformers 對 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 微調，具體LoraConfig 和 TrainingArguments 該怎麼配置?"},
]

output = pipe(messages, max_length=100, num_return_sequences=1)
print(output)