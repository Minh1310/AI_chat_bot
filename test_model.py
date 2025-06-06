from transformers import pipeline

# Tải mô hình TinyLLaMA
model_name = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
generator = pipeline("text-generation", model=model_name)

# Test mô hình
prompt = "Bạn là một nhân viên bán hàng thân thiện. Hãy trả lời: Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"
response = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
print(response)