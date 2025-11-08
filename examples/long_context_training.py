# Train with 128K context
from transformers import AutoModelForCausalLM, AutoTokenizer
from mem_safe import mem_safe_forward
import torch

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Simulate 128K input
text = "Hello " * 64000  # ~128K tokens
inputs = tokenizer(text, return_tensors="pt", truncation=False)
x = inputs["input_ids"]

# Forward with mem-safe
with torch.enable_grad():
    hidden = mem_safe_forward(model, x, use_checkpoint=True, dynamic_chunk=True, verbose=True)
    # Continue with loss, backward...
    print(f"Output shape: {hidden.shape}")
