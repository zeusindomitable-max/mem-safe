# Inference 128K context
from transformers import AutoModelForCausalLM, AutoTokenizer
from mem_safe import mem_safe_forward

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model.eval()

text = "Summarize: " + "long text " * 50000
inputs = tokenizer(text, return_tensors="pt", truncation=False)

with torch.no_grad():
    output = mem_safe_forward(model, inputs["input_ids"], use_checkpoint=False, verbose=True)
    print("Inference successful on 128K+ context!")
