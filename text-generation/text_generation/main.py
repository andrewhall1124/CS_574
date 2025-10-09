from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare the prompt
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)