
from transformers import AutoModelForCausalLM
print("imported")

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
print("got model")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left", max_new_tokens=64)

model_inputs = tokenizer(["write 1/2 in latex:"], return_tensors="pt",).to("mps")
print("got tokenizer")

generated_ids = model.generate(**model_inputs, max_length= 64)
print("generated")

text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("decoded; response:")
print(text)


