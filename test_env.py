from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # lighter
model = AutoModel.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Fake news detection is interesting!", return_tensors="pt")
outputs = model(**inputs)
print("DistilBERT output shape:", outputs.last_hidden_state.shape)
print("✅ Script finished successfully")