from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "bert-base-uncased"

print("⏳ Downloading model and tokenizer from Hugging Face...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained("model/bert-base-uncased")
model.save_pretrained("model/bert-base-uncased")

print("✅ Model and tokenizer saved to: model/bert-base-uncased/")
