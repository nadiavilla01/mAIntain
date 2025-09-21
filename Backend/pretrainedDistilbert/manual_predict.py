# manual_predict.py
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.nn.functional import softmax
import torch
import pandas as pd
from datetime import datetime
import os

model_path = "./distilbert_intent_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model.eval()

# === INPUT ===
text = input("💬 Enter query: ")
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# === CONFIDENCE ===
probs = softmax(outputs.logits, dim=1)
confidence = probs.max().item()
predicted = outputs.logits.argmax().item()
intent = model.config.id2label[predicted]

# === FALLBACK CHECK ===
CONFIDENCE_THRESHOLD = 0.75
fallback = "yes" if confidence < CONFIDENCE_THRESHOLD else "no"
if fallback == "yes":
    intent = "intent_uncertain"

# === PRINT RESULT ===
print(f"🧠 Predicted intent: {intent}")
print(f"🔎 Confidence: {confidence:.4f}")
if fallback == "yes":
    print("⚠️  Low confidence — fallback triggered")

# === LOG TO CSV ===
log_path = "intent_logs.csv"
entry = {
    "timestamp": datetime.now().isoformat(),
    "text": text,
    "predicted_intent": intent,
    "confidence": confidence,
    "fallback": fallback
}

# Append or create log file
df = pd.DataFrame([entry])
if os.path.exists(log_path):
    df.to_csv(log_path, mode='a', header=False, index=False)
else:
    df.to_csv(log_path, mode='w', header=True, index=False)
