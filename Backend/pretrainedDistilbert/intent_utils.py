import os
import csv
import torch
from datetime import datetime
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


CONFIDENCE_THRESHOLD = 0.75
MODEL_PATH = "./distilbert_intent_model"
LOG_FILE = "intent_logs.csv"


os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to("cpu")
model.eval()


_logged_cache = set()


def infer_intent(query: str):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()

    id2label = model.config.id2label
    intent = id2label[predicted]
    confidence = probs[0][predicted].item()

    # === Fallback Handling ===
    fallback = "yes" if confidence < CONFIDENCE_THRESHOLD else "no"
    if fallback == "yes":
        intent = "intent_uncertain"

    # === Log intent history ===
    log_intent_history(query, intent, confidence, fallback)

    return intent, confidence


def log_intent_history(text, intent, confidence, fallback):
    global _logged_cache
    key = (text.strip(), intent)
    if key in _logged_cache:
        return  # Avoid repeated identical logs
    _logged_cache.add(key)

    timestamp = datetime.utcnow().isoformat()
    row = [timestamp, text.strip(), intent, confidence, fallback]

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "text", "predicted_intent", "confidence", "fallback"])
        writer.writerow(row)