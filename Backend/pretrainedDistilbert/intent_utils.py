import os
import csv
import torch
from datetime import datetime
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# === CONFIG ===
CONFIDENCE_THRESHOLD = 0.75
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "distilbert_intent_model")
LOG_FILE = os.path.join(BASE_DIR, "intent_logs.csv")

# === ENVIRONMENT ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === LOAD MODEL ===
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to("cpu")
model.eval()

# === CACHE TO AVOID REDUNDANT LOGGING ===
_logged_cache = set()


def infer_intent(query: str):
    """
    Classifies the intent from user query using DistilBERT.
    Applies confidence-based fallback and logs the result.
    """
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
    """
    Logs classified intents with timestamp to a CSV file,
    avoiding duplicate consecutive entries.
    """
    global _logged_cache
    key = (text.strip(), intent)
    if key in _logged_cache:
        return
    _logged_cache.add(key)

    timestamp = datetime.utcnow().isoformat()
    row = [timestamp, text.strip(), intent, confidence, fallback]

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "text", "predicted_intent", "confidence", "fallback"])
        writer.writerow(row)
