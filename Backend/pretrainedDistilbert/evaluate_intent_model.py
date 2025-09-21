import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader


csv_path = "./intent_dataset_distilbert.csv"
fmea_path = "./FMEA.csv"
model_path = "./distilbert_intent_model"


df = pd.read_csv(csv_path)
df["label"] = df["intent"].astype("category").cat.codes
label2id = dict(zip(df["intent"].astype("category").cat.categories, range(len(df["intent"].unique()))))
id2label = {v: k for k, v in label2id.items()}


fmea_df = pd.read_csv(fmea_path)
if "maintenance_alert" not in label2id:
    label2id["maintenance_alert"] = len(label2id)
    id2label[label2id["maintenance_alert"]] = "maintenance_alert"

fmea_texts = [
    f"{row['failure_mode']} caused by {row['cause']} can result in {row['suggested_action']}."
    for _, row in fmea_df.iterrows()
]
fmea_labels = [label2id["maintenance_alert"]] * len(fmea_texts)
fmea_df = pd.DataFrame({"text": fmea_texts, "label": fmea_labels})
df = pd.concat([df[["text", "label"]], fmea_df], ignore_index=True)


_, test_texts, _, test_labels = train_test_split(df["text"], df["label"], test_size=0.2, stratify=df["label"])


class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

test_dataset = IntentDataset(test_texts.tolist(), test_labels.tolist())
test_loader = DataLoader(test_dataset, batch_size=8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()


all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())


print("🔍 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=[id2label[i] for i in sorted(id2label)]))

print("🧩 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))