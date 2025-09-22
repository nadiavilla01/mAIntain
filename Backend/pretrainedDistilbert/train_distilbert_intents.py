import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm
import json

csv_path = "./intent_dataset_distilbert.csv"
fmea_path = "./FMEA.csv"
model_path = "./distilbert_intent_model"

df = pd.read_csv(csv_path)
df["label"] = df["intent"].astype("category").cat.codes

label2id = dict(zip(df["intent"].astype("category").cat.categories, range(len(df["intent"].unique()))))
id2label = {v: k for k, v in label2id.items()}

fmea_df = pd.read_csv(fmea_path)
fmea_texts = [
    f"{row['failure_mode']} caused by {row['cause']} can result in {row['suggested_action']}."
    for _, row in fmea_df.iterrows()
]

fmea_labels = ["maintenance_alert"] * len(fmea_texts)

if "maintenance_alert" not in label2id:
    new_id = len(label2id)
    label2id["maintenance_alert"] = new_id
    id2label[new_id] = "maintenance_alert"

aug_df = pd.DataFrame({
    "text": fmea_texts,
    "intent": fmea_labels,
    "label": [label2id["maintenance_alert"]] * len(fmea_texts)
})

df = pd.concat([df, aug_df], ignore_index=True)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"]
)

train_dataset = IntentDataset(train_texts.tolist(), train_labels.tolist())
test_dataset = IntentDataset(test_texts.tolist(), test_labels.tolist())

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=4)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
epochs = 20

model.train()
for epoch in range(epochs):
    print(f" Epoch {epoch + 1}")
    model.train()
    running_train_loss = 0.0

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f" Epoch {epoch + 1} avg train loss: {avg_train_loss:.4f}")

    # 
    # model.eval()
    # running_val_loss = 0.0
    # with torch.no_grad():
    #     for batch in val_loader:
    #         input_ids = batch["input_ids"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)
    #         labels = batch["labels"].to(device)
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #         running_val_loss += outputs.loss.item()
    # avg_val_loss = running_val_loss / len(val_loader)
    # val_losses.append(avg_val_loss)
    # print(f"Epoch {epoch + 1} avg val loss: {avg_val_loss:.4f}")

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(" Model saved to ./distilbert_intent_model")

with open(f"{model_path}/label2id.json", "w") as f:
    json.dump(label2id, f, indent=2)
with open(f"{model_path}/id2label.json", "w") as f:
    json.dump(id2label, f, indent=2)

print(f"\n Training losses: {train_losses}")
# print(f" Validation losses: {val_losses}")  # Uncomment if using val loop
