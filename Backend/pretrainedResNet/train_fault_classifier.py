import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os


DATA_DIR = "./synthetic_images"
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "fault_classifier_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f" Found {len(dataset)} images across {len(dataset.classes)} classes: {dataset.classes}")


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))
model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f" Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")


torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n Model saved to {MODEL_SAVE_PATH}")


model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=dataset.classes))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))