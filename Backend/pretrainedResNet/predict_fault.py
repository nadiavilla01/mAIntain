import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import sys
import os


MODEL_PATH = "fault_classifier_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    'crack',
    'rust',
    'wear',
    'heat',
    'no_fault',
    'mixed_faults',
    'dataset_sample'
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


if len(sys.argv) < 2:
    print("❌ Please provide an image path to classify.")
    sys.exit()

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print(f"❌ File not found: {img_path}")
    sys.exit()

image = Image.open(img_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(DEVICE)


print("📦 Loading model...")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=True)
model.to(DEVICE)
model.eval()


with torch.no_grad():
    outputs = model(image_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item()


print("\n📸 Image Classified:")
print(f"   🖼️ File: {img_path}")
print(f"   🔍 Predicted Class: {CLASSES[pred_idx]}")
print(f"   📊 Confidence: {confidence:.4f}")