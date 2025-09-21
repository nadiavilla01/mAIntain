import os
from uuid import uuid4
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
import numpy as np

router = APIRouter()


BASE_DIR = os.path.dirname(__file__)

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "../../fault_classifier_resnet18.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ['crack', 'dataset_sample', 'heat', 'mixed_faults', 'no_fault', 'rust', 'wear']


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded fault classifier from {MODEL_PATH}")
except Exception as e:
    print(f" Could not load model at {MODEL_PATH}: {e}")

model.to(DEVICE)
model.eval()


def save_gradcam(img_pil, input_tensor, target_class, save_path):
    """
    Saves Grad-CAM overlay to save_path (PNG).
    Uses model.layer4[1].conv2 as the target layer.
    """
    features = None
    grads = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    target_layer = model.layer4[1].conv2
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        outputs = model(input_tensor)  
        model.zero_grad()
        outputs[0, target_class].backward()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        original_np = np.array(img_pil.resize((224, 224)))
        original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        overlayed = cv2.addWeighted(heatmap, 0.4, original_bgr, 0.6, 0)

        cv2.imwrite(save_path, overlayed)
    finally:
        h1.remove()
        h2.remove()


@router.post("/predict-fault")
async def predict_fault(file: UploadFile = File(...)):
    try:
        # Load image
        img = Image.open(file.file).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_idx = outputs.argmax(dim=1).item()
            pred_label = LABELS[pred_idx]
            pred_conf = probs[pred_idx].item()

        filename = f"gradcam_{uuid4().hex}.png"
        cam_path = os.path.join(STATIC_DIR, filename)
        save_gradcam(img, input_tensor, pred_idx, cam_path)

        return JSONResponse({
            "predicted_class": pred_label,
            "confidence": round(float(pred_conf), 3),
            "gradcam_image": f"/static/{filename}"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)