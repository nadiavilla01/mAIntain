import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


MODEL_PATH = "fault_classifier_resnet18.pth"
IMAGE_PATH = "synthetic_images/rust/rust_1.png"  # Change 
SAVE_PATH = "gradcampp_output.png"
LABELS = ['crack', 'dataset_sample', 'heat', 'mixed_faults', 'no_fault', 'rust', 'wear']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


original_img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(original_img).unsqueeze(0).to(DEVICE)


model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


grads = None
features = None

def save_grad(grad):
    global grads
    grads = grad

def forward_hook(module, input, output):
    global features
    features = output
    output.register_hook(save_grad)

target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(forward_hook)


output = model(input_tensor)
predicted_class = output.argmax(dim=1).item()
print(f" Predicted: {LABELS[predicted_class]}")


model.zero_grad()
class_score = output[0, predicted_class]
class_score.backward(retain_graph=True)


grads_np = grads.cpu().data.numpy()[0]        
features_np = features.cpu().data.numpy()[0]  


grad_2 = grads_np ** 2
grad_3 = grad_2 * grads_np
sum_activations = np.sum(features_np, axis=(1, 2))

eps = 1e-8
alpha_num = grad_2
alpha_denom = 2 * grad_2 + sum_activations[:, None, None] * grad_3
alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, eps)

alphas = alpha_num / alpha_denom
weights = np.sum(alphas * np.maximum(grads_np, 0), axis=(1, 2))


cam = np.sum(weights[:, None, None] * features_np, axis=0)
cam = np.maximum(cam, 0)
cam = cam / cam.max()
cam = cv2.resize(cam, (224, 224))


heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
original_np = np.array(original_img.resize((224, 224)))
overlayed = np.uint8(heatmap * 0.4 + original_np)

cv2.imwrite(SAVE_PATH, overlayed)
print(f" Grad-CAM++ saved to {SAVE_PATH}")