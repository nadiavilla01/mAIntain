import os
import random
from PIL import Image, ImageDraw, ImageEnhance
from datetime import datetime


BASE_DIR = "synthetic_images"
IMAGE_SIZE = (224, 224)
NUM_IMAGES_PER_CLASS = 300



def generate_crack_image():
    img = Image.new("RGB", IMAGE_SIZE, (200, 200, 200))
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 3)):
        x1, y1 = random.randint(20, 200), random.randint(20, 200)
        x2, y2 = x1 + random.randint(-50, 50), y1 + random.randint(-50, 50)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=random.randint(1, 3))
    return img

def generate_rust_image():
    img = Image.new("RGB", IMAGE_SIZE, (180, 180, 180))
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(10, 30)):
        x, y = random.randint(0, 223), random.randint(0, 223)
        r = random.randint(5, 15)
        draw.ellipse((x, y, x + r, y + r), fill=(139, 69, 19))  
    return img

def generate_heat_image():
    img = Image.new("RGB", IMAGE_SIZE, (100, 100, 100))
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.5)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(3, 8)):
        x = random.randint(30, 180)
        y = random.randint(30, 180)
        radius = random.randint(10, 30)
        draw.ellipse((x, y, x+radius, y+radius), fill=(255, 0, 0))  # hot spot
    return img

def generate_wear_image():
    img = Image.new("RGB", IMAGE_SIZE, (160, 160, 160))
    draw = ImageDraw.Draw(img)
    for _ in range(5):
        x = random.randint(0, 200)
        y = random.randint(0, 200)
        w = random.randint(10, 20)
        h = random.randint(2, 5)
        draw.rectangle((x, y, x + w, y + h), fill=(100, 100, 100))
    return img

def generate_no_fault_image():
    return Image.new("RGB", IMAGE_SIZE, (220, 220, 220))

def generate_mixed_faults_image():
    img = generate_heat_image()
    draw = ImageDraw.Draw(img)
    draw.line((20, 20, 100, 100), fill=(0, 0, 0), width=2)
    return img


FAULT_GENERATORS = {
    "crack": generate_crack_image,
    "rust": generate_rust_image,
    "heat": generate_heat_image,
    "wear": generate_wear_image,
    "no_fault": generate_no_fault_image,
    "mixed_faults": generate_mixed_faults_image,
}


for fault_type, generator in FAULT_GENERATORS.items():
    folder = os.path.join(BASE_DIR, fault_type)
    os.makedirs(folder, exist_ok=True)
    
    existing = len(os.listdir(folder))
    print(f"🧪 Generating for: {fault_type} (currently {existing} files)")

    for i in range(NUM_IMAGES_PER_CLASS):
        img = generator()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{fault_type}_{existing + i + 1}_{timestamp}.png"
        path = os.path.join(folder, filename)
        img.save(path)

print("Synthetic image generation complete.")