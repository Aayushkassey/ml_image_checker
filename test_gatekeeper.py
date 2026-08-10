import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

# 1. Device and Model Setup (MobileNetV3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v3_small()
num_ftrs = model.classifier[3].in_features

# Exact architecture matching train structure
model.classifier[3] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 4)
)

# Load trained weights
model_path = 'gatekeeper_model.pth'
if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Model weights loaded successfully!")
else:
    print(f"❌ Error: Model file '{model_path}' not found. Please train first.")

model = model.to(device)
model.eval()

# 2. Test Transforms
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Base Output Folder Name and Categories
BASE_OUTPUT_DIR = 'test_result'
CATEGORY_FOLDERS = ['High_Waste', 'Not_Waste', 'Low_Waste', 'Medium_Waste']

# 📁 Ensure test_result and all 4 sub-folders exist beforehand
def init_result_structure():
    for sub_dir in CATEGORY_FOLDERS:
        full_dir = os.path.join(BASE_OUTPUT_DIR, sub_dir)
        os.makedirs(full_dir, exist_ok=True)

init_result_structure()

# 3. Prediction & Auto-Sorting Logic
def predict_and_save_waste(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found -> {image_path}")
        return

    try:
        original_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"⚠️ Skip: Unable to process file '{image_path}' -> {e}")
        return

    input_tensor = test_transforms(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prob, pred = torch.max(probabilities, 1)

    # Class order matching PyTorch ImageFolder (Alphabetical order)
    classes = [
        'High Waste (Send Truck!) 🚛',
        'Invalid / Clean Place (Reject) ❌',
        'Low Waste (No Truck Needed) ⚠️',
        'Medium Waste 📦'
    ]

    confidence = prob.item() * 100
    idx = pred.item()
    predicted_class = classes[idx]
    target_folder = CATEGORY_FOLDERS[idx]

    status = "⚠️ Low Confidence" if confidence < 45.0 else "✅ Confident Prediction"

    print(f"\n📸 Photo: {image_path}")
    print(f"🤖 Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}% ({status})")

    # Save logic with confidence name format
    target_dir_path = os.path.join(BASE_OUTPUT_DIR, target_folder)
    base_name = os.path.basename(image_path).rsplit('.', 1)[0]
    output_filename = f"{base_name}_{confidence:.1f}percent.jpg"
    output_path = os.path.join(target_dir_path, output_filename)

    original_image.save(output_path)
    print(f"📁 Saved to: {output_path}")


# 4. Multi-Format Directory/File Smart Resolver
def run_test_pipeline(source_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    if os.path.isdir(source_path):
        print(f"📂 Processing directory: '{source_path}'")
        image_files = [
            f for f in os.listdir(source_path) 
            if f.lower().endswith(valid_extensions)
        ]

        if not image_files:
            print("⚠️ No supported image files found in the directory!")
            return

        for img_file in image_files:
            full_path = os.path.join(source_path, img_file)
            predict_and_save_waste(full_path)

    elif os.path.isfile(source_path):
        predict_and_save_waste(source_path)

    else:
        found = False
        for ext in valid_extensions:
            possible_path = source_path + ext
            if os.path.isfile(possible_path):
                print(f"🔍 Auto-detected image file: '{possible_path}'")
                predict_and_save_waste(possible_path)
                found = True
                break

        if not found:
            print(f"❌ Error: No image found matching '{source_path}'")


# 🚀 चलाउने ठाउँ:
target_input = 'test.jpg'  # वा 'test_folder'
run_test_pipeline(target_input)