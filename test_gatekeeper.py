import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

# 1. Device and Model Setup (MobileNetV3 - Matching train_gatekeeper.py)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v3_small()
num_ftrs = model.classifier[3].in_features

# Exact architecture matching train structure (Dropout 0.5 + 4 Classes)
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

# 3. Prediction Logic
def predict_waste(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found -> {image_path}")
        return

    image = Image.open(image_path).convert('RGB')
    input_tensor = test_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prob, pred = torch.max(probabilities, 1)
    
    # Class order corresponds to Alphabetical Folder Structure
    classes = [
        'High Waste (Send Truck!) 🚛',
        'Invalid / Clean Place (Reject) ❌',
        'Low Waste (No Truck Needed) ⚠️',
        'Medium Waste 📦'
    ]
    
    confidence = prob.item() * 100
    predicted_class = classes[pred.item()]

    # Flag low confidence results
    if confidence < 45.0:
        status = "⚠️ Low Confidence (Manual Inspection Recommended)"
    else:
        status = "✅ Confident Prediction"

    print(f"\n📸 Photo: {image_path}")
    print(f"🤖 Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}% ({status})")

# Run prediction on a test image
test_image_name = 'test.jpg' 
predict_waste(test_image_name)