# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader
# import os

# # 1. Enhanced Data Augmentation to prevent Overfitting
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(20),
#         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
#         transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# data_dir = 'ml_waste_system'

# image_datasets = {
#     x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
#     for x in ['train', 'val']
# }

# dataloaders = {
#     x: DataLoader(image_datasets[x], batch_size=16, shuffle=True) 
#     for x in ['train', 'val']
# }

# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes 
# print(f"Recognized Classes: {class_names}")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Training Device: {device}")

# # 2. Model Architecture with Dropout (0.5)
# model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# num_ftrs = model.classifier[3].in_features
# model.classifier[3] = nn.Sequential(
#     nn.Dropout(0.5),
#     nn.Linear(num_ftrs, 4)
# )
# model = model.to(device)

# # Weighted Loss: Focus more on strictly identifying 'invalid_or_clean' to stop false positives
# # Index order: [high_waste, invalid_or_clean, low_waste, medium_waste]
# class_weights = torch.tensor([1.0, 1.8, 1.0, 1.0]).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights)

# # Stronger Weight Decay (1e-3) to explicitly fix overfitting
# optimizer = optim.Adam(model.parameters(), lr=0.00015, weight_decay=1e-3)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', patience=2, factor=0.5
# )

# # 3. Optimized Training Loop
# num_epochs = 20
# print("\n--- Low-Overfitting Anti-False-Positive Training Started ---")

# best_acc = 0.0

# for epoch in range(num_epochs):
#     print(f"\nEpoch {epoch+1}/{num_epochs}")
#     print("-" * 25)

#     for phase in ['train', 'val']:
#         if phase == 'train':
#             model.train()
#         else:
#             model.eval()

#         running_loss = 0.0
#         running_corrects = 0

#         for inputs, labels in dataloaders[phase]:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()

#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)

#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / dataset_sizes[phase]
#         epoch_acc = running_corrects.double() / dataset_sizes[phase]

#         print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#         if phase == 'val':
#             scheduler.step(epoch_loss)

#             if epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 torch.save(model.state_dict(), 'gatekeeper_model.pth')
#                 print(f"🌟 New Best Model Saved! Acc: {best_acc:.4f}")

# print(f"\n🎯 Highest Validation Accuracy Achieved: {best_acc*100:.2f}%")
# print("✅ Best Model saved successfully as gatekeeper_model.pth")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# Set seed for reproducible results (Optionally guaranteed consistency)
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 1. Enhanced Data Augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'ml_waste_system'

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
    for x in ['train', 'val']
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=16, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=16, shuffle=False) # Evaluation clean/consistent हुन्छ
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes 
print(f"Recognized Classes: {class_names}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training Device: {device}")

# 2. Model Architecture
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 4)
)
model = model.to(device)

# Standard Loss: Standard CrossEntropy for Balanced Dataset
criterion = nn.CrossEntropyLoss()

# Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=0.00015, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5
)

# 3. Training Loop
num_epochs = 25
print("\n--- Balanced Dataset Training Started ---")

best_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 25)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if phase == 'val':
            scheduler.step(epoch_loss)

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'gatekeeper_model.pth')
                print(f"🌟 New Best Model Saved! Acc: {best_acc:.4f}")

print(f"\n🎯 Highest Validation Accuracy Achieved: {best_acc*100:.2f}%")
print("✅ Best Model saved successfully as gatekeeper_model.pth")